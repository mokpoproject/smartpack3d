"""
vision.py - ArUco 마커 기준 기하 측정 모듈 (OpenCV)

핵심 원리
---------
1. 사진 한 장만으로는 cm 크기를 알 수 없으므로, 크기를 아는 ArUco 마커를 기준 물체로 씁니다.
2. 마커의 4개 꼭짓점으로 호모그래피를 구해 촬영 평면을 정면(수직)으로 펴 줍니다(PlaneRectifier).
3. 펴진 이미지에서 마커가 아닌 가장 큰 윤곽을 물체로 보고 cm 단위로 측정합니다(ObjectSegmenter).
4. 촬영 조건(마커 크기, 기울기, 윤곽 품질)으로 신뢰도를 계산합니다.

사진 종류
---------
- 상면 사진(top): 바닥에 물건과 마커를 같은 평면에 놓고 위에서 촬영 -> 가로, 깊이
- 측면 사진(side): 물건과 마커를 옆에서 촬영 -> 높이
  (정면 사진으로는 깊이를 알 수 없으므로 '정면'을 '위에서 본 평면 사진'으로 정의합니다.)

한계
----
- 물체 윤곽은 임계값 기반 간이 분할입니다(그림자, 배경 대비에 민감). 객체 분할 모델은 후속 과제입니다.
- 물체 높이가 있으면 위에서 본 상면이 카메라에 가까워 실제보다 크게 보일 수 있습니다.
- 측면 사진의 높이는 마커가 똑바로 세워져 있다고 가정합니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MAX_IMAGE_SIDE = 1600        # 처리 속도를 위해 이미지 긴 변을 이 크기 이하로 줄입니다.
MIN_MARKER_PX = 40.0         # 마커 한 변의 최소 픽셀 길이. 이보다 작으면 측정을 거부합니다.
GOOD_MARKER_PX = 100.0       # 이 이상이면 마커 해상도가 충분하다고 봅니다.
MIN_DIM_CM = 0.5             # 이보다 작은 치수는 비현실적으로 보고 버립니다.
MAX_DIM_CM = 150.0           # 이보다 큰 치수는 비현실적으로 보고 버립니다.
MAX_CANVAS_PX = 3000         # 평면 보정 결과 이미지의 최대 한 변 픽셀 수.
MIN_OBJECT_CM2 = 4.0         # 이보다 작은 윤곽은 물체로 보지 않습니다.
MAX_OBJECT_AREA_RATIO = 0.8  # 보정 이미지의 이 비율보다 큰 윤곽은 배경으로 봅니다.

# 사용 가능한 ArUco 사전 이름 -> OpenCV 상수 (UI 선택지로 사용)
ARUCO_DICTS: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


# --------------------------------------------------------------------------- #
# 데이터 구조
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarkerDetection:
    """감지된 ArUco 마커 한 개. corners는 이미지 좌표 (4, 2), 순서는 TL, TR, BR, BL."""

    marker_id: int
    corners: np.ndarray

    @property
    def side_px(self) -> float:
        """마커 네 변의 평균 길이(px)."""
        pts = self.corners
        return float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))

    def max_angle_deviation_deg(self) -> float:
        """인접한 변들이 이루는 각도가 90도에서 얼마나 벗어났는지(최대값, 도)."""
        pts = self.corners
        devs = []
        for i in range(4):
            a = pts[(i - 1) % 4] - pts[i]
            b = pts[(i + 1) % 4] - pts[i]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            devs.append(abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 90.0))
        return max(devs)

    def skew_ratio(self) -> float:
        """네 변 길이의 최소/최대 비율. 1.0에 가까울수록 정면에 가깝습니다."""
        pts = self.corners
        lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        return float(min(lengths) / (max(lengths) + 1e-9))


@dataclass
class ObjectBlob:
    """평면 보정 이미지에서 찾은 물체 윤곽 정보(좌표는 보정 캔버스 기준 px)."""

    contour: np.ndarray
    area_px: float
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]  # cv2.minAreaRect 결과
    bbox: Tuple[int, int, int, int]                              # x, y, w, h
    solidity: float                                              # 면적 / 볼록껍질 면적
    touches_border: bool                                         # 캔버스 가장자리에 닿았는지


@dataclass
class PhotoMeasurement:
    """사진 한 장의 측정 결과."""

    ok: bool
    dims_cm: Dict[str, float] = field(default_factory=dict)  # top: width/depth, side: height
    marker_found: bool = False
    marker_id: Optional[int] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    preview_bgr: Optional[np.ndarray] = None  # 측정 검증용 오버레이 이미지

    @classmethod
    def failed(cls, error: str, marker_found: bool = False) -> "PhotoMeasurement":
        return cls(ok=False, error=error, marker_found=marker_found)


@dataclass
class ItemEstimate:
    """여러 사진을 합친 최종 추정 결과. 항상 사용자 확인이 필요합니다(needs_review=True)."""

    width: Optional[float] = None
    depth: Optional[float] = None
    height: Optional[float] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    needs_review: bool = True

    @property
    def has_any_dim(self) -> bool:
        return any(v is not None for v in (self.width, self.depth, self.height))

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.75:
            return "높음"
        if self.confidence >= 0.5:
            return "보통"
        return "낮음"


# --------------------------------------------------------------------------- #
# 이미지 입출력 유틸
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> np.ndarray:
    """업로드/카메라 바이트를 BGR 이미지로 바꿉니다. 실패하면 ValueError."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # EXIF 회전 정보도 자동 적용됩니다.
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. JPG/PNG 사진인지 확인해 주세요.")
    h, w = img.shape[:2]
    scale = MAX_IMAGE_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def roi_to_corners(roi: Tuple[float, float, float, float], shape: Tuple[int, ...]) -> np.ndarray:
    """정규화된 ROI(x0, y0, x1, y1, 0~1)를 이미지 픽셀 사각형 꼭짓점 (4, 2)으로 바꿉니다."""
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    return np.array(
        [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# 마커 검출
# --------------------------------------------------------------------------- #
class MarkerDetector:
    """ArUco 마커를 찾는 클래스."""

    def __init__(self, dictionary_id: int = cv2.aruco.DICT_4X4_50) -> None:
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id),
            cv2.aruco.DetectorParameters(),
        )

    def detect(self, image_bgr: np.ndarray) -> List[MarkerDetection]:
        """이미지에서 보이는 모든 마커를 찾습니다."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [
            MarkerDetection(int(mid), c.reshape(4, 2).astype(np.float32))
            for c, mid in zip(corners, ids.flatten())
        ]

    def find(self, image_bgr: np.ndarray, marker_id: Optional[int] = None) -> Optional[MarkerDetection]:
        """기준 마커 한 개를 고릅니다. marker_id가 있으면 그 번호만, 없으면 가장 큰 마커."""
        dets = self.detect(image_bgr)
        if marker_id is not None:
            dets = [d for d in dets if d.marker_id == marker_id]
        if not dets:
            return None
        return max(dets, key=lambda d: d.side_px)


# --------------------------------------------------------------------------- #
# 평면 보정
# --------------------------------------------------------------------------- #
class PlaneRectifier:
    """마커가 놓인 평면을 정면으로 펴서 cm 단위 캔버스를 만드는 클래스.

    호모그래피 H는 마커 평면 위의 점만 정확하게 변환합니다.
    따라서 물체는 마커와 같은 평면(같은 바닥 또는 같은 벽면)에 있어야 합니다.
    """

    def __init__(self, marker: MarkerDetection, marker_cm: float, image_shape: Tuple[int, ...]) -> None:
        if marker_cm <= 0:
            raise ValueError("마커의 실제 한 변 길이는 0보다 커야 합니다.")
        self.marker = marker
        self.marker_cm = marker_cm
        self.image_shape = image_shape
        self.px_per_cm = float(np.clip(marker.side_px / marker_cm, 3.0, 80.0))
        self._build()

    def _build(self) -> None:
        """현재 px_per_cm으로 호모그래피와 캔버스 크기를 계산합니다(캔버스가 너무 크면 축소)."""
        h, w = self.image_shape[:2]
        image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)

        for _ in range(4):
            side = self.marker_cm * self.px_per_cm
            dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
            h0 = cv2.getPerspectiveTransform(self.marker.corners, dst)
            projected = cv2.perspectiveTransform(image_corners, h0).reshape(-1, 2)
            if not np.all(np.isfinite(projected)):
                raise ValueError("마커 기울기가 너무 커서 평면 보정에 실패했습니다. 더 정면에서 찍어 주세요.")
            x0, y0 = projected.min(axis=0)
            x1, y1 = projected.max(axis=0)
            cw, ch = math.ceil(x1 - x0), math.ceil(y1 - y0)
            if max(cw, ch) <= MAX_CANVAS_PX:
                break
            self.px_per_cm *= MAX_CANVAS_PX / max(cw, ch)  # 캔버스가 너무 크면 해상도를 낮춥니다.
        else:
            raise ValueError("사진 범위가 마커에 비해 너무 넓습니다. 마커와 물건이 더 가깝게 보이도록 찍어 주세요.")

        translate = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        self.H = translate @ h0
        self.canvas_size: Tuple[int, int] = (max(cw, 1), max(ch, 1))  # (가로, 세로)

    def warp(self, image_bgr: np.ndarray) -> np.ndarray:
        """사진을 정면 평면 캔버스로 변환합니다."""
        return cv2.warpPerspective(image_bgr, self.H, self.canvas_size, flags=cv2.INTER_LINEAR)

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """원본 이미지 점들을 캔버스 좌표로 변환합니다."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    @property
    def marker_poly_canvas(self) -> np.ndarray:
        """캔버스에서 마커가 차지하는 사각형 (4, 2)."""
        return self.map_points(self.marker.corners)

    def quality(self) -> float:
        """마커가 얼마나 정면에 가깝게 찍혔는지 0~1 점수. 1.0이 가장 좋습니다."""
        angle_term = float(np.clip(1.0 - self.marker.max_angle_deviation_deg() / 30.0, 0.2, 1.0))
        skew_term = float(np.clip((self.marker.skew_ratio() - 0.6) / 0.35, 0.2, 1.0))
        return angle_term * skew_term


# --------------------------------------------------------------------------- #
# 물체 분할
# --------------------------------------------------------------------------- #
class ObjectSegmenter:
    """보정 캔버스에서 마커가 아닌 가장 큰 물체 윤곽을 찾는 클래스.

    현재는 Otsu 임계값 기반의 간이 분할입니다.
    추후 객체 분할 모델(예: SAM, GrabCut 등)로 바꿀 때는 segment()의 반환 형식만 맞추면 됩니다.
    """

    def __init__(self, min_area_cm2: float = MIN_OBJECT_CM2) -> None:
        self.min_area_cm2 = min_area_cm2

    def segment(
        self,
        canvas_bgr: np.ndarray,
        px_per_cm: float,
        marker_poly: np.ndarray,
        roi_poly: Optional[np.ndarray] = None,
    ) -> Optional[ObjectBlob]:
        ch, cw = canvas_bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 마커 영역(주변 여유 포함)은 물체 후보에서 제외합니다.
        exclude = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(exclude, [np.round(marker_poly).astype(np.int32)], 255)
        exclude = cv2.dilate(exclude, np.ones((9, 9), np.uint8))

        # 경계 띠: ROI가 있으면 ROI 테두리, 없으면 캔버스 가장자리.
        # 물체 윤곽이 이 띠에 닿으면 '배경이 잘렸을 가능성'이 높으므로 감점합니다.
        roi_mask = None
        edge_band = np.zeros((ch, cw), np.uint8)
        if roi_poly is not None:
            roi_mask = np.zeros((ch, cw), np.uint8)
            cv2.fillPoly(roi_mask, [np.round(roi_poly).astype(np.int32)], 255)
            edge_band = roi_mask - cv2.erode(roi_mask, np.ones((3, 3), np.uint8))
        else:
            edge_band[0, :] = edge_band[-1, :] = 255
            edge_band[:, 0] = edge_band[:, -1] = 255

        min_area_px = self.min_area_cm2 * px_per_cm * px_per_cm
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)

        best: Optional[ObjectBlob] = None
        best_key = None
        # 밝은 물체/어두운 물체 두 경우를 모두 시도합니다.
        for mask in (binary, cv2.bitwise_not(binary)):
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))
            if roi_mask is not None:
                mask = cv2.bitwise_and(mask, roi_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area_px or area > MAX_OBJECT_AREA_RATIO * cw * ch:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                filled = np.zeros((ch, cw), np.uint8)
                cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
                touches = bool(cv2.bitwise_and(filled, edge_band).any())
                hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-9
                blob = ObjectBlob(
                    contour=c,
                    area_px=float(area),
                    rect=cv2.minAreaRect(c),
                    bbox=(x, y, w, h),
                    solidity=float(area / hull_area),
                    touches_border=touches,
                )
                # 사진 가장자리에 닿지 않은 윤곽을 우선하고, 그다음 큰 윤곽을 고릅니다.
                key = (not touches, area)
                if best_key is None or key > best_key:
                    best, best_key = blob, key
        return best


# --------------------------------------------------------------------------- #
# 측정 서비스 (파사드)
# --------------------------------------------------------------------------- #
class ObjectMeasurer:
    """마커 검출 -> 평면 보정 -> 물체 분할 -> cm 측정을 한 번에 수행하는 클래스."""

    def __init__(self, detector: Optional[MarkerDetector] = None,
                 segmenter: Optional[ObjectSegmenter] = None) -> None:
        self.detector = detector or MarkerDetector()
        self.segmenter = segmenter or ObjectSegmenter()

    def measure_top(
        self,
        image_bgr: np.ndarray,
        marker_cm: float,
        marker_id: Optional[int] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> PhotoMeasurement:
        """상면 사진에서 가로(긴 변)와 깊이(짧은 변)를 측정합니다."""
        return self._measure(image_bgr, marker_cm, marker_id, roi, view="top")

    def measure_side(
        self,
        image_bgr: np.ndarray,
        marker_cm: float,
        marker_id: Optional[int] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> PhotoMeasurement:
        """측면 사진에서 높이를 측정합니다. 마커는 똑바로 세워져 있어야 합니다."""
        return self._measure(image_bgr, marker_cm, marker_id, roi, view="side")

    def _measure(self, image_bgr: np.ndarray, marker_cm: float, marker_id: Optional[int],
                 roi: Optional[Tuple[float, float, float, float]], view: str) -> PhotoMeasurement:
        warnings: List[str] = []
        all_markers = self.detector.detect(image_bgr)
        if not all_markers:
            return PhotoMeasurement.failed(
                "ArUco 마커를 찾지 못했습니다. 마커 전체가 밝은 곳에서 선명하게 보이도록 다시 찍어 주세요."
            )
        if marker_id is not None and not any(m.marker_id == marker_id for m in all_markers):
            return PhotoMeasurement.failed(f"지정한 마커 번호({marker_id})를 찾지 못했습니다.", marker_found=True)

        marker = self.detector.find(image_bgr, marker_id)
        assert marker is not None  # 위에서 존재 여부를 확인했습니다.
        if len(all_markers) > 1 and marker_id is None:
            warnings.append("마커가 여러 개 보여 가장 큰 마커를 기준으로 사용했습니다.")
        if marker.side_px < MIN_MARKER_PX:
            return PhotoMeasurement.failed(
                f"마커가 너무 작게 찍혔습니다({marker.side_px:.0f}px). 더 가까이에서 찍어 주세요.",
                marker_found=True,
            )

        try:
            rectifier = PlaneRectifier(marker, marker_cm, image_bgr.shape)
        except ValueError as e:
            return PhotoMeasurement.failed(str(e), marker_found=True)

        canvas = rectifier.warp(image_bgr)
        roi_poly = rectifier.map_points(roi_to_corners(roi, image_bgr.shape)) if roi else None
        blob = self.segmenter.segment(canvas, rectifier.px_per_cm, rectifier.marker_poly_canvas, roi_poly)
        if blob is None:
            return PhotoMeasurement.failed(
                "물체 윤곽을 찾지 못했습니다. 물체와 배경의 밝기 차이를 크게 하거나, "
                "'물건 영역 직접 지정'으로 물건만 포함되게 지정해 주세요.",
                marker_found=True,
            )

        ppc = rectifier.px_per_cm
        dims: Dict[str, float] = {}
        if view == "top":
            (_, _), (rw, rh), _ = blob.rect
            long_cm, short_cm = max(rw, rh) / ppc, min(rw, rh) / ppc
            dims = {"width": round(long_cm, 1), "depth": round(short_cm, 1)}
            warnings.append("물건 높이가 있으면 위에서 본 크기가 실제보다 크게 나올 수 있습니다. 카메라를 물건 위에 수직으로 두세요.")
        else:
            _, _, _, bh = blob.bbox
            dims = {"height": round(bh / ppc, 1)}
            warnings.append("측면 사진의 높이는 마커가 똑바로 세워져 있다고 가정합니다.")

        # 측정 신뢰도 계산 (각 요인의 곱)
        side_px = marker.side_px
        marker_conf = 1.0 if side_px >= GOOD_MARKER_PX else 0.7
        if side_px < GOOD_MARKER_PX:
            warnings.append(f"마커 해상도가 낮습니다({side_px:.0f}px, 권장 {GOOD_MARKER_PX:.0f}px 이상).")
        quality = rectifier.quality()
        if quality < 0.8:
            warnings.append("마커가 비스듬히 찍혀 보정 오차가 커질 수 있습니다.")
        blob_conf = 1.0
        if blob.touches_border:
            blob_conf *= 0.5
            warnings.append("물건이 사진 가장자리에 닿아 있습니다. 물건 전체가 보이도록 다시 찍어 주세요.")
        if blob.solidity < 0.85:
            blob_conf *= 0.75
            warnings.append("물건 윤곽이 불규칙합니다. 그림자가 섞였을 수 있습니다.")

        confidence = round(marker_conf * quality * blob_conf, 2)
        preview = self._preview(canvas, marker, rectifier, blob)
        return PhotoMeasurement(
            ok=True,
            dims_cm=dims,
            marker_found=True,
            marker_id=marker.marker_id,
            confidence=confidence,
            warnings=warnings,
            preview_bgr=preview,
        )

    @staticmethod
    def _preview(canvas: np.ndarray, marker: MarkerDetection,
                 rectifier: PlaneRectifier, blob: ObjectBlob) -> np.ndarray:
        """사용자가 측정 결과를 눈으로 확인할 수 있도록 윤곽과 마커를 그린 이미지를 만듭니다."""
        img = canvas.copy()
        cv2.polylines(img, [np.round(rectifier.marker_poly_canvas).astype(np.int32)], True, (0, 200, 0), 2)
        cv2.drawContours(img, [blob.contour], -1, (0, 0, 255), 2)
        box = cv2.boxPoints(blob.rect).astype(np.int32)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        return img


# --------------------------------------------------------------------------- #
# 여러 사진 결과 통합 + 비현실 값 검사
# --------------------------------------------------------------------------- #
class SizeEstimator:
    """상면/측면 측정 결과를 하나의 치수 추정값으로 합칩니다."""

    def combine(self, top: Optional[PhotoMeasurement], side: Optional[PhotoMeasurement]) -> ItemEstimate:
        est = ItemEstimate()

        if top is None or not top.ok:
            est.errors.append("가로·깊이 측정 사진(상면)을 먼저 성공적으로 측정해야 합니다.")
        else:
            est.width = top.dims_cm.get("width")
            est.depth = top.dims_cm.get("depth")
            est.warnings.extend(top.warnings)

        if side is None:
            est.warnings.append("측면 사진이 없어 높이는 추정하지 않았습니다(기존 값 유지).")
        elif not side.ok:
            est.errors.append(f"측면 사진 측정 실패: {side.error}")
        else:
            est.height = side.dims_cm.get("height")
            est.warnings.extend(side.warnings)

        confidences = [m.confidence for m in (top, side) if m is not None and m.ok]
        est.confidence = min(confidences) if confidences else 0.0

        # 비현실적인 치수는 버리고 오류로 알려 줍니다.
        for attr, label in (("width", "가로"), ("depth", "깊이"), ("height", "높이")):
            value = getattr(est, attr)
            if value is not None and not (MIN_DIM_CM <= value <= MAX_DIM_CM):
                est.errors.append(
                    f"{label} {value}cm는 비현실적인 값이라 적용하지 않습니다. 마커 크기 입력과 촬영 거리를 확인해 주세요."
                )
                setattr(est, attr, None)
        return est

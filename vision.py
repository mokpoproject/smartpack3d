"""
vision.py - ArUco 마커 기준 기하 측정 모듈 (OpenCV)

핵심 원리
---------
1. 사진 한 장만으로는 cm 크기를 알 수 없으므로, 크기를 아는 기준 물체를 함께 찍습니다.
   - ArUco 마커 (정밀 모드): 무늬로 꼭짓점을 정확히 찾습니다. 기본 인쇄 파일은 한 변 6cm.
   - 체크카드 (간편 모드): 국제 규격 85.60 × 53.98 mm. 누구나 가지고 있어 따로 준비할 필요가 없습니다.
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

# 기준 물체
DEFAULT_MARKER_CM = 6.0      # markers/aruco_id7_6cm_A4.pdf를 100%로 인쇄했을 때 마커 한 변 길이
CARD_W_CM = 8.56             # 신용·체크카드 국제 규격 ISO/IEC 7810 ID-1 (85.60 × 53.98 mm)
CARD_H_CM = 5.398
CARD_ASPECT = CARD_W_CM / CARD_H_CM
REF_MARKER = "marker"
REF_CARD = "card"

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


def _edge_lengths(pts: np.ndarray) -> List[float]:
    return [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]


def _max_angle_dev(pts: np.ndarray) -> float:
    devs = []
    for i in range(4):
        a = pts[(i - 1) % 4] - pts[i]
        b = pts[(i + 1) % 4] - pts[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        devs.append(abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 90.0))
    return max(devs)


@dataclass(frozen=True)
class CardDetection:
    """감지된 체크카드. corners (4, 2)는 첫 변(0→1)이 카드의 긴 변이 되도록 정렬됩니다."""

    corners: np.ndarray
    marker_id: Optional[int] = None  # 마커와 같은 인터페이스를 위해 둡니다(카드는 번호 없음).

    @property
    def side_px(self) -> float:
        """짧은 변의 평균 길이(px). 해상도 판단에 씁니다."""
        e = _edge_lengths(self.corners)
        return (e[1] + e[3]) / 2.0

    def max_angle_deviation_deg(self) -> float:
        return _max_angle_dev(self.corners)

    def skew_ratio(self) -> float:
        """마주 보는 변끼리의 길이 비율 중 작은 값. 1.0에 가까울수록 정면에 가깝습니다."""
        e = _edge_lengths(self.corners)
        return float(min(min(e[0], e[2]) / (max(e[0], e[2]) + 1e-9),
                         min(e[1], e[3]) / (max(e[1], e[3]) + 1e-9)))


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
    reference: str = REF_MARKER                # 기준 물체 종류: "marker" 또는 "card"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    preview_bgr: Optional[np.ndarray] = None  # 측정 검증용 오버레이 이미지

    @classmethod
    def failed(cls, error: str, marker_found: bool = False, reference: str = REF_MARKER) -> "PhotoMeasurement":
        return cls(ok=False, error=error, marker_found=marker_found, reference=reference)


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
# 체크카드 검출
# --------------------------------------------------------------------------- #
class CardDetector:
    """사진에서 체크카드(85.60 × 53.98 mm) 모양의 사각형을 찾는 클래스.

    카드는 마커처럼 무늬가 없으므로, 윤곽 중에서
    (1) 꼭짓점 4개로 근사되고 (2) 직사각형에 가깝고 (3) 가로:세로 비율이 카드(약 1.586)와 비슷한 것을 고릅니다.
    모서리가 둥글어서 생기는 오차는 네 변에 직선을 맞춘 뒤 교점을 구해 줄입니다.
    """

    MIN_AREA_RATIO = 0.004   # 사진 넓이 대비 최소 크기
    MAX_AREA_RATIO = 0.85
    ASPECT_TOL = 0.18        # |log(비율 / 카드 비율)| 허용치 (비스듬한 촬영 고려)

    def find(self, image_bgr: np.ndarray,
             roi: Optional[Tuple[float, float, float, float]] = None) -> Optional[CardDetection]:
        h, w = image_bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        # 밝기 기준(Otsu) 윤곽이 카드 테두리를 가장 정확히 따라가므로 먼저 찾고,
        # 배경과 밝기 차이가 작아 실패하면 경계선(Canny) 윤곽으로 다시 찾습니다.
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8))
        for masks in ([otsu, cv2.bitwise_not(otsu)], [edges]):
            card = self._best(masks, w, h, roi, gray)
            if card is not None:
                return card
        return None

    def _best(self, masks, w: int, h: int,
              roi: Optional[Tuple[float, float, float, float]], gray: np.ndarray) -> Optional[CardDetection]:
        roi_box = None
        if roi is not None:
            x0, y0, x1, y1 = roi
            roi_box = (x0 * w, y0 * h, x1 * w, y1 * h)

        best, best_score = None, None
        for m in masks:
            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.MIN_AREA_RATIO * w * h or area > self.MAX_AREA_RATIO * w * h:
                    continue
                quad = self._quad(c)
                if quad is None:
                    continue
                quad = self._refine(c, quad)
                quad = self._subpixel(gray, quad)
                cx, cy = quad.mean(axis=0)
                if roi_box is not None and not (roi_box[0] <= cx <= roi_box[2] and roi_box[1] <= cy <= roi_box[3]):
                    continue
                e = _edge_lengths(quad)
                long_ = (e[0] + e[2]) / 2.0
                short = (e[1] + e[3]) / 2.0
                if short < 1:
                    continue
                aspect_err = abs(math.log((long_ / short) / CARD_ASPECT))
                if aspect_err > self.ASPECT_TOL or _max_angle_dev(quad) > 25:
                    continue
                if min(e[0], e[2]) / max(e[0], e[2]) < 0.7 or min(e[1], e[3]) / max(e[1], e[3]) < 0.7:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(c))
                quad_area = cv2.contourArea(quad.astype(np.float32))
                fill = hull_area / (quad_area + 1e-9)
                if not 0.9 <= fill <= 1.1:
                    continue  # 직사각형 모양이 아님
                score = aspect_err + abs(1.0 - fill)
                if best_score is None or score < best_score:
                    best, best_score = CardDetection(quad.astype(np.float32)), score
        return best

    @staticmethod
    def _order(pts: np.ndarray) -> np.ndarray:
        """TL, TR, BR, BL 순서로 정렬한 뒤, 첫 변이 긴 변이 되도록 회전합니다."""
        pts = pts.reshape(4, 2).astype(np.float64)
        s = pts.sum(axis=1)
        d = pts[:, 1] - pts[:, 0]
        ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])
        if len({tuple(p) for p in ordered}) < 4:
            return None
        e = _edge_lengths(ordered)
        if e[0] + e[2] < e[1] + e[3]:
            ordered = np.roll(ordered, -1, axis=0)
        return ordered

    def _quad(self, contour: np.ndarray) -> Optional[np.ndarray]:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return self._order(approx)
            if len(approx) < 4:
                break
        return None

    @staticmethod
    def _refine(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """각 변 가운데 부분의 윤곽 점에 직선을 맞추고, 이웃한 직선의 교점을 새 꼭짓점으로 씁니다.
        (카드 모서리가 둥글어서 근사 꼭짓점이 안쪽으로 들어가는 오차를 줄입니다.)"""
        pts = contour.reshape(-1, 2).astype(np.float64)
        lines = []
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            ab = b - a
            L = np.linalg.norm(ab)
            if L < 1:
                return quad
            u = ab / L
            n = np.array([-u[1], u[0]])
            t = (pts - a) @ u / L
            dist = np.abs((pts - a) @ n)
            sel = pts[(t > 0.15) & (t < 0.85) & (dist < max(3.0, 0.03 * L))]
            if len(sel) < 10:
                return quad
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
        out = []
        for i in range(4):
            (p1, d1), (p2, d2) = lines[i - 1], lines[i]
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return quad
            s, _ = np.linalg.solve(A, p2 - p1)
            out.append(p1 + s * d1)
        out = np.array(out)
        # 맞춘 꼭짓점이 원래 근사점에서 너무 멀면(오검출) 원래 값을 씁니다.
        if np.max(np.linalg.norm(out - quad, axis=1)) > 0.08 * max(_edge_lengths(quad)):
            return quad
        return out

    @staticmethod
    def _subpixel(gray: np.ndarray, quad: np.ndarray, half: int = 6, samples: int = 40) -> np.ndarray:
        """각 변을 따라 수직 방향 밝기 변화가 가장 큰 위치를 소수점 픽셀까지 찾아 직선을 다시 맞춥니다.
        (픽셀 계단 모양 윤곽보다 정확해서, 비스듬히 찍은 사진의 원근 보정 오차를 줄여 줍니다.)"""
        g = gray.astype(np.float32)
        lines = []
        center = quad.mean(axis=0)
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            ab = b - a
            L = np.linalg.norm(ab)
            if L < 10:
                return quad
            u = ab / L
            n = np.array([-u[1], u[0]])
            if np.dot((a + b) / 2 - center, n) < 0:
                n = -n  # 바깥쪽을 향하도록
            offs = np.arange(-half, half + 1, dtype=np.float32)
            pts = []
            for t in np.linspace(0.15, 0.85, samples):
                p0 = a + t * ab
                xs = (p0[0] + offs * n[0]).astype(np.float32)
                ys = (p0[1] + offs * n[1]).astype(np.float32)
                prof = cv2.remap(g, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).ravel()
                d = np.abs(np.diff(prof))
                k = int(np.argmax(d))
                if d[k] < 8:  # 경계가 뚜렷하지 않음
                    continue
                frac = 0.0
                if 0 < k < len(d) - 1:
                    den = d[k - 1] - 2 * d[k] + d[k + 1]
                    if abs(den) > 1e-6:
                        frac = 0.5 * (d[k - 1] - d[k + 1]) / den
                off = offs[k] + 0.5 + frac
                pts.append(p0 + off * n)
            if len(pts) < samples // 3:
                return quad
            vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
        out = []
        for i in range(4):
            (p1, d1), (p2, d2) = lines[i - 1], lines[i]
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return quad
            s, _ = np.linalg.solve(A, p2 - p1)
            out.append(p1 + s * d1)
        out = np.array(out)
        if np.max(np.linalg.norm(out - quad, axis=1)) > 0.05 * max(_edge_lengths(quad)):
            return quad
        return out


# --------------------------------------------------------------------------- #
# 평면 보정
# --------------------------------------------------------------------------- #
class PlaneRectifier:
    """마커가 놓인 평면을 정면으로 펴서 cm 단위 캔버스를 만드는 클래스.

    호모그래피 H는 마커 평면 위의 점만 정확하게 변환합니다.
    따라서 물체는 마커와 같은 평면(같은 바닥 또는 같은 벽면)에 있어야 합니다.
    """

    def __init__(self, marker, marker_cm: float, image_shape: Tuple[int, ...],
                 ref_h_cm: Optional[float] = None) -> None:
        """marker: MarkerDetection 또는 CardDetection. 기준 물체는 marker_cm × ref_h_cm 직사각형
        (ref_h_cm이 없으면 정사각형 마커)이며, 첫 변(꼭짓점 0→1)이 marker_cm 쪽입니다."""
        if marker_cm <= 0 or (ref_h_cm is not None and ref_h_cm <= 0):
            raise ValueError("기준 물체의 실제 길이는 0보다 커야 합니다.")
        self.marker = marker
        self.marker_cm = marker_cm
        self.ref_h_cm = ref_h_cm if ref_h_cm is not None else marker_cm
        self.image_shape = image_shape
        e = _edge_lengths(np.asarray(marker.corners, dtype=np.float64))
        ppc = ((e[0] + e[2]) / 2.0 / self.marker_cm + (e[1] + e[3]) / 2.0 / self.ref_h_cm) / 2.0
        self.px_per_cm = float(np.clip(ppc, 3.0, 80.0))
        self._build()

    def _build(self) -> None:
        """현재 px_per_cm으로 호모그래피와 캔버스 크기를 계산합니다(캔버스가 너무 크면 축소)."""
        h, w = self.image_shape[:2]
        image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)

        for _ in range(4):
            rw = self.marker_cm * self.px_per_cm
            rh = self.ref_h_cm * self.px_per_cm
            dst = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], dtype=np.float32)
            h0 = cv2.getPerspectiveTransform(np.asarray(self.marker.corners, dtype=np.float32), dst)
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
            raise ValueError("사진 범위가 기준 물체(마커·카드)에 비해 너무 넓습니다. 기준 물체와 물건이 더 가깝게 보이도록 찍어 주세요.")

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
        exclude, roi_mask, edge_band = self._masks(canvas_bgr.shape, marker_poly, roi_poly)
        best: Optional[ObjectBlob] = None
        best_key = None
        for blob in self._otsu_blobs(canvas_bgr, px_per_cm, exclude, roi_mask, edge_band):
            # 사진 가장자리에 닿지 않은 윤곽을 우선하고, 그다음 큰 윤곽을 고릅니다.
            key = (not blob.touches_border, blob.area_px)
            if best_key is None or key > best_key:
                best, best_key = blob, key
        return best

    @staticmethod
    def _masks(shape, marker_poly, roi_poly):
        ch, cw = shape[:2]
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
        return exclude, roi_mask, edge_band

    def _otsu_blobs(self, canvas_bgr, px_per_cm, exclude, roi_mask, edge_band) -> List[ObjectBlob]:
        """밝은 물체/어두운 물체 두 경우의 Otsu 윤곽 후보를 모두 돌려줍니다."""
        ch, cw = canvas_bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        min_area_px = self.min_area_cm2 * px_per_cm * px_per_cm
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        blobs: List[ObjectBlob] = []
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
                blobs.append(ObjectBlob(
                    contour=c,
                    area_px=float(area),
                    rect=cv2.minAreaRect(c),
                    bbox=(x, y, w, h),
                    solidity=float(area / hull_area),
                    touches_border=touches,
                ))
        return blobs


    # ------------------------------------------------------------------ #
    # 탭 + GrabCut: 사용자가 누른 점을 기준으로 물건만 오려냅니다.
    # ------------------------------------------------------------------ #
    GRABCUT_MAX_SIDE = 800  # 속도를 위해 이 크기로 줄여서 계산합니다.

    def segment_seed(
        self,
        canvas_bgr: np.ndarray,
        px_per_cm: float,
        marker_poly: np.ndarray,
        seed_xy: Tuple[float, float],
        roi_poly: Optional[np.ndarray] = None,
        rounds: int = 3,
        ref_inside: bool = False,
    ) -> Optional[ObjectBlob]:
        """seed_xy(캔버스 좌표)를 '확실한 물건'으로 두고 GrabCut으로 물건 영역을 찾습니다.

        1차: 사진 전체를 '아마 배경'으로 두고 탭한 점 주변만 '확실한 물건'으로 표시.
        2차 이후: 1차 결과를 '아마 물건'으로, 그 주변 사각형만 후보로 두고 다시 계산해
                  무늬가 많은 물건(바코드, 로고)도 한 덩어리로 잡히게 합니다.
        ref_inside=True: 기준 카드/마커가 물건 앞면에 붙어 있는 경우. 기준 영역을 '확실한 물건'으로 둡니다.
        """
        ch, cw = canvas_bgr.shape[:2]
        scale = min(1.0, self.GRABCUT_MAX_SIDE / max(ch, cw))
        img = cv2.resize(canvas_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else canvas_bgr
        h, w = img.shape[:2]
        ppc = px_per_cm * scale
        sx, sy = int(round(seed_xy[0] * scale)), int(round(seed_xy[1] * scale))
        if not (0 <= sx < w and 0 <= sy < h):
            return None

        ref = np.zeros((h, w), np.uint8)
        cv2.fillPoly(ref, [np.round(marker_poly * scale).astype(np.int32)], 255)
        ref = cv2.dilate(ref, np.ones((5, 5), np.uint8))
        if ref[sy, sx] and not ref_inside:
            return None  # 기준 물체(마커/카드)를 누른 경우
        roi = None
        if roi_poly is not None:
            roi = np.zeros((h, w), np.uint8)
            cv2.fillPoly(roi, [np.round(roi_poly * scale).astype(np.int32)], 255)

        seed_r = max(2, int(0.5 * ppc))

        def run(mask: np.ndarray) -> Optional[np.ndarray]:
            cv2.circle(mask, (sx, sy), seed_r, cv2.GC_FGD, -1)
            mask[ref > 0] = cv2.GC_FGD if ref_inside else cv2.GC_BGD
            if roi is not None:
                mask[roi == 0] = cv2.GC_BGD
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(img, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                return None
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if ref_inside:
                fg[ref > 0] = 255
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            _, lab = cv2.connectedComponents(fg)
            if lab[sy, sx] == 0:
                return None
            comp = (lab == lab[sy, sx]).astype(np.uint8) * 255
            # 구멍 채우기
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            comp = np.zeros_like(comp)
            cv2.drawContours(comp, [max(cs, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            # 얇게 튀어나온 부분(그림자·배경 줄기) 제거
            k = max(3, int(0.5 * ppc)) | 1
            opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cs2, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs2:
                inside = [q for q in cs2 if cv2.pointPolygonTest(q, (float(sx), float(sy)), False) >= 0]
                comp = np.zeros_like(comp)
                cv2.drawContours(comp, [max(inside or cs2, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            return comp

        comp = run(np.full((h, w), cv2.GC_PR_BGD, np.uint8))
        if comp is None:
            return None
        area = float(cv2.countNonZero(comp))
        for _ in range(rounds):
            x, y, bw, bh = cv2.boundingRect(comp)
            m = int(0.25 * max(bw, bh) + ppc)
            mask = np.full((h, w), cv2.GC_BGD, np.uint8)
            mask[max(0, y - m):min(h, y + bh + m), max(0, x - m):min(w, x + bw + m)] = cv2.GC_PR_BGD
            mask[comp > 0] = cv2.GC_PR_FGD
            nxt = run(mask)
            if nxt is None:
                break
            new_area = float(cv2.countNonZero(nxt))
            comp = nxt
            if abs(new_area - area) < 0.01 * area:
                break
            area = new_area

        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area_cm2 * ppc * ppc:
            return None
        # 원래 캔버스 좌표로 되돌립니다.
        c = np.round(c.astype(np.float64) / scale).astype(np.int32)
        area_px = float(cv2.contourArea(c))
        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-9
        x, y, bw, bh = cv2.boundingRect(c)
        touches = x <= 1 or y <= 1 or x + bw >= cw - 1 or y + bh >= ch - 1
        gc_blob = ObjectBlob(
            contour=c,
            area_px=area_px,
            rect=cv2.minAreaRect(c),
            bbox=(x, y, bw, bh),
            solidity=float(area_px / hull_area),
            touches_border=bool(touches),
        )

        if ref_inside:
            return gc_blob
        # 보완: 무늬가 많은 물건은 GrabCut이 일부만 잡을 수 있습니다.
        # 탭한 점을 포함하고, GrabCut 결과를 거의 다 덮으며, 너무 크지 않은 Otsu 윤곽이 있으면 그것을 씁니다.
        exclude, roi_mask, edge_band = self._masks(canvas_bgr.shape, marker_poly, roi_poly)
        gc_mask = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(gc_mask, [c], -1, 255, cv2.FILLED)
        best = gc_blob
        for blob in self._otsu_blobs(canvas_bgr, px_per_cm, exclude, roi_mask, edge_band):
            if blob.touches_border or blob.area_px <= best.area_px:
                continue
            if blob.area_px > 8.0 * area_px:
                continue
            if cv2.pointPolygonTest(blob.contour, (float(seed_xy[0]), float(seed_xy[1])), False) < 0:
                continue
            om = np.zeros((ch, cw), np.uint8)
            cv2.drawContours(om, [blob.contour], -1, 255, cv2.FILLED)
            covered = cv2.countNonZero(cv2.bitwise_and(om, gc_mask)) / max(area_px, 1.0)
            if covered >= 0.9:
                best = blob
        return best


# --------------------------------------------------------------------------- #
# 측정 서비스 (파사드)
# --------------------------------------------------------------------------- #
class ObjectMeasurer:
    """마커 검출 -> 평면 보정 -> 물체 분할 -> cm 측정을 한 번에 수행하는 클래스."""

    def __init__(self, detector: Optional[MarkerDetector] = None,
                 segmenter: Optional[ObjectSegmenter] = None,
                 card_detector: Optional[CardDetector] = None) -> None:
        self.detector = detector or MarkerDetector()
        self.segmenter = segmenter or ObjectSegmenter()
        self.card_detector = card_detector or CardDetector()

    def measure_top(
        self,
        image_bgr: np.ndarray,
        marker_cm: float = DEFAULT_MARKER_CM,
        marker_id: Optional[int] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
        reference: str = REF_MARKER,
        card_roi: Optional[Tuple[float, float, float, float]] = None,
        seed: Optional[Tuple[float, float]] = None,
    ) -> PhotoMeasurement:
        """상면 사진에서 가로(긴 변)와 깊이(짧은 변)를 측정합니다.
        reference="card"이면 마커 대신 체크카드를 기준으로 씁니다(marker_cm 무시).
        seed=(x, y)는 사용자가 물건을 탭한 위치(사진 기준 0~1 비율)이며, 주면 GrabCut으로 분할합니다."""
        return self._measure(image_bgr, marker_cm, marker_id, roi, "top", reference, card_roi, seed)

    def measure_side(
        self,
        image_bgr: np.ndarray,
        marker_cm: float = DEFAULT_MARKER_CM,
        marker_id: Optional[int] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
        reference: str = REF_MARKER,
        card_roi: Optional[Tuple[float, float, float, float]] = None,
        seed: Optional[Tuple[float, float]] = None,
        ref_on_object: bool = False,
    ) -> PhotoMeasurement:
        """측면 사진에서 높이를 측정합니다.
        ref_on_object=False: 기준(마커/카드)을 물건 옆에 똑바로 세워 찍은 사진.
        ref_on_object=True : 기준을 물건 앞면에 평평하게 대고 정면에서 찍은 사진(추천, 더 정확).
                             이때는 탭하지 않아도 기준 위치에서 물건을 찾아 갑니다."""
        return self._measure(image_bgr, marker_cm, marker_id, roi, "side_face" if ref_on_object else "side",
                             reference, card_roi, seed)

    def _find_reference(self, image_bgr, marker_cm, marker_id, reference, card_roi, warnings):
        """기준 물체를 찾아 (검출 결과, 긴 변 cm, 짧은 변 cm) 또는 실패 결과를 돌려줍니다."""
        if reference == REF_CARD:
            card = self.card_detector.find(image_bgr, card_roi)
            if card is None:
                return PhotoMeasurement.failed(
                    "체크카드를 찾지 못했습니다. 카드 전체가 보이게, 배경과 색이 다른 바닥 위에 놓고 찍어 주세요. "
                    "그래도 안 되면 '카드 영역 직접 지정'을 사용하세요.",
                    reference=REF_CARD,
                )
            return card, CARD_W_CM, CARD_H_CM
        return self._find_marker(image_bgr, marker_cm, marker_id, warnings)

    def _measure(self, image_bgr: np.ndarray, marker_cm: float, marker_id: Optional[int],
                 roi: Optional[Tuple[float, float, float, float]], view: str,
                 reference: str = REF_MARKER,
                 card_roi: Optional[Tuple[float, float, float, float]] = None,
                 seed: Optional[Tuple[float, float]] = None) -> PhotoMeasurement:
        warnings: List[str] = []
        found = self._find_reference(image_bgr, marker_cm, marker_id, reference, card_roi, warnings)
        if isinstance(found, PhotoMeasurement):
            return found
        marker, ref_w_cm, ref_h_cm = found
        return self._measure_with(image_bgr, marker, ref_w_cm, ref_h_cm, roi, view, reference, warnings, seed)

    def _find_marker(self, image_bgr, marker_cm, marker_id, warnings):
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
        return marker, marker_cm, marker_cm

    def _measure_with(self, image_bgr, marker, ref_w_cm, ref_h_cm, roi, view, reference, warnings, seed=None):
        ref_name = "카드" if reference == REF_CARD else "마커"
        if marker.side_px < MIN_MARKER_PX:
            return PhotoMeasurement.failed(
                f"{ref_name}가 너무 작게 찍혔습니다({marker.side_px:.0f}px). 더 가까이에서 찍어 주세요.",
                marker_found=True, reference=reference,
            )

        try:
            rectifier = PlaneRectifier(marker, ref_w_cm, image_bgr.shape, ref_h_cm=ref_h_cm)
        except ValueError as e:
            return PhotoMeasurement.failed(str(e), marker_found=True, reference=reference)

        canvas = rectifier.warp(image_bgr)
        roi_poly = rectifier.map_points(roi_to_corners(roi, image_bgr.shape)) if roi else None
        blob = None
        used_seed = False
        face = view == "side_face"
        if face:
            # 기준이 물건 앞면에 붙어 있으므로, 탭이 없으면 기준 중심을 물건 위치로 씁니다.
            ih, iw = image_bgr.shape[:2]
            if seed is not None:
                seed_canvas = rectifier.map_points(np.array([[seed[0] * iw, seed[1] * ih]], np.float64))[0]
            else:
                seed_canvas = rectifier.marker_poly_canvas.mean(axis=0)
            blob = self.segmenter.segment_seed(canvas, rectifier.px_per_cm, rectifier.marker_poly_canvas,
                                               (float(seed_canvas[0]), float(seed_canvas[1])), roi_poly,
                                               ref_inside=True)
            used_seed = blob is not None
            if blob is None:
                return PhotoMeasurement.failed(
                    f"{ref_name}가 붙은 물건 면을 찾지 못했습니다. 물건 앞면이 사진 가운데에 크게 나오게, "
                    f"{ref_name}는 앞면 안쪽에 평평하게 대고 다시 찍어 주세요.",
                    marker_found=True, reference=reference,
                )
        elif seed is not None:
            ih, iw = image_bgr.shape[:2]
            seed_canvas = rectifier.map_points(np.array([[seed[0] * iw, seed[1] * ih]], np.float64))[0]
            blob = self.segmenter.segment_seed(canvas, rectifier.px_per_cm, rectifier.marker_poly_canvas,
                                               (float(seed_canvas[0]), float(seed_canvas[1])), roi_poly)
            if blob is None:
                warnings.append("탭한 위치에서 물건을 찾지 못해 자동 인식으로 측정했습니다. "
                                f"{ref_name}가 아닌 물건 위를 눌렀는지 확인하세요.")
            else:
                used_seed = True
        if blob is None:
            blob = self.segmenter.segment(canvas, rectifier.px_per_cm, rectifier.marker_poly_canvas, roi_poly)
        if blob is None:
            return PhotoMeasurement.failed(
                "물체 윤곽을 찾지 못했습니다. 물체와 배경의 밝기 차이를 크게 하거나, "
                "'물건 영역 직접 지정'으로 물건만 포함되게 지정해 주세요.",
                marker_found=True, reference=reference,
            )

        ppc = rectifier.px_per_cm
        dims: Dict[str, float] = {}
        if view == "top":
            (_, _), (rw, rh), _ = blob.rect
            long_cm, short_cm = max(rw, rh) / ppc, min(rw, rh) / ppc
            dims = {"width": round(long_cm, 1), "depth": round(short_cm, 1)}
            warnings.append("물건 높이가 있으면 위에서 본 크기가 실제보다 크게 나올 수 있습니다. 카메라를 물건 위에 수직으로 두세요.")
        elif face:
            # 사진의 '위쪽' 방향을 보정 평면으로 옮겨, 그 방향에 가까운 물건 변의 길이를 높이로 씁니다.
            # (카드를 가로/세로 어느 방향으로 대도 됩니다.)
            ih, iw = image_bgr.shape[:2]
            c_img = np.asarray(marker.corners, np.float64).mean(axis=0)
            step = max(10.0, marker.side_px * 0.5)
            up_pts = rectifier.map_points(np.array([c_img, c_img - [0.0, step]]))
            up = up_pts[1] - up_pts[0]
            up /= (np.linalg.norm(up) + 1e-9)
            box = cv2.boxPoints(blob.rect)
            e1, e2 = box[1] - box[0], box[2] - box[1]
            l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
            a1 = abs(float(np.dot(e1 / (l1 + 1e-9), up)))
            a2 = abs(float(np.dot(e2 / (l2 + 1e-9), up)))
            height_px, width_px = (l1, l2) if a1 >= a2 else (l2, l1)
            dims = {"height": round(height_px / ppc, 1)}
            warnings.append(f"{ref_name}가 물건 앞면에 평평하게 닿아 있다고 가정합니다. "
                            f"(참고: 앞면 가로 약 {width_px / ppc:.1f}cm)")
            # 기준이 물건 밖으로 삐져나오면 경계가 기준 테두리로 잘못 잡힐 수 있습니다.
            ref_mask = np.zeros(canvas.shape[:2], np.uint8)
            cv2.fillPoly(ref_mask, [np.round(rectifier.marker_poly_canvas).astype(np.int32)], 255)
            obj_mask = np.zeros(canvas.shape[:2], np.uint8)
            cv2.drawContours(obj_mask, [blob.contour], -1, 255, cv2.FILLED)
            inner = cv2.erode(obj_mask, np.ones((5, 5), np.uint8))
            if cv2.countNonZero(cv2.bitwise_and(ref_mask, cv2.bitwise_not(inner))) > 0.02 * cv2.countNonZero(ref_mask):
                warnings.append(f"{ref_name}가 물건 면 가장자리에 걸쳐 있습니다. 물건 높이가 {ref_name}보다 낮으면 "
                                f"{ref_name}를 세로로 돌리거나 측면 촬영 방법을 '옆에 세우기'로 바꿔 보세요.")
        else:
            _, _, _, bh = blob.bbox
            dims = {"height": round(bh / ppc, 1)}
            warnings.append(f"측면 사진의 높이는 {ref_name}가 똑바로 세워져 있다고 가정합니다.")

        # 측정 신뢰도 계산 (각 요인의 곱)
        side_px = marker.side_px
        marker_conf = 1.0 if side_px >= GOOD_MARKER_PX else 0.7
        if side_px < GOOD_MARKER_PX:
            warnings.append(f"{ref_name} 해상도가 낮습니다({side_px:.0f}px, 권장 {GOOD_MARKER_PX:.0f}px 이상).")
        quality = rectifier.quality()
        if quality < 0.8:
            warnings.append(f"{ref_name}가 비스듬히 찍혀 보정 오차가 커질 수 있습니다.")
        blob_conf = 1.0
        if blob.touches_border:
            blob_conf *= 0.5
            warnings.append("물건이 사진 가장자리에 닿아 있습니다. 물건 전체가 보이도록 다시 찍어 주세요.")
        if blob.solidity < (0.8 if used_seed else 0.85):
            blob_conf *= 0.75
            warnings.append("물건 윤곽이 불규칙합니다. 그림자가 섞였을 수 있습니다.")

        ref_conf = 1.0
        if reference == REF_CARD:
            ref_conf = 0.9  # 카드는 무늬가 없어 마커보다 꼭짓점 위치가 덜 정확합니다.
            warnings.append("체크카드 기준 측정입니다. 더 정확하게 재려면 ArUco 마커 모드를 사용하세요.")
        confidence = round(marker_conf * quality * blob_conf * ref_conf, 2)
        preview = self._preview(canvas, marker, rectifier, blob)
        if used_seed:
            sc = tuple(int(v) for v in seed_canvas)
            cv2.circle(preview, sc, max(6, int(0.3 * ppc)), (0, 255, 255), -1)
            cv2.circle(preview, sc, max(6, int(0.3 * ppc)), (0, 0, 0), 2)
        return PhotoMeasurement(
            ok=True,
            dims_cm=dims,
            marker_found=True,
            marker_id=marker.marker_id,
            reference=reference,
            confidence=confidence,
            warnings=warnings,
            preview_bgr=preview,
        )

    @staticmethod
    def _preview(canvas: np.ndarray, marker,
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
                    f"{label} {value}cm는 비현실적인 값이라 적용하지 않습니다. 기준 물체(마커 크기 입력·카드)와 촬영 거리를 확인해 주세요."
                )
                setattr(est, attr, None)
        return est

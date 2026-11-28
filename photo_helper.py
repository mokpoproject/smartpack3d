"""
photo_helper.py - 가이드 카메라 / 물건 탭 컴포넌트 (Streamlit 커스텀 컴포넌트 래퍼)

components/photo_helper/index.html 하나로 두 기능을 제공합니다(빌드 과정 없음).
- guide_camera(): 기준 칸(카드/마커)과 물건 칸이 그려진 카메라. 찍은 뒤 물건을 탭할 수 있습니다.
- tap_image():    올린 사진에서 물건을 한 번 탭해 위치를 받습니다.

주의: 브라우저 보안 정책 때문에 카메라는 https 주소나 localhost에서만 켜집니다.
      (같은 와이파이의 휴대폰에서 http://PC주소:8501 로 접속하면 카메라가 막힙니다.)
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "photo_helper")
_component = components.declare_component("smartpack_photo_helper", path=_DIR)

Norm2 = Tuple[float, float]
Norm4 = Tuple[float, float, float, float]


@dataclass
class GuideShot:
    image_bytes: Optional[bytes]
    seed: Optional[Norm2]
    ref_roi: Optional[Norm4]


def _norm(v, n: int):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    try:
        vals = tuple(min(1.0, max(0.0, float(x))) for x in v)
    except (TypeError, ValueError):
        return None
    return vals


def _decode_data_url(url) -> Optional[bytes]:
    if not isinstance(url, str) or "," not in url:
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def guide_camera(reference: str, key: str) -> GuideShot:
    """가이드 네모 카메라를 보여 주고 (사진, 탭 위치, 기준 칸 위치)를 돌려줍니다."""
    v = _component(mode="camera", reference=reference, key=key, default=None)
    if not isinstance(v, dict):
        return GuideShot(None, None, None)
    return GuideShot(_decode_data_url(v.get("image")), _norm(v.get("seed"), 2), _norm(v.get("ref_roi"), 4))


def preview_data_url(data: bytes, max_side: int = 960) -> Optional[str]:
    """탭 화면에 보낼 축소 미리보기 (원본과 가로세로 비율이 같으므로 탭 위치 비율이 그대로 맞습니다)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    sc = min(1.0, max_side / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def tap_image(data: bytes, key: str) -> Optional[Norm2]:
    """올린 사진을 보여 주고, 사용자가 탭한 위치(0~1 비율)를 돌려줍니다. 아직 안 눌렀으면 None."""
    url = preview_data_url(data)
    if url is None:
        return None
    v = _component(mode="tap", image=url, key=key, default=None)
    if not isinstance(v, dict):
        return None
    return _norm(v.get("seed"), 2)

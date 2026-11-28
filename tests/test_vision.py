"""사진 크기 측정 테스트 (합성 이미지): python -m pytest -q"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import DEFAULT_MARKER_CM, REF_CARD, ObjectMeasurer  # noqa: E402

PX = 24  # 1cm = 24px
W, H = 2000, 1500


def _card(img, x, y, color=(150, 140, 120)):
    w, h, r = int(8.56 * PX), int(5.398 * PX), int(0.3 * PX)
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def _marker(img, x, y, cm=DEFAULT_MARKER_CM):
    side = int(cm * PX)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cv2.rectangle(img, (x - 20, y - 20), (x + side + 20, y + side + 20), (255, 255, 255), -1)
    img[y:y + side, x:x + side] = cv2.cvtColor(cv2.aruco.generateImageMarker(d, 7, side), cv2.COLOR_GRAY2BGR)


def _scene(kind, tilt=False):
    img = np.full((H, W, 3), 235, np.uint8)
    if kind == "card":
        _card(img, 640, 560)
    else:
        _marker(img, 640, 560)
    cv2.rectangle(img, (900, 500), (900 + 12 * PX, 500 + 8 * PX), (90, 60, 40), -1)  # 12 x 8 cm 물건
    if tilt:
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[60, 40], [1940, 0], [W, H], [0, 1460]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(200, 200, 200))
    return img


def _err(r):
    assert r.ok, r.error
    a = sorted([r.dims_cm["width"], r.dims_cm["depth"]], reverse=True)
    return max(abs(a[0] - 12), abs(a[1] - 8))


def test_marker_mode_default_size():
    assert _err(ObjectMeasurer().measure_top(_scene("marker"))) <= 0.3


def test_card_mode_front():
    assert _err(ObjectMeasurer().measure_top(_scene("card"), reference=REF_CARD)) <= 0.3


def test_card_mode_tilted():
    assert _err(ObjectMeasurer().measure_top(_scene("card", tilt=True), reference=REF_CARD)) <= 0.5


def test_card_not_found_gives_error():
    img = np.full((H, W, 3), 235, np.uint8)
    r = ObjectMeasurer().measure_top(img, reference=REF_CARD)
    assert not r.ok and "체크카드" in r.error

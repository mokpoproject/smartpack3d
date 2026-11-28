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


def _distractor_scene():
    """큰 어두운 배경 물체(노트북 화면 같은)가 있어 자동 인식이 헷갈리는 장면."""
    img = _scene("marker")
    cv2.rectangle(img, (1300, 200), (1300 + 25 * PX // 1, 200 + 40 * PX), (40, 40, 40), -1)  # 큰 어두운 물체(가장자리 닿음)
    cv2.rectangle(img, (150, 1000), (150 + 20 * PX, 1000 + 16 * PX), (60, 50, 50), -1)   # 20 x 16 cm 다른 물건
    return img


def test_tap_grabcut_picks_tapped_object():
    img = _distractor_scene()
    seed = ((900 + 6 * PX) / W, (500 + 4 * PX) / H)  # 12 x 8 cm 물건 가운데를 탭
    assert _err(ObjectMeasurer().measure_top(img, seed=seed)) <= 0.4


def test_tap_on_marker_falls_back_with_warning():
    img = _scene("marker")
    seed = ((640 + 3 * PX) / W, (560 + 3 * PX) / H)  # 마커를 잘못 탭
    r = ObjectMeasurer().measure_top(img, seed=seed)
    assert r.ok and any("탭한 위치" in w for w in r.warnings)


def _face_scene(box_w_cm, box_h_cm, card_vertical=False, tilt=False):
    """측면 사진: 물건 앞면(어두운 상자)에 밝은 카드를 대고 정면에서 찍은 장면."""
    img = np.full((H, W, 3), 225, np.uint8)
    img[1100:] = (170, 175, 180)  # 바닥
    bw, bh = int(box_w_cm * PX), int(box_h_cm * PX)
    x0, y1 = 700, 1100
    cv2.rectangle(img, (x0, y1 - bh), (x0 + bw, y1), (60, 80, 110), -1)
    card = np.full((int(5.398 * PX) + 40, int(8.56 * PX) + 40, 3), (60, 80, 110), np.uint8)
    _card(card, 20, 20, color=(235, 235, 240))
    card = card[20:-20, 20:-20]
    if card_vertical:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    ch_, cw_ = card.shape[:2]
    cx, cy = x0 + bw // 2 - cw_ // 2, y1 - bh // 2 - ch_ // 2
    img[cy:cy + ch_, cx:cx + cw_] = card
    if tilt:
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[80, 30], [1950, 0], [W, H], [0, 1470]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(200, 200, 200))
    return img


def _height(r):
    assert r.ok, r.error
    return r.dims_cm["height"]


def test_face_mode_height_card_horizontal():
    r = ObjectMeasurer().measure_side(_face_scene(14, 9), reference=REF_CARD, ref_on_object=True)
    assert abs(_height(r) - 9) <= 0.3


def test_face_mode_height_card_vertical_tilted():
    r = ObjectMeasurer().measure_side(_face_scene(14, 11, card_vertical=True, tilt=True),
                                      reference=REF_CARD, ref_on_object=True)
    assert abs(_height(r) - 11) <= 0.5


def test_face_mode_low_object_card_near_edge():
    r = ObjectMeasurer().measure_side(_face_scene(16, 6), reference=REF_CARD, ref_on_object=True)
    assert abs(_height(r) - 6) <= 0.4

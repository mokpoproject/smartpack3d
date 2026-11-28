"""
app.py - SmartPack 3D Streamlit UI (표현 계층)

실행:
    streamlit run app.py

구성:
- 탭 1: 물건 입력 표 (기존 기능 유지)
- 탭 2: 사진으로 크기 추정 (마커/체크카드 기준, 가이드 카메라, 물건 탭 + GrabCut, LLM 보조 판정)
- 탭 3: 3D 자동 배치 결과 (기존 기능 유지)

업무 로직은 packing.py / vision.py / llm_service.py / plot3d.py에 있고,
이 파일은 입력을 받아 각 서비스를 호출하고 결과를 보여 주기만 합니다.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from llm_service import ItemAdvice, ItemAdvisor, create_advisor
from packing import (
    COLORS, ORIENT_FREE, ORIENT_OPTIONS, ORIENT_UPRIGHT,
    Bin, Item, Packer, limit_text, volume,
)
from photo_helper import guide_camera, tap_image
from plot3d import make_figure
from vision import (
    ARUCO_DICTS, CARD_H_CM, CARD_W_CM, DEFAULT_MARKER_CM, REF_CARD, REF_MARKER,
    ItemEstimate, MarkerDetector, ObjectMeasurer, PhotoMeasurement, SizeEstimator, decode_image,
)

MARKER_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers", "aruco_id7_6cm_A4.pdf")
MODE_MARKER = "ArUco 마커 (정밀)"
MODE_CARD = "체크카드 (간편)"

# 모바일에서는 사이드바가 자동으로 접히도록 'auto'로 둡니다.
st.set_page_config(page_title="SmartPack 3D", page_icon="📦", layout="wide", initial_sidebar_state="auto")

# --------------------------------------------------------------------------- #
# 열 이름 상수 (표 편집 UI와 데이터 변환에서 공통으로 씁니다)
# --------------------------------------------------------------------------- #
COL_NAME, COL_W, COL_D, COL_H = "물건", "가로(cm)", "깊이(cm)", "높이(cm)"
COL_KG, COL_FRAGILE, COL_LOAD = "무게(kg)", "깨짐/눌림 주의", "버팀 무게(kg)"
COL_ORIENT, COL_LOCK, COL_SOURCE = "방향", "크기 잠금", "크기 출처"

SOURCE_MANUAL = "직접 입력"
SOURCE_AI_UNCONFIRMED = "사진 추정(확인 필요)"
SOURCE_OPTIONS = [SOURCE_MANUAL, SOURCE_AI_UNCONFIRMED, "사진 추정(확인 완료)"]

EXAMPLE_ITEMS = [
    {COL_NAME: "책", COL_W: 20.0, COL_D: 15.0, COL_H: 4.0, COL_KG: 0.5, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
    {COL_NAME: "수납 상자", COL_W: 12.0, COL_D: 10.0, COL_H: 8.0, COL_KG: 0.6, COL_FRAGILE: False, COL_LOAD: 2.0, COL_ORIENT: ORIENT_FREE, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
    {COL_NAME: "필통", COL_W: 18.0, COL_D: 6.0, COL_H: 5.0, COL_KG: 0.3, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
    {COL_NAME: "텀블러", COL_W: 8.0, COL_D: 8.0, COL_H: 20.0, COL_KG: 0.4, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_UPRIGHT, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
    {COL_NAME: "과자 상자", COL_W: 15.0, COL_D: 10.0, COL_H: 6.0, COL_KG: 0.1, COL_FRAGILE: True, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
    {COL_NAME: "작은 상자", COL_W: 8.0, COL_D: 8.0, COL_H: 7.0, COL_KG: 0.2, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False, COL_SOURCE: SOURCE_MANUAL},
]


# --------------------------------------------------------------------------- #
# 캐시된 서비스 객체 (Streamlit 재실행마다 새로 만들지 않도록)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_measurer(dict_name: str) -> ObjectMeasurer:
    return ObjectMeasurer(detector=MarkerDetector(ARUCO_DICTS[dict_name]))


@st.cache_resource
def get_advisor() -> ItemAdvisor:
    return create_advisor()


# --------------------------------------------------------------------------- #
# 세션 상태 초기화
# --------------------------------------------------------------------------- #
if "editor_ver" not in st.session_state:
    st.session_state.editor_ver = 0
if "table_df" not in st.session_state:
    st.session_state.table_df = pd.DataFrame(EXAMPLE_ITEMS)
if "apply_msg" not in st.session_state:
    st.session_state.apply_msg = None


# --------------------------------------------------------------------------- #
# 사이드바: 수납함 크기
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("수납함 설정")
    bin_w = st.number_input("가로 (cm)", min_value=1.0, max_value=300.0, value=40.0, step=1.0)
    bin_d = st.number_input("깊이 (cm)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    bin_h = st.number_input("높이 (cm)", min_value=1.0, max_value=300.0, value=25.0, step=1.0)
    st.info(f"전체 부피: {volume(bin_w, bin_d, bin_h):,.0f} cm³")
    st.caption("휴대폰에서는 이 메뉴를 왼쪽 위 ≫ 버튼으로 열 수 있습니다.")

tab_items, tab_photo, tab_result = st.tabs(["1. 물건 입력", "2. 사진으로 크기 추정", "3. 3D 자동 배치"])

# =========================================================================== #
# 탭 1: 물건 입력 표 (기존 기능 유지)
# =========================================================================== #
with tab_items:
    st.subheader("1. 정리할 물건 입력")
    st.write(
        "표를 직접 고치세요. 맨 아래 빈 줄에 입력하면 물건이 추가되고, "
        "줄 왼쪽을 선택한 뒤 휴지통을 누르면 삭제됩니다. "
        "사진 추정 값은 ‘크기 출처’에 표시되며 사용자가 확인해야 합니다."
    )

    edited = st.data_editor(
        st.session_state.table_df,
        key=f"items_editor_{st.session_state.editor_ver}",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            COL_NAME: st.column_config.TextColumn(COL_NAME, required=True, default="새 물건"),
            COL_W: st.column_config.NumberColumn(COL_W, min_value=0.1, step=0.5, required=True, default=10.0),
            COL_D: st.column_config.NumberColumn(COL_D, min_value=0.1, step=0.5, required=True, default=10.0),
            COL_H: st.column_config.NumberColumn(COL_H, min_value=0.1, step=0.5, required=True, default=5.0),
            COL_KG: st.column_config.NumberColumn(COL_KG, min_value=0.0, step=0.1, default=0.5, format="%.2f"),
            COL_FRAGILE: st.column_config.CheckboxColumn(
                COL_FRAGILE, default=False,
                help="체크하면 나중에(위쪽에) 넣고, 버팀 무게를 비워 두면 위에 아무것도 올리지 않습니다.",
            ),
            COL_LOAD: st.column_config.NumberColumn(
                COL_LOAD, min_value=0.0, step=0.1, format="%.2f",
                help="이 물건 위에 올라갈 수 있는 최대 무게. 비워 두면 일반 물건은 제한 없음, 주의 물건은 0kg.",
            ),
            COL_ORIENT: st.column_config.SelectboxColumn(
                COL_ORIENT, options=ORIENT_OPTIONS, required=True, default=ORIENT_FREE,
                help="자유: 6방향 회전 / 세우기 고정: 높이 방향 유지, 바닥에서만 회전 / 완전 고정: 입력한 방향 그대로",
            ),
            COL_LOCK: st.column_config.CheckboxColumn(
                COL_LOCK, default=False,
                help="체크하면 사진 추정으로 이 물건의 크기가 바뀌지 않습니다.",
            ),
            COL_SOURCE: st.column_config.SelectboxColumn(
                COL_SOURCE, options=SOURCE_OPTIONS, default=SOURCE_MANUAL,
                help="사진으로 추정한 값은 확인 후 ‘사진 추정(확인 완료)’으로 바꿔 주세요.",
            ),
        },
    )
    # 다른 탭(사진 추정 적용, 3D 배치)에서 현재 편집 상태를 쓸 수 있도록 저장합니다.
    st.session_state.last_edited = edited

    if st.button("예시 물건으로 초기화"):
        st.session_state.table_df = pd.DataFrame(EXAMPLE_ITEMS)
        st.session_state.editor_ver += 1
        st.session_state.pop("last_placed", None)
        st.session_state.pop("estimate", None)
        st.rerun()


# =========================================================================== #
# 탭 2: 사진으로 크기 추정
# =========================================================================== #
PHOTO_UPLOAD = "사진 올리기"
PHOTO_GUIDE = "가이드 카메라"
PHOTO_BASIC = "기본 카메라"


class PhotoInput:
    """사진 한 장에 대한 입력 묶음: 사진, 물건 탭 위치, (가이드 카메라의) 기준 칸 위치."""

    def __init__(self, data: Optional[bytes] = None, seed=None, ref_roi=None) -> None:
        self.data = data
        self.seed = seed
        self.ref_roi = ref_roi


def photo_input(label: str, key: str, reference: str) -> PhotoInput:
    """사진 입력: 올리기 / 가이드 카메라 / 기본 카메라 중 선택하고, 물건을 한 번 탭하게 합니다."""
    st.markdown(f"**{label}**")
    how = st.radio("사진 입력 방법", [PHOTO_UPLOAD, PHOTO_GUIDE, PHOTO_BASIC], horizontal=True,
                   key=f"{key}_how", label_visibility="collapsed")

    if how == PHOTO_GUIDE:
        st.caption("휴대폰에서는 https 주소(예: Streamlit Cloud)로 접속해야 카메라가 켜집니다. PC는 localhost에서 바로 됩니다.")
        shot = guide_camera(reference, key=f"{key}_guide")
        return PhotoInput(shot.image_bytes, shot.seed, shot.ref_roi if reference == REF_CARD else None)

    data: Optional[bytes] = None
    if how == PHOTO_BASIC:
        cam = st.camera_input(label, key=f"{key}_camera", label_visibility="collapsed")
        data = cam.getvalue() if cam is not None else None
    else:
        up = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"{key}_upload",
                              label_visibility="collapsed")
        data = up.getvalue() if up is not None else None
    if data is None:
        return PhotoInput()

    seed = None
    if st.toggle("👆 물건을 탭해서 지정 (추천)", value=True, key=f"{key}_tap_on",
                 help="사진에서 물건을 한 번 누르면 그 점을 기준으로 색과 경계를 따라 물건만 오려냅니다(GrabCut)."):
        seed = tap_image(data, key=f"{key}_tap_{hash(data) & 0xFFFFFFFF:x}")
        if seed is None:
            st.caption("아직 탭하지 않았습니다. 탭하지 않으면 자동 인식으로 측정합니다.")
    return PhotoInput(data, seed)


def roi_input(key: str, label: str = "물건 영역 직접 지정 (배경이 복잡할 때)") -> Optional[Tuple[float, float, float, float]]:
    """배경이 복잡할 때 물건(또는 카드) 영역을 사각형으로 지정합니다(선택)."""
    if not st.checkbox(label, key=f"{key}_roi_on"):
        return None
    c1, c2 = st.columns(2)
    x = c1.slider("좌우 범위 (%)", 0, 100, (5, 95), key=f"{key}_rx")
    y = c2.slider("상하 범위 (%)", 0, 100, (5, 95), key=f"{key}_ry")
    return (x[0] / 100, y[0] / 100, x[1] / 100, y[1] / 100)


def measure_photo(measurer: ObjectMeasurer, data: bytes, marker_cm: float, view: str,
                  roi: Optional[Tuple[float, float, float, float]], reference: str = REF_MARKER,
                  card_roi: Optional[Tuple[float, float, float, float]] = None,
                  seed: Optional[Tuple[float, float]] = None,
                  ) -> Tuple[PhotoMeasurement, Optional[np.ndarray]]:
    """사진 bytes를 디코딩해 측정합니다. 디코딩 오류도 실패 결과로 바꿉니다."""
    try:
        img = decode_image(data)
    except ValueError as e:
        return PhotoMeasurement.failed(str(e), reference=reference), None
    if view == "top":
        return measurer.measure_top(img, marker_cm, roi=roi, reference=reference, card_roi=card_roi, seed=seed), img
    return measurer.measure_side(img, marker_cm, roi=roi, reference=reference, card_roi=card_roi, seed=seed), img


def run_estimation(target_idx: int, target_name: str, marker_cm: float, dict_name: str,
                   top: "PhotoInput", side: "PhotoInput",
                   roi_top, roi_side, reference: str = REF_MARKER,
                   card_roi_top=None, card_roi_side=None) -> None:
    """사진 측정 -> 결과 통합 -> LLM 보조 판정을 수행하고 결과를 세션에 저장합니다."""
    measurer = get_measurer(dict_name)
    top_in, side_in = top, side
    top = side = None
    top_img = None
    # 카드 영역: 사용자가 직접 지정한 값 > 가이드 카메라의 기준 칸 > 자동
    if top_in.data is not None:
        top, top_img = measure_photo(measurer, top_in.data, marker_cm, "top", roi_top, reference,
                                     card_roi_top or top_in.ref_roi, top_in.seed)
    if side_in.data is not None:
        side, _ = measure_photo(measurer, side_in.data, marker_cm, "side", roi_side, reference,
                                card_roi_side or side_in.ref_roi, side_in.seed)

    est: ItemEstimate = SizeEstimator().combine(top, side) if top is not None else ItemEstimate(
        errors=["가로·깊이 측정 사진(상면)을 업로드해 주세요."])

    # LLM은 카테고리/주의/방향 힌트만 판정합니다. 치수는 사용하지 않습니다.
    jpeg = cv2.imencode(".jpg", top_img)[1].tobytes() if top_img is not None else None
    advice: ItemAdvice = get_advisor().advise(target_name, jpeg)

    st.session_state.estimate = {
        "idx": target_idx, "name": target_name,
        "est": est, "top": top, "side": side, "advice": advice,
    }


def apply_estimate(row_idx: int, est: ItemEstimate) -> None:
    """추정 치수를 표의 해당 행에 적용합니다(버튼 콜백). size_lock 행은 절대 바꾸지 않습니다."""
    df = st.session_state.get("last_edited")
    if df is None or row_idx >= len(df):
        st.session_state.apply_msg = ("error", "적용할 행을 찾지 못했습니다. 다시 추정해 주세요.")
        return
    if bool(df.iloc[row_idx].get(COL_LOCK)):
        st.session_state.apply_msg = ("warning", "크기 잠금이 켜진 물건이라 적용하지 않았습니다.")
        return
    df = df.copy().reset_index(drop=True)
    for col, value in ((COL_W, est.width), (COL_D, est.depth), (COL_H, est.height)):
        if value is not None:
            df.at[row_idx, col] = float(value)
    df.at[row_idx, COL_SOURCE] = SOURCE_AI_UNCONFIRMED
    st.session_state.table_df = df
    st.session_state.editor_ver += 1  # 표를 새 데이터로 다시 그립니다.
    st.session_state.pop("last_placed", None)
    st.session_state.apply_msg = ("success", f"‘{df.at[row_idx, COL_NAME]}’의 치수를 적용했습니다. 값을 확인해 주세요.")


def show_estimate_result() -> None:
    """추정 결과 화면: 마커 감지, 치수, 신뢰도, 경고, 검증 이미지, 적용 버튼."""
    res = st.session_state.get("estimate")
    if res is None:
        st.info("사진을 올리고 ‘AI 크기 추정’을 누르면 결과가 여기에 표시됩니다.")
        return

    est: ItemEstimate = res["est"]
    top: Optional[PhotoMeasurement] = res["top"]
    side: Optional[PhotoMeasurement] = res["side"]
    advice: ItemAdvice = res["advice"]

    st.markdown(f"**추정 대상:** {res['name']} (표 {res['idx'] + 1}행)")

    marker_txt = []
    ref_label = "ArUco 마커"
    for label, m in (("상면", top), ("측면", side)):
        if m is not None:
            if m.reference == REF_CARD:
                ref_label = f"체크카드 ({CARD_W_CM} × {CARD_H_CM:.2f} cm)"
                found = "감지됨" if m.marker_found else "미감지"
            else:
                found = f"감지됨 (ID {m.marker_id})" if m.marker_found else "미감지"
            marker_txt.append(f"{label}: {found}")
    if marker_txt:
        st.caption(f"기준 물체: {ref_label} · " + " / ".join(marker_txt))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("가로 (cm)", "-" if est.width is None else f"{est.width:.1f}")
    c2.metric("깊이 (cm)", "-" if est.depth is None else f"{est.depth:.1f}")
    c3.metric("높이 (cm)", "-" if est.height is None else f"{est.height:.1f}")
    c4.metric("측정 신뢰도", f"{est.confidence_label} ({est.confidence:.0%})" if est.has_any_dim else "-")

    for err in est.errors:
        st.error(err)
    for w in dict.fromkeys(est.warnings):  # 중복 제거
        st.warning(w)

    previews = [(lbl, m) for lbl, m in (("상면 측정 검증", top), ("측면 측정 검증", side))
                if m is not None and m.preview_bgr is not None]
    if previews:
        cols = st.columns(len(previews))
        for col, (lbl, m) in zip(cols, previews):
            col.image(cv2.cvtColor(m.preview_bgr, cv2.COLOR_BGR2RGB), caption=lbl, width="stretch")
        st.caption("초록: 기준 물체(마커·카드) / 빨강: 물건 윤곽 / 파랑: 물건 최소 사각형. 윤곽이 다르면 영역을 직접 지정해 주세요.")

    st.markdown("**AI 보조 판정** (치수 아님, 참고용)")
    if advice.available:
        fragile_txt = "예" if advice.fragile else ("아니오" if advice.fragile is False else "-")
        st.info(
            f"카테고리: {advice.category or '-'} · 깨짐 주의: {fragile_txt} · "
            f"권장 방향: {advice.orientation_hint or '-'} · 신뢰도 {advice.confidence:.0%}\n\n"
            f"{advice.notes}"
        )
    else:
        st.caption(advice.error or advice.notes)

    # 적용 가능 여부 판단
    df = st.session_state.get("last_edited")
    locked = False
    if df is not None and res["idx"] < len(df):
        locked = bool(df.iloc[res["idx"]].get(COL_LOCK))
    if locked:
        st.warning("이 물건은 ‘크기 잠금’이 켜져 있어 치수를 적용할 수 없습니다. 잠금을 끄면 적용할 수 있습니다.")

    st.button(
        "이 행에 치수 적용",
        type="primary",
        width="stretch",
        disabled=locked or not est.has_any_dim,
        on_click=apply_estimate,
        args=(res["idx"], est),
        help="적용 후 ‘크기 출처’가 ‘사진 추정(확인 필요)’으로 바뀝니다.",
    )
    st.caption("적용된 값은 사용자가 확인해야 합니다. 높이가 비어 있으면 기존 높이를 그대로 둡니다.")


with tab_photo:
    st.subheader("2. 사진으로 크기 추정")
    mode = st.radio(
        "크기 기준", [MODE_CARD, MODE_MARKER], horizontal=True, key="ref_mode",
        help="사진 한 장에는 거리 정보가 없어서, 크기를 아는 물체를 함께 찍어야 cm를 계산할 수 있습니다.",
    )
    reference = REF_CARD if mode == MODE_CARD else REF_MARKER
    ref_word = "체크카드" if reference == REF_CARD else "마커"
    if reference == REF_CARD:
        st.markdown(
            "**준비:** 지갑에 있는 **체크카드·신용카드·학생증**(모두 8.56 × 5.40cm 규격)을 물건 **바로 옆**에 놓으세요.\n\n"
            "- **① 상면 사진 (가로·깊이):** 바닥에 물건과 카드를 놓고 **위에서 수직으로** 찍습니다.\n"
            "- **② 측면 사진 (높이, 선택):** 카드를 물건 옆에 **가로로 세워** 옆에서 찍습니다.\n"
            "- 카드와 **색이 다른 바닥** 위에서 찍고, 카드 네 모서리가 모두 보이게 하세요. 카드 번호 면은 가려도 됩니다."
        )
    else:
        st.markdown(
            "**준비:** 아래 마커 파일을 **배율 100%**로 인쇄해 물건 **바로 옆, 같은 평면**에 놓으세요.\n\n"
            "- **① 상면 사진 (가로·깊이):** 바닥에 물건과 마커를 놓고 **위에서 수직으로** 찍습니다.\n"
            "- **② 측면 사진 (높이, 선택):** 물건과 마커를 **옆에서** 찍습니다. 마커는 똑바로 세우세요.\n"
            "- 마커는 흐리지 않게, 가능한 한 크게 찍으세요."
        )

    if st.session_state.get("last_edited") is None or len(st.session_state.last_edited) == 0:
        st.warning("먼저 ‘1. 물건 입력’ 탭에 물건을 한 개 이상 입력해 주세요.")
    else:
        rows = st.session_state.last_edited.reset_index(drop=True)
        options = [f"{i + 1}. {str(r[COL_NAME])}" for i, r in rows.iterrows()]
        target_label = st.selectbox("추정할 물건", options, key="target_item")
        target_idx = options.index(target_label)
        if bool(rows.iloc[target_idx].get(COL_LOCK)):
            st.warning("선택한 물건은 ‘크기 잠금’이 켜져 있습니다. 측정은 할 수 있지만 표에는 적용되지 않습니다.")

        marker_cm = DEFAULT_MARKER_CM
        dict_name = "DICT_4X4_50"
        if reference == REF_MARKER:
            c1, c2 = st.columns([2, 1])
            if os.path.exists(MARKER_PDF):
                with open(MARKER_PDF, "rb") as f:
                    c1.download_button("📄 인쇄용 마커 받기 (A4, 한 변 6cm)", f.read(),
                                       file_name="SmartPack_마커_6cm_A4.pdf", mime="application/pdf",
                                       width="stretch")
            custom = c2.checkbox("마커 크기 직접 입력", key="marker_custom",
                                 help="화면에 띄운 마커나 다른 크기로 인쇄한 마커를 쓸 때만 켜세요.")
            if custom:
                c3, c4 = st.columns(2)
                marker_cm = c3.number_input("마커 한 변의 실제 길이 (cm)", min_value=0.5, max_value=50.0,
                                            value=DEFAULT_MARKER_CM, step=0.05,
                                            help="검은 사각형 바깥 테두리 한 변을 자로 잰 값")
                dict_name = c4.selectbox("ArUco 사전", list(ARUCO_DICTS.keys()), index=0,
                                         help="마커를 만들 때 쓴 사전과 같아야 합니다.")
            else:
                st.caption(f"인쇄용 마커 기준 한 변 {DEFAULT_MARKER_CM:.1f}cm로 계산합니다. "
                           "인쇄 후 확인용 자가 10cm인지 한 번 재 보세요.")
        else:
            st.caption(f"카드 규격 {CARD_W_CM} × {CARD_H_CM:.2f}cm (ISO/IEC 7810 ID-1)로 계산합니다.")

        card_roi_top = card_roi_side = None
        with st.expander("① 상면 사진 (가로·깊이) - 필수", expanded=True):
            top_in = photo_input("상면 사진", "top", reference)
            roi_top = roi_input("top")
            if reference == REF_CARD:
                card_roi_top = roi_input("top_card", "카드 영역 직접 지정 (카드를 못 찾을 때)")
        with st.expander("② 측면 사진 (높이) - 선택", expanded=False):
            side_in = photo_input("측면 사진", "side", reference)
            roi_side = roi_input("side")
            if reference == REF_CARD:
                card_roi_side = roi_input("side_card", "카드 영역 직접 지정 (카드를 못 찾을 때)")

        if st.button("✨ AI 크기 추정", type="primary", width="stretch"):
            if top_in.data is None:
                st.error("상면 사진을 먼저 올려 주세요.")
            else:
                with st.spinner(f"{ref_word} 기준으로 측정하는 중..."):
                    run_estimation(target_idx, str(rows.iloc[target_idx][COL_NAME]), marker_cm, dict_name,
                                   top_in, side_in, roi_top, roi_side, reference,
                                   card_roi_top, card_roi_side)

        if st.session_state.apply_msg:
            kind, msg = st.session_state.apply_msg
            getattr(st, kind)(msg)
            st.session_state.apply_msg = None

        st.divider()
        show_estimate_result()


# =========================================================================== #
# 탭 3: 3D 자동 배치 (기존 기능 유지)
# =========================================================================== #
with tab_result:
    st.subheader("3. 자동 배치 결과")

    # 편집 표 -> Item 객체로 변환 (잘못된 행은 제외)
    item_objects: list[Item] = []
    skipped: list[str] = []

    def to_number(v) -> Optional[float]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    for i, row in enumerate(st.session_state.last_edited.to_dict("records")):
        name = str(row.get(COL_NAME) or "").strip()
        dims = [to_number(row.get(c)) for c in (COL_W, COL_D, COL_H)]
        if not name or any(v is None or v <= 0 for v in dims):
            if name or any(v is not None for v in dims):
                skipped.append(name or f"{i + 1}번째 줄")
            continue
        item_objects.append(Item(
            name=name, width=dims[0], depth=dims[1], height=dims[2],
            color=COLORS[len(item_objects) % len(COLORS)],
            weight=to_number(row.get(COL_KG)) or 0.0,
            fragile=bool(row.get(COL_FRAGILE)),
            max_load=to_number(row.get(COL_LOAD)),
            orient=row.get(COL_ORIENT) or ORIENT_FREE,
            size_lock=bool(row.get(COL_LOCK)),
        ))
    if skipped:
        st.warning("이름이나 크기가 비어 있어 제외한 물건: " + ", ".join(skipped))

    bin_size = Bin(bin_w, bin_d, bin_h)
    signature = (tuple(
        (it.name, it.width, it.depth, it.height, it.weight, it.fragile, it.max_load, it.orient)
        for it in item_objects
    ), (bin_w, bin_d, bin_h))

    if st.button("✨ 3D 자동 정리 실행", type="primary", width="stretch"):
        placed, unpacked = Packer(bin_size).pack(item_objects)
        st.session_state.last_placed = placed
        st.session_state.last_unpacked = unpacked
        st.session_state.last_bin = (bin_w, bin_d, bin_h)
        st.session_state.last_signature = signature
        st.session_state.last_count = len(item_objects)

    if "last_placed" in st.session_state:
        placed = st.session_state.last_placed
        unpacked = st.session_state.last_unpacked
        if st.session_state.get("last_signature") != signature:
            st.warning("물건이나 수납함 설정이 바뀌었습니다. ‘3D 자동 정리 실행’을 다시 눌러 주세요.")
        else:
            total_bin_volume = bin_size.volume()
            used_volume = sum(volume(p.width, p.depth, p.height) for p in placed)
            utilization = 100 * used_volume / total_bin_volume if total_bin_volume else 0
            total_weight = sum(p.item.weight for p in placed)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("배치 성공", f"{len(placed)}개 / {st.session_state.last_count}개")
            m2.metric("공간 활용률", f"{utilization:.1f}%")
            m3.metric("총 무게", f"{total_weight:.2f} kg")
            m4.metric("미배치 물건", f"{len(unpacked)}개")

            st.plotly_chart(make_figure(placed, bin_w, bin_d, bin_h), width="stretch")

            st.subheader("배치 위치")
            if placed:
                st.dataframe([
                    {
                        "물건": p.item.name,
                        "X (cm)": round(p.x, 1),
                        "Y (cm)": round(p.y, 1),
                        "Z (cm)": round(p.z, 1),
                        "가로": round(p.width, 1),
                        "깊이": round(p.depth, 1),
                        "높이": round(p.height, 1),
                        "무게 (kg)": p.item.weight,
                        "위에 실린 무게 (kg)": round(p.carried, 2),
                        "버팀 한도": limit_text(p.item),
                        "주의": "깨짐/눌림" if p.item.fragile else "",
                        "방향": p.item.orient,
                    }
                    for p in placed
                ], width="stretch", hide_index=True)
            if unpacked:
                st.warning(
                    "다음 물건은 배치하지 못했습니다: "
                    + ", ".join(item.name for item in unpacked)
                    + ". 공간 부족 외에도 방향 고정, 받침 부족(바닥 면적의 70% 이상이 받쳐져야 함), "
                    "아래 물건의 버팀 무게 초과 때문일 수 있습니다. 이 간단한 탐색 방식이 자리를 못 찾은 경우도 포함됩니다."
                )
            st.caption(
                "공간 활용률은 배치된 물건 부피 합계 ÷ 수납함 부피입니다. "
                "무게는 받쳐 주는 면적 비율대로 아래 물건에 나눠 실린다고 단순하게 계산합니다. "
                "배치 알고리즘은 빠른 휴리스틱이며 전역 최적해를 보장하지 않습니다."
            )
    else:
        st.info("물건을 확인한 뒤 ‘3D 자동 정리 실행’을 눌러 결과를 확인하세요.")

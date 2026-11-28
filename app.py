
import streamlit as st

from .packing_core import (
    Item,
    pack_items,
    make_figure,
    volume,
    COLORS,
    ORIENT_OPTIONS,
    ORIENT_FREE
)

# 페이지 설정
st.set_page_config(
    page_title="SmartPack 3D",
    page_icon="🎒",
    layout="wide"
)

# 물건 목록 저장
if "items" not in st.session_state:
    st.session_state["items"] = []

st.title("🎒 SmartPack 3D")
st.subheader("AI 사물 인식 기반 스마트 가방 정리 서비스")
st.write("가방에 넣을 물건을 등록하고 효율적으로 정리해 보세요!")

st.divider()

# 1. 물건 등록
st.header("1. 물건 등록")

tab1, tab2 = st.tabs(["📷 카메라 촬영", "🎤 음성 입력"])

with tab1:
    st.write("가방에 넣을 물건을 촬영하세요.")
    image = st.camera_input("물건 촬영하기")

    if image is not None:
        st.image(image, caption="촬영한 물건")

with tab2:
    st.write("음성 파일을 등록하세요.")
    audio = st.file_uploader(
        "음성 파일 선택",
        type=["wav", "mp3", "m4a"]
    )

    if audio is not None:
        st.audio(audio)

st.divider()

# 2. 물건 직접 등록
st.header("2. 물건 직접 등록")

with st.form("item_form"):
    name = st.text_input("물건 이름")

    col1, col2, col3 = st.columns(3)

    with col1:
        width = st.number_input(
            "가로(cm)", min_value=0.1, value=10.0
        )

    with col2:
        depth = st.number_input(
            "깊이(cm)", min_value=0.1, value=10.0
        )

    with col3:
        height = st.number_input(
            "높이(cm)", min_value=0.1, value=10.0
        )

    weight = st.number_input(
        "무게(kg)", min_value=0.0, value=0.5
    )

    fragile = st.checkbox("깨짐/눌림 주의 물건")

    orient = st.selectbox(
        "물건 방향",
        options=ORIENT_OPTIONS,
        index=0
    )

    submitted = st.form_submit_button("➕ 물건 추가")

    if submitted:
        if name.strip():
            st.session_state["items"].append({
                "id": max(
                    [item["id"] for item in st.session_state["items"]],
                    default=0
                ) + 1,
                "name": name.strip(),
                "width": width,
                "depth": depth,
                "height": height,
                "weight": weight,
                "fragile": fragile,
                "orient": orient
            })

            st.success(f"{name} 등록 완료!")
        else:
            st.warning("물건 이름을 입력해 주세요.")

st.divider()

# 3. 물건 목록 및 삭제
st.header("3. 등록된 물건 목록")

st.write(
    f"현재 등록된 물건: {len(st.session_state['items'])}개"
)

if st.session_state["items"]:
    for item in st.session_state["items"]:
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(
                f"📦 **{item['name']}** "
                f"({item['width']} × {item['depth']} × "
                f"{item['height']}cm)"
            )

        with col2:
            if st.button("🗑️ 삭제", key=f"delete_{item['id']}"):
                st.session_state["items"].remove(item)
                st.rerun()

else:
    st.info("아직 등록된 물건이 없습니다.")

st.divider()

# 4. 가방 크기 설정
st.header("4. 가방 크기 설정")

col1, col2, col3 = st.columns(3)

with col1:
    bin_w = st.number_input(
        "가방 가로(cm)", min_value=1.0, value=40.0
    )

with col2:
    bin_d = st.number_input(
        "가방 깊이(cm)", min_value=1.0, value=30.0
    )

with col3:
    bin_h = st.number_input(
        "가방 높이(cm)", min_value=1.0, value=25.0
    )

st.caption(
    f"가방 전체 부피: {volume(bin_w, bin_d, bin_h):,.0f} cm³"
)

st.divider()

# 5. 3D 자동 배치 결과
st.header("5. 3D 가방 배치 결과")

# 화면에 등록된 정보를 알고리즘의 Item 객체로 변환
item_objects = []

for i, item in enumerate(st.session_state["items"]):
    item_objects.append(
        Item(
            name=item["name"],
            width=item["width"],
            depth=item["depth"],
            height=item["height"],
            color=COLORS[i % len(COLORS)],
            weight=item["weight"],
            fragile=item["fragile"],
            orient=item["orient"]
        )
    )

# 물건 및 가방 크기 변경 여부 확인
signature = (
    tuple(
        (
            item.name,
            item.width,
            item.depth,
            item.height,
            item.weight,
            item.fragile,
            item.orient
        )
        for item in item_objects
    ),
    (bin_w, bin_d, bin_h)
)

if st.button("✨ 3D 자동 정리 실행", type="primary"):
    placed, unpacked = pack_items(
        item_objects,
        bin_w,
        bin_d,
        bin_h
    )

    st.session_state["last_placed"] = placed
    st.session_state["last_unpacked"] = unpacked
    st.session_state["last_signature"] = signature
    st.session_state["last_bin"] = (bin_w, bin_d, bin_h)

# 실행 결과 표시
if "last_placed" in st.session_state:

    if st.session_state["last_signature"] != signature:
        st.warning(
            "물건이나 가방 크기가 변경되었습니다. "
            "3D 자동 정리를 다시 실행해 주세요."
        )

    else:
        placed = st.session_state["last_placed"]
        unpacked = st.session_state["last_unpacked"]

        total_bin_volume = volume(bin_w, bin_d, bin_h)

        used_volume = sum(
            volume(p.width, p.depth, p.height)
            for p in placed
        )

        utilization = (
            100 * used_volume / total_bin_volume
            if total_bin_volume else 0
        )

        total_weight = sum(
            p.item.weight for p in placed
        )

        # 결과 요약
        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "배치 성공",
            f"{len(placed)}개 / {len(item_objects)}개"
        )

        m2.metric(
            "공간 활용률",
            f"{utilization:.1f}%"
        )

        m3.metric(
            "총 무게",
            f"{total_weight:.2f} kg"
        )

        m4.metric(
            "미배치 물건",
            f"{len(unpacked)}개"
        )

        # 3D 그래프
        st.subheader("3D 자동 배치 시각화")

        fig = make_figure(
            placed,
            bin_w,
            bin_d,
            bin_h
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # 배치 위치 표시
        st.subheader("물건별 배치 위치")

        if placed:
            st.dataframe([
                {
                    "물건": p.item.name,
                    "X": round(p.x, 1),
                    "Y": round(p.y, 1),
                    "Z": round(p.z, 1),
                    "가로": round(p.width, 1),
                    "깊이": round(p.depth, 1),
                    "높이": round(p.height, 1)
                }
                for p in placed
            ], use_container_width=True)

        if unpacked:
            st.warning(
                "배치하지 못한 물건: "
                + ", ".join(item.name for item in unpacked)
            )

else:
    st.info(
        "물건을 등록하고 가방 크기를 설정한 뒤 "
        "'3D 자동 정리 실행'을 눌러 주세요."
    )

"""
SmartPack 3D — 간단한 3D 자동 정리 프로토타입
실행:
    pip install streamlit plotly
    streamlit run smartpack3d.py

물건의 크기·무게·주의사항을 입력받아 직육면체 물건을 수납함에 자동 배치합니다.
- 큰 물건부터, 튼튼한 물건을 먼저 넣고 깨짐/눌림 주의 물건은 나중에(위쪽에) 넣습니다.
- 물건은 아래에서 충분히 받쳐져야 하며(지지 검사), 각 물건이 버틸 수 있는 무게를 넘지 않게 쌓습니다.
- 방향 고정(세우기 고정/완전 고정)을 지원합니다.
"""

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 물건 바닥 면적 중 이 비율 이상이 아래 물건(또는 바닥)에 닿아 있어야 놓을 수 있습니다.
SUPPORT_RATIO = 0.7

ORIENT_FREE = "자유"
ORIENT_UPRIGHT = "세우기 고정"
ORIENT_FIXED = "완전 고정"
ORIENT_OPTIONS = [ORIENT_FREE, ORIENT_UPRIGHT, ORIENT_FIXED]


@dataclass
class Item:
    name: str
    width: float
    depth: float
    height: float
    color: str
    weight: float = 0.0              # kg
    fragile: bool = False            # 깨짐/눌림 주의
    max_load: Optional[float] = None  # 위에 버틸 수 있는 무게(kg), None이면 제한 없음
    orient: str = ORIENT_FREE
    size_lock: bool = False          # AI 자동 크기 추정 시 덮어쓰지 않음

    def load_limit(self) -> float:
        """위에 올라갈 수 있는 최대 무게. 주의 물건인데 값이 비어 있으면 0kg(아무것도 못 올림)."""
        if self.max_load is not None:
            return self.max_load
        return 0.0 if self.fragile else math.inf


@dataclass
class PlacedItem:
    item: Item
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    supports: List[Tuple[int, float]] = field(default_factory=list)  # (아래 물건 번호, 하중 분담 비율)
    carried: float = 0.0  # 이 물건 위에 실린 총 무게(kg)


COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def volume(w: float, d: float, h: float) -> float:
    return w * d * h


def overlaps(a: PlacedItem, b: PlacedItem, eps: float = 1e-8) -> bool:
    return (
        a.x < b.x + b.width - eps
        and a.x + a.width > b.x + eps
        and a.y < b.y + b.depth - eps
        and a.y + a.depth > b.y + eps
        and a.z < b.z + b.height - eps
        and a.z + a.height > b.z + eps
    )


def inside_bin(p: PlacedItem, bw: float, bd: float, bh: float) -> bool:
    return (
        p.x >= 0 and p.y >= 0 and p.z >= 0
        and p.x + p.width <= bw + 1e-8
        and p.y + p.depth <= bd + 1e-8
        and p.z + p.height <= bh + 1e-8
    )


def candidate_positions(placed: List[PlacedItem]) -> List[Tuple[float, float, float]]:
    """물건 모서리에서 새 배치를 시도하는 단순한 극점 후보 생성."""
    points = {(0.0, 0.0, 0.0)}
    for p in placed:
        points.add((p.x + p.width, p.y, p.z))
        points.add((p.x, p.y + p.depth, p.z))
        points.add((p.x, p.y, p.z + p.height))
        points.add((p.x + p.width, p.y + p.depth, p.z))
        points.add((p.x + p.width, p.y, p.z + p.height))
        points.add((p.x, p.y + p.depth, p.z + p.height))
    return sorted(points, key=lambda q: (q[2], q[1], q[0]))


def orientations(item: Item) -> List[Tuple[float, float, float]]:
    """방향 고정 설정에 따라 시도할 (가로, 깊이, 높이) 목록."""
    w, d, h = item.width, item.depth, item.height
    if item.orient == ORIENT_FIXED:
        return [(w, d, h)]
    if item.orient == ORIENT_UPRIGHT:
        # 높이는 그대로 두고 바닥에서만 90도 회전
        return list(dict.fromkeys([(w, d, h), (d, w, h)]))
    return list(dict.fromkeys(permutations((w, d, h))))


def overlap_area(c: PlacedItem, p: PlacedItem) -> float:
    ox = min(c.x + c.width, p.x + p.width) - max(c.x, p.x)
    oy = min(c.y + c.depth, p.y + p.depth) - max(c.y, p.y)
    return max(ox, 0.0) * max(oy, 0.0)


def find_supports(candidate: PlacedItem, placed: List[PlacedItem]) -> Optional[List[Tuple[int, float]]]:
    """후보 바로 아래에서 받쳐 주는 물건과 분담 비율. 받침이 부족하면 None."""
    if candidate.z < 1e-6:
        return []  # 바닥에 놓임
    areas = []
    for idx, p in enumerate(placed):
        if abs(p.z + p.height - candidate.z) < 1e-6:
            a = overlap_area(candidate, p)
            if a > 1e-9:
                areas.append((idx, a))
    total = sum(a for _, a in areas)
    if total < SUPPORT_RATIO * candidate.width * candidate.depth:
        return None
    return [(idx, a / total) for idx, a in areas]


def added_loads(supports: List[Tuple[int, float]], weight: float, placed: List[PlacedItem]) -> Dict[int, float]:
    """새 물건의 무게가 아래 물건들에 얼마씩 더해지는지 (아래로 계속 전달)."""
    acc: Dict[int, float] = {}

    def push(idx: int, amount: float):
        acc[idx] = acc.get(idx, 0.0) + amount
        for j, share in placed[idx].supports:
            push(j, amount * share)

    for idx, share in supports:
        push(idx, weight * share)
    return acc


def find_position(
    item: Item,
    placed: List[PlacedItem],
    bw: float,
    bd: float,
    bh: float,
) -> Optional[Tuple[PlacedItem, Dict[int, float]]]:
    """조건(겹침·범위·지지·하중)을 만족하는 후보 중 낮고 빈틈이 적은 자리를 선택."""
    best = None
    best_score = None

    for w, d, h in orientations(item):
        if w > bw or d > bd or h > bh:
            continue

        for x, y, z in candidate_positions(placed):
            candidate = PlacedItem(item, x, y, z, w, d, h)
            if not inside_bin(candidate, bw, bd, bh):
                continue
            if any(overlaps(candidate, other) for other in placed):
                continue
            supports = find_supports(candidate, placed)
            if supports is None:
                continue  # 공중에 뜨거나 너무 조금 걸침
            loads = added_loads(supports, item.weight, placed)
            if any(placed[i].carried + add > placed[i].item.load_limit() + 1e-9 for i, add in loads.items()):
                continue  # 아래 물건이 눌림
            candidate.supports = supports

            leftover = (
                (bw - (x + w)) * d * h
                + (bd - (y + d)) * w * h
                + (bh - (z + h)) * w * d
            )
            score = (z, leftover, y, x)
            if best_score is None or score < best_score:
                best_score = score
                best = (candidate, loads)

    return best


def pack_items(items: List[Item], bw: float, bd: float, bh: float):
    # 튼튼한 물건 먼저 → 큰 부피 → 무거운 순. 주의 물건은 나중에 넣어 위쪽에 오게 합니다.
    ordered = sorted(
        items,
        key=lambda it: (it.fragile, -volume(it.width, it.depth, it.height), -it.weight),
    )
    placed: List[PlacedItem] = []
    unpacked: List[Item] = []

    for item in ordered:
        found = find_position(item, placed, bw, bd, bh)
        if found is None:
            unpacked.append(item)
            continue
        pos, loads = found
        for i, add in loads.items():
            placed[i].carried += add
        placed.append(pos)
    return placed, unpacked


def limit_text(item: Item) -> str:
    lim = item.load_limit()
    return "제한 없음" if math.isinf(lim) else f"{lim:g} kg"


def placed_hover(p: PlacedItem) -> str:
    it = p.item
    flags = []
    if it.fragile:
        flags.append("깨짐/눌림 주의")
    if it.orient != ORIENT_FREE:
        flags.append(it.orient)
    return (
        f"<br>무게: {it.weight:g} kg"
        f"<br>위에 실린 무게: {p.carried:.2f} kg / 한도 {limit_text(it)}"
        + (f"<br>{', '.join(flags)}" if flags else "")
    )


def cuboid_trace(x, y, z, w, d, h, color, name, extra=""):
    # Plotly Mesh3d로 직육면체의 8개 꼭짓점과 12개 삼각형을 그립니다.
    vertices = [
        (x, y, z), (x+w, y, z), (x+w, y+d, z), (x, y+d, z),
        (x, y, z+h), (x+w, y, z+h), (x+w, y+d, z+h), (x, y+d, z+h),
    ]
    xs, ys, zs = zip(*vertices)
    # 6개 면 × 삼각형 2개 = 12개 (아래, 위, 앞, 뒤, 왼쪽, 오른쪽 순)
    i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=1.0, name=name,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.05),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"위치: ({x:.1f}, {y:.1f}, {z:.1f}) cm<br>"
            f"크기: {w:.1f} × {d:.1f} × {h:.1f} cm"
            f"{extra}"
            "<extra></extra>"
        ),
        flatshading=True,
    )


def make_figure(placed, bw, bd, bh):
    fig = go.Figure()

    # 투명한 수납함을 나타내는 모서리 선
    corners = [
        (0,0,0), (bw,0,0), (bw,bd,0), (0,bd,0), (0,0,0),
        (0,0,bh), (bw,0,bh), (bw,bd,bh), (0,bd,bh), (0,0,bh),
        (0,bd,bh), (0,bd,0), (bw,bd,0), (bw,0,0), (bw,0,bh),
    ]
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in corners], y=[p[1] for p in corners],
        z=[p[2] for p in corners], mode="lines",
        line=dict(color="#333333", width=5), name="수납함",
        hoverinfo="skip",
    ))

    for idx, p in enumerate(placed):
        fig.add_trace(cuboid_trace(
            p.x, p.y, p.z, p.width, p.depth, p.height,
            p.item.color, p.item.name, placed_hover(p),
        ))

    # 물건끼리 구분되도록 각 상자의 모서리를 검은 선으로 그립니다.
    ex, ey, ez = [], [], []
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for p in placed:
        v = [
            (p.x, p.y, p.z), (p.x+p.width, p.y, p.z),
            (p.x+p.width, p.y+p.depth, p.z), (p.x, p.y+p.depth, p.z),
            (p.x, p.y, p.z+p.height), (p.x+p.width, p.y, p.z+p.height),
            (p.x+p.width, p.y+p.depth, p.z+p.height), (p.x, p.y+p.depth, p.z+p.height),
        ]
        for a, b in edges:
            ex += [v[a][0], v[b][0], None]
            ey += [v[a][1], v[b][1], None]
            ez += [v[a][2], v[b][2], None]
    if placed:
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#222222", width=3),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="가로 X (cm)", yaxis_title="깊이 Y (cm)",
            zaxis_title="높이 Z (cm)",
            xaxis=dict(range=[0, bw], backgroundcolor="rgba(240,240,240,0.25)"),
            yaxis=dict(range=[0, bd], backgroundcolor="rgba(240,240,240,0.25)"),
            zaxis=dict(range=[0, bh], backgroundcolor="rgba(240,240,240,0.25)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        updatemenus=[
            # 블렌더 숫자패드처럼 시점을 바로 바꾸는 버튼
            dict(
                type="buttons", direction="right", showactive=True,
                x=0, y=1.08, xanchor="left", yanchor="bottom",
                pad=dict(r=6, t=0),
                buttons=[
                    view_button("3D", (1.6, -1.6, 1.2), (0, 0, 1)),
                    view_button("위", (0, 0, 2.5), (0, 1, 0)),
                    view_button("앞", (0, -2.5, 0), (0, 0, 1)),
                    view_button("옆", (2.5, 0, 0), (0, 0, 1)),
                ],
            ),
            # 원근(보이는 그대로) / 정투영(블렌더 Ortho, 거리와 상관없이 같은 크기)
            dict(
                type="buttons", direction="right", showactive=True,
                x=0.42, y=1.08, xanchor="left", yanchor="bottom",
                pad=dict(r=6, t=0),
                buttons=[
                    dict(label="원근", method="relayout",
                         args=[{"scene.camera.projection.type": "perspective"}]),
                    dict(label="정투영", method="relayout",
                         args=[{"scene.camera.projection.type": "orthographic"}]),
                ],
            ),
        ],
    )
    return fig


def view_button(label, eye, up):
    # 카메라 위치(eye)와 화면 위쪽 방향(up)만 바꿔서 원근/정투영 설정은 유지합니다.
    return dict(
        label=label, method="relayout",
        args=[{
            "scene.camera.eye": dict(x=eye[0], y=eye[1], z=eye[2]),
            "scene.camera.up": dict(x=up[0], y=up[1], z=up[2]),
            "scene.camera.center": dict(x=0, y=0, z=0),
        }],
    )




st.set_page_config(page_title="SmartPack 3D", page_icon="📦", layout="wide")
st.title("📦 SmartPack 3D")
st.caption("공간 활용형 3D 자동 정리 프로토타입 · 크기는 cm, 무게는 kg")

with st.sidebar:
    st.header("수납함 설정")
    bin_w = st.number_input("가로 (cm)", min_value=1.0, max_value=300.0, value=40.0, step=1.0)
    bin_d = st.number_input("깊이 (cm)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    bin_h = st.number_input("높이 (cm)", min_value=1.0, max_value=300.0, value=25.0, step=1.0)
    st.info(f"전체 부피: {volume(bin_w, bin_d, bin_h):,.0f} cm³")

COL_NAME, COL_W, COL_D, COL_H = "물건", "가로(cm)", "깊이(cm)", "높이(cm)"
COL_KG, COL_FRAGILE, COL_LOAD = "무게(kg)", "깨짐/눌림 주의", "버팀 무게(kg)"
COL_ORIENT, COL_LOCK = "방향", "크기 잠금"

EXAMPLE_ITEMS = [
    {COL_NAME: "책", COL_W: 20.0, COL_D: 15.0, COL_H: 4.0, COL_KG: 0.5, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False},
    {COL_NAME: "수납 상자", COL_W: 12.0, COL_D: 10.0, COL_H: 8.0, COL_KG: 0.6, COL_FRAGILE: False, COL_LOAD: 2.0, COL_ORIENT: ORIENT_FREE, COL_LOCK: False},
    {COL_NAME: "필통", COL_W: 18.0, COL_D: 6.0, COL_H: 5.0, COL_KG: 0.3, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False},
    {COL_NAME: "텀블러", COL_W: 8.0, COL_D: 8.0, COL_H: 20.0, COL_KG: 0.4, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_UPRIGHT, COL_LOCK: False},
    {COL_NAME: "과자 상자", COL_W: 15.0, COL_D: 10.0, COL_H: 6.0, COL_KG: 0.1, COL_FRAGILE: True, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False},
    {COL_NAME: "작은 상자", COL_W: 8.0, COL_D: 8.0, COL_H: 7.0, COL_KG: 0.2, COL_FRAGILE: False, COL_LOAD: None, COL_ORIENT: ORIENT_FREE, COL_LOCK: False},
]

st.subheader("1. 정리할 물건 입력")
st.write(
    "표를 직접 클릭해서 고치세요. 맨 아래 빈 줄에 입력하면 물건이 추가되고, "
    "줄 왼쪽을 선택한 뒤 휴지통을 누르면 삭제됩니다."
)

if "editor_ver" not in st.session_state:
    st.session_state.editor_ver = 0

edited = st.data_editor(
    pd.DataFrame(EXAMPLE_ITEMS),
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
            help="나중에 AI가 사진으로 크기를 추정해도 이 물건의 크기는 바꾸지 않습니다.",
        ),
    },
)

if st.button("예시 물건으로 초기화"):
    st.session_state.editor_ver += 1
    st.session_state.pop("last_placed", None)
    st.rerun()


def to_number(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


item_objects: List[Item] = []
skipped = []
for i, row in enumerate(edited.to_dict("records")):
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

st.divider()
st.subheader("2. 자동 배치 결과")
signature = (tuple(
    (it.name, it.width, it.depth, it.height, it.weight, it.fragile, it.max_load, it.orient)
    for it in item_objects
), (bin_w, bin_d, bin_h))

if st.button("✨ 3D 자동 정리 실행", type="primary", width="stretch"):
    placed, unpacked = pack_items(item_objects, bin_w, bin_d, bin_h)
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
        total_bin_volume = volume(bin_w, bin_d, bin_h)
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

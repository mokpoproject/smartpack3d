"""
SmartPack 3D — 간단한 3D 자동 정리 프로토타입
실행:
    pip install streamlit plotly
    streamlit run smartpack3d.py

물건의 크기·무게·주의사항을 입력받아 직육면체 물건을 수납함에 자동 배치합니다.
- 큰 물건부터, 튼튼한 물건을 먼저 넣고 깨짐/눌림 주의 물건은 나중에(위쪽에) 넣습니다.
- 물건은 아래에서 충분히 받쳐져야 하며(지지 검사), 각 물건이 버틸 수 있는 무게를 넘지 않게 쌓습니다.
- 방향 고정(세우기 고정/완전 고정)을 지원합니다.
- 빈틈 줄이기: 투영 극점(Projected Extreme Point)으로 틈 안쪽 자리까지 후보로 만들고,
  벽·바닥·다른 물건과 맞닿는 면적이 넓은 자리를 우선해 붙여 넣습니다.
"""

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def placement_score(x, y, z, w, d, h, leftover, contact):
    """값이 작을수록 좋은 자리."""
    # 맞닿는 면적이 넓은 자리 우선 → 같으면 낮은 자리 → 남는 공간이 적은 자리
    return (-round(contact, 6), z, leftover, y, x)


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


def _project(px: float, py: float, pz: float, axis: int, placed: List[PlacedItem]) -> float:
    """점 (px, py, pz)를 axis 방향(0=x, 1=y, 2=z)으로 원점 쪽으로 밀었을 때 처음 닿는 좌표.
    벽(0) 또는 그 방향에 있는 물건의 면에서 멈춥니다."""
    pos = (px, py, pz)
    stop = 0.0
    for q in placed:
        lo = (q.x, q.y, q.z)
        hi = (q.x + q.width, q.y + q.depth, q.z + q.height)
        # 나머지 두 축에서 점이 물건의 면 범위 안에 있어야 부딪힘
        if all(lo[k] - 1e-9 <= pos[k] < hi[k] - 1e-9 for k in range(3) if k != axis):
            if hi[axis] <= pos[axis] + 1e-9:
                stop = max(stop, hi[axis])
    return stop


def candidate_positions(placed: List[PlacedItem]) -> List[Tuple[float, float, float]]:
    """투영 극점(Projected Extreme Point) 후보 생성.
    놓인 물건의 세 모서리 점(오른쪽·뒤쪽·위쪽)을 만들고, 각 점을 나머지 두 방향으로
    벽이나 다른 물건에 닿을 때까지 밀어서 틈 안쪽 자리까지 후보로 만듭니다."""
    points = {(0.0, 0.0, 0.0)}
    for p in placed:
        corners = [
            ((p.x + p.width, p.y, p.z), 0),   # 오른쪽 모서리 → y, z 방향으로 투영
            ((p.x, p.y + p.depth, p.z), 1),   # 뒤쪽 모서리 → x, z 방향으로 투영
            ((p.x, p.y, p.z + p.height), 2),  # 위쪽 모서리 → x, y 방향으로 투영
        ]
        for (cx, cy, cz), own in corners:
            points.add((cx, cy, cz))
            for axis in range(3):
                if axis == own:
                    continue
                pt = [cx, cy, cz]
                pt[axis] = _project(cx, cy, cz, axis, placed)
                points.add(tuple(pt))
    return sorted(points, key=lambda q: (q[2], q[1], q[0]))


def contact_ratio(c: PlacedItem, placed: List[PlacedItem], bw: float, bd: float, bh: float) -> float:
    """후보 물건의 겉면 중 벽·바닥·다른 물건과 맞닿는 면적의 비율(0~1)."""
    w, d, h = c.width, c.depth, c.height
    total = 2 * (w * d + w * h + d * h)
    touch = 0.0
    # 수납함 벽과 바닥 (뚜껑 쪽 윗면은 제외)
    if c.x < 1e-6:
        touch += d * h
    if abs(c.x + w - bw) < 1e-6:
        touch += d * h
    if c.y < 1e-6:
        touch += w * h
    if abs(c.y + d - bd) < 1e-6:
        touch += w * h
    if c.z < 1e-6:
        touch += w * d

    def span(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    for q in placed:
        ox = span(c.x, c.x + w, q.x, q.x + q.width)
        oy = span(c.y, c.y + d, q.y, q.y + q.depth)
        oz = span(c.z, c.z + h, q.z, q.z + q.height)
        if abs(c.x - (q.x + q.width)) < 1e-6 or abs(c.x + w - q.x) < 1e-6:
            touch += oy * oz
        if abs(c.y - (q.y + q.depth)) < 1e-6 or abs(c.y + d - q.y) < 1e-6:
            touch += ox * oz
        if abs(c.z - (q.z + q.height)) < 1e-6 or abs(c.z + h - q.z) < 1e-6:
            touch += ox * oy
    return touch / total if total else 0.0


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

    points = candidate_positions(placed)  # 방향마다 다시 계산하지 않도록 한 번만 생성
    for w, d, h in orientations(item):
        if w > bw or d > bd or h > bh:
            continue

        for x, y, z in points:
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
            contact = contact_ratio(candidate, placed, bw, bd, bh)
            score = placement_score(x, y, z, w, d, h, leftover, contact)
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

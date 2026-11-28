"""
plot3d.py - 배치 결과를 Plotly 3D 그림으로 그리는 모듈 (표현 계층)

packing.py의 데이터 모델(PlacedItem)만 사용합니다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import plotly.graph_objects as go

from packing import ORIENT_FREE, PlacedItem, limit_text


def view_button(label: str, eye: tuple, up: tuple) -> Dict[str, Any]:
    """카메라 위치(eye)와 화면 위쪽 방향(up)만 바꾸는 버튼 정의. 원근/정투영 설정은 유지합니다."""
    return dict(
        label=label, method="relayout",
        args=[{
            "scene.camera.eye": dict(x=eye[0], y=eye[1], z=eye[2]),
            "scene.camera.up": dict(x=up[0], y=up[1], z=up[2]),
            "scene.camera.center": dict(x=0, y=0, z=0),
        }],
    )


def placed_hover(p: PlacedItem) -> str:
    """배치 물건에 마우스를 올렸을 때 보여줄 설명 문자열."""
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


def cuboid_trace(x: float, y: float, z: float, w: float, d: float, h: float,
                 color: str, name: str, extra: str = "") -> go.Mesh3d:
    """직육면체 한 개를 Mesh3d(꼭짓점 8개, 삼각형 12개)로 그립니다."""
    vertices = [
        (x, y, z), (x + w, y, z), (x + w, y + d, z), (x, y + d, z),
        (x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h),
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


def make_figure(placed: Sequence[PlacedItem], bw: float, bd: float, bh: float) -> go.Figure:
    """수납함과 배치된 물건들을 하나의 3D Figure로 만듭니다."""
    fig = go.Figure()

    # 투명한 수납함을 나타내는 모서리 선
    corners = [
        (0, 0, 0), (bw, 0, 0), (bw, bd, 0), (0, bd, 0), (0, 0, 0),
        (0, 0, bh), (bw, 0, bh), (bw, bd, bh), (0, bd, bh), (0, 0, bh),
        (0, bd, bh), (0, bd, 0), (bw, bd, 0), (bw, 0, 0), (bw, 0, bh),
    ]
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in corners], y=[p[1] for p in corners],
        z=[p[2] for p in corners], mode="lines",
        line=dict(color="#333333", width=5), name="수납함",
        hoverinfo="skip",
    ))

    for p in placed:
        fig.add_trace(cuboid_trace(
            p.x, p.y, p.z, p.width, p.depth, p.height,
            p.item.color, p.item.name, placed_hover(p),
        ))

    # 물건끼리 구분되도록 각 상자의 모서리를 검은 선으로 그립니다.
    ex: List[Any] = []
    ey: List[Any] = []
    ez: List[Any] = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for p in placed:
        v = [
            (p.x, p.y, p.z), (p.x + p.width, p.y, p.z),
            (p.x + p.width, p.y + p.depth, p.z), (p.x, p.y + p.depth, p.z),
            (p.x, p.y, p.z + p.height), (p.x + p.width, p.y, p.z + p.height),
            (p.x + p.width, p.y + p.depth, p.z + p.height), (p.x, p.y + p.depth, p.z + p.height),
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
            # 시점을 바로 바꾸는 버튼
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
            # 원근 / 정투영
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

"""
packing.py - 데이터 모델과 3D 자동 배치 알고리즘

이 모듈은 Streamlit, OpenCV, LLM에 의존하지 않는 순수 Python 모듈입니다.
(팀 개발 시 단위 테스트를 쉽게 하기 위해 UI/비전/LLM 코드와 분리되어 있습니다.)

주요 구성:
- Item / PlacedItem : 물건 입력 데이터, 배치 결과 데이터
- Packer            : 수납함(Bin) 크기를 받아 물건을 배치하는 클래스 (pack_items를 감쌈)
- 하중·지지 검사 함수들 : 기존 프로토타입 로직을 그대로 유지
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

# 물건 바닥 면적 중 이 비율 이상이 아래 물건(또는 바닥)에 닿아 있어야 놓을 수 있습니다.
SUPPORT_RATIO: float = 0.7

ORIENT_FREE = "자유"
ORIENT_UPRIGHT = "세우기 고정"
ORIENT_FIXED = "완전 고정"
ORIENT_OPTIONS: List[str] = [ORIENT_FREE, ORIENT_UPRIGHT, ORIENT_FIXED]

COLORS: List[str] = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


@dataclass
class Item:
    """물건 한 개의 입력 데이터(단위: cm, kg)."""

    name: str
    width: float
    depth: float
    height: float
    color: str
    weight: float = 0.0                # kg
    fragile: bool = False              # 깨짐/눌림 주의
    max_load: Optional[float] = None   # 위에 버틸 수 있는 무게(kg), None이면 제한 없음
    orient: str = ORIENT_FREE
    size_lock: bool = False            # True면 AI 사진 추정으로 치수를 덮어쓰지 않음

    def load_limit(self) -> float:
        """위에 올라갈 수 있는 최대 무게. 주의 물건인데 값이 비어 있으면 0kg(아무것도 못 올림)."""
        if self.max_load is not None:
            return self.max_load
        return 0.0 if self.fragile else math.inf


@dataclass
class PlacedItem:
    """수납함 안에 배치된 물건의 위치와 실제 적용 크기."""

    item: Item
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    supports: List[Tuple[int, float]] = field(default_factory=list)  # (아래 물건 번호, 하중 분담 비율)
    carried: float = 0.0  # 이 물건 위에 실린 총 무게(kg)


@dataclass(frozen=True)
class Bin:
    """수납함 크기(cm)."""

    width: float
    depth: float
    height: float

    def volume(self) -> float:
        return volume(self.width, self.depth, self.height)


def volume(w: float, d: float, h: float) -> float:
    """직육면체 부피."""
    return w * d * h


def overlaps(a: PlacedItem, b: PlacedItem, eps: float = 1e-8) -> bool:
    """두 배치 물건이 3차원으로 겹치는지 검사합니다."""
    return (
        a.x < b.x + b.width - eps
        and a.x + a.width > b.x + eps
        and a.y < b.y + b.depth - eps
        and a.y + a.depth > b.y + eps
        and a.z < b.z + b.height - eps
        and a.z + a.height > b.z + eps
    )


def inside_bin(p: PlacedItem, bw: float, bd: float, bh: float) -> bool:
    """물건이 수납함 범위 안에 있는지 검사합니다."""
    return (
        p.x >= 0 and p.y >= 0 and p.z >= 0
        and p.x + p.width <= bw + 1e-8
        and p.y + p.depth <= bd + 1e-8
        and p.z + p.height <= bh + 1e-8
    )


def candidate_positions(placed: Sequence[PlacedItem]) -> List[Tuple[float, float, float]]:
    """물건 모서리에서 새 배치를 시도하는 단순한 극점(extreme point) 후보 생성."""
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
    """두 물건의 수평 투영(바닥면) 겹침 넓이."""
    ox = min(c.x + c.width, p.x + p.width) - max(c.x, p.x)
    oy = min(c.y + c.depth, p.y + p.depth) - max(c.y, p.y)
    return max(ox, 0.0) * max(oy, 0.0)


def find_supports(candidate: PlacedItem, placed: Sequence[PlacedItem]) -> Optional[List[Tuple[int, float]]]:
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


def added_loads(supports: List[Tuple[int, float]], weight: float, placed: Sequence[PlacedItem]) -> Dict[int, float]:
    """새 물건의 무게가 아래 물건들에 얼마씩 더해지는지 계산합니다(아래 방향으로 연쇄 전달)."""
    acc: Dict[int, float] = {}

    def push(idx: int, amount: float) -> None:
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
    """조건(겹침·범위·지지·하중)을 만족하는 후보 중 낮고 빈틈이 적은 자리를 선택합니다."""
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


def pack_items(items: List[Item], bw: float, bd: float, bh: float) -> Tuple[List[PlacedItem], List[Item]]:
    """물건 목록을 수납함에 배치합니다. (배치 결과, 미배치 물건) 튜플을 반환합니다."""
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


class Packer:
    """수납함 하나에 물건 목록을 배치하는 서비스 클래스.

    UI에서는 Packer(bin).pack(items)만 호출하면 되고,
    배치 세부 규칙을 바꿀 때는 이 클래스 또는 위 함수들만 수정하면 됩니다.
    """

    def __init__(self, bin_size: Bin) -> None:
        self.bin_size = bin_size

    def pack(self, items: List[Item]) -> Tuple[List[PlacedItem], List[Item]]:
        b = self.bin_size
        return pack_items(items, b.width, b.depth, b.height)


def limit_text(item: Item) -> str:
    """버팀 한도를 사람이 읽기 쉬운 문자열로 바꿉니다."""
    lim = item.load_limit()
    return "제한 없음" if math.isinf(lim) else f"{lim:g} kg"

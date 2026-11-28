"""배치 알고리즘 기본 동작 테스트: python -m pytest -q"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packing import Item, overlaps, pack_items, volume  # noqa: E402


def random_items(seed, bw=40, bd=30, bh=25, fill=0.92):
    r = random.Random(seed)
    items, vol, i = [], 0.0, 0
    while vol < fill * bw * bd * bh:
        w = r.choice([5, 6, 8, 10, 12, 15, 20, 25])
        d = r.choice([4, 5, 6, 8, 10, 12, 15])
        h = r.choice([2, 3, 4, 5, 6, 8, 10])
        items.append(Item(f"i{i}", w, d, h, "#000", round(r.uniform(0.1, 1.5), 2)))
        vol += w * d * h
        i += 1
    return items


def test_no_overlap_and_inside_bin():
    placed, _ = pack_items(random_items(3), 40, 30, 25)
    for i, a in enumerate(placed):
        assert a.x >= 0 and a.y >= 0 and a.z >= 0
        assert a.x + a.width <= 40 + 1e-6 and a.y + a.depth <= 30 + 1e-6 and a.z + a.height <= 25 + 1e-6
        for b in placed[i + 1:]:
            assert not overlaps(a, b)


def test_fragile_item_carries_nothing():
    items = [Item("과자", 30, 20, 5, "#000", 0.1, fragile=True), Item("벽돌", 30, 20, 5, "#000", 3.0)]
    placed, _ = pack_items(items, 30, 20, 10)
    z = {p.item.name: p.z for p in placed}
    assert z["벽돌"] == 0 and z["과자"] == 5


def test_no_floating_item():
    items = [Item("작은", 5, 5, 5, "#000", 0.1, orient="완전 고정"),
             Item("큰판", 30, 20, 2, "#000", 0.1, orient="완전 고정")]
    placed, _ = pack_items(items, 30, 20, 10)
    z = {p.item.name: p.z for p in placed}
    assert z["큰판"] == 0  # 작은 물건 위에 걸쳐 뜨지 않고 바닥에 놓임


def test_improved_utilization_benchmark():
    # 빈틈 개선(투영 극점 + 접촉 면적) 기준값: 12세트 평균 89.7%
    utils = []
    for s in range(12):
        placed, _ = pack_items(random_items(s), 40, 30, 25)
        utils.append(sum(volume(p.width, p.depth, p.height) for p in placed) / (40 * 30 * 25))
    assert sum(utils) / len(utils) > 0.89

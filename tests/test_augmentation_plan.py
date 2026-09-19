"""增补方案接口端到端测试与分支限界最大独立集单测。"""

from __future__ import annotations

import os

import pytest

from app.augmentation import maximum_conflict_free_set
from app.dimer import analyze_pair


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("aug_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# 原批次：F=GGGGGAAAAA / R=GGGGGTTTTT，ct=4、tt=3
PAYLOAD = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}


@pytest.fixture(scope="module")
def batch_id(client):
    return client.post("/api/batches", json=PAYLOAD).json()["id"]


def _plan(client, batch_id, candidates, max_accepted=18):
    return client.post(
        f"/api/batches/{batch_id}/augmentation-plan",
        json={"max_accepted": max_accepted, "candidates": candidates},
    )


# 互补事实（由 analyze_pair 独立保证，见 test_pair_facts）：
#   全 G 任意长度之间：low；全 G 与 F/R：非 high/medium；
#   10A vs 10T：high；同字母 A-A / T-T：low；
#   CCCCCGGGG 自二聚体：high；CCCCCGGGGG 与全 G：连续互补 5 → high/medium。
SAFE_G = [
    {"name": "g1", "sequence": "GGG"},
    {"name": "g2", "sequence": "GGGG"},
    {"name": "g3", "sequence": "GGGGG"},
    {"name": "g4", "sequence": "GGGGGG"},
]


def test_pair_facts():
    """测试依赖的二聚体事实基线（若算法变化，测试前提需同步调整）。"""
    assert analyze_pair("g", "GGG", "g", "GGG", 4, 3, is_self_dimer=True).risk_level == "low"
    assert analyze_pair("g1", "GGG", "g2", "GGGGGG", 4, 3).risk_level == "low"
    assert analyze_pair("a", "AAAAAAAAAA", "t", "TTTTTTTTTT", 4, 3).risk_level == "high"
    assert analyze_pair("a", "AAAAAAAAAA", "a2", "AAAAAA", 4, 3).risk_level == "low"
    hairpin = analyze_pair("h", "CCCCCGGGGG", "h", "CCCCCGGGGG", 4, 3,
                           is_self_dimer=True)
    assert hairpin.risk_level == "high"
    # 字典序测试用的两条不相交边
    assert analyze_pair("c0", "CCCCCCCC", "c1", "GGGGGGGG", 4, 3).risk_level == "high"
    assert analyze_pair("c2", "TTTTT", "c3", "AAAAA", 4, 3).risk_level == "high"
    cross = [
        analyze_pair("c0", "CCCCCCCC", "c2", "TTTTT", 4, 3).risk_level,
        analyze_pair("c0", "CCCCCCCC", "c3", "AAAAA", 4, 3).risk_level,
        analyze_pair("c1", "GGGGGGGG", "c2", "TTTTT", 4, 3).risk_level,
        analyze_pair("c1", "GGGGGGGG", "c3", "AAAAA", 4, 3).risk_level,
    ]
    assert cross == ["low"] * 4
    for s in ("CCCCCCCC", "GGGGGGGG", "TTTTT", "AAAAA"):
        assert analyze_pair(s, s, s, s, 4, 3, is_self_dimer=True).risk_level == "low"


# ---------- 正常方案 ----------

def test_all_safe_candidates_selected_when_under_limit(client, batch_id):
    data = _plan(client, batch_id, SAFE_G, max_accepted=4).json()
    assert data["candidate_count"] == 4
    assert data["max_accepted"] == 4
    assert data["selected_count"] == 4
    assert data["rejected_count"] == 0
    assert [s["name"] for s in data["selected"]] == ["g1", "g2", "g3", "g4"]
    assert [s["position"] for s in data["selected"]] == [0, 1, 2, 3]
    assert data["rejected"] == []
    assert data["continuous_threshold"] == 4
    assert data["three_prime_threshold"] == 3


def test_capacity_rejects_marked_and_lex_order_preserved(client, batch_id):
    # 空冲突图 + 上限 2：前两个候选入选，其余 capacity 且无阻断配对
    data = _plan(client, batch_id, SAFE_G, max_accepted=2).json()
    assert [s["name"] for s in data["selected"]] == ["g1", "g2"]
    rejected = data["rejected"]
    assert [(r["name"], r["position"]) for r in rejected] == [
        ("g3", 2),
        ("g4", 3),
    ]
    for r in rejected:
        assert r["reject_reason"] == "capacity"
        assert r["blocked_pairs"] == []


def test_self_dimer_high_preblocks_with_blocking_pair(client, batch_id):
    cands = [
        {"name": "g1", "sequence": "GGG"},
        {"name": "hairpin", "sequence": "CCCCCGGGGG"},
    ]
    data = _plan(client, batch_id, cands, max_accepted=18).json()
    assert [s["name"] for s in data["selected"]] == ["g1"]
    rejected = {r["name"]: r for r in data["rejected"]}
    hp = rejected["hairpin"]
    assert hp["reject_reason"] == "risk"
    # 自二聚体阻断：与自身 high，且与 g1（入选候选）的异二聚体也列入
    kinds = {
        (p["counterpart_kind"], p["counterpart_name"])
        for p in hp["blocked_pairs"]
    }
    assert ("self", "hairpin") in kinds
    assert ("candidate", "g1") in kinds
    self_pair = next(p for p in hp["blocked_pairs"] if p["counterpart_kind"] == "self")
    assert self_pair["is_self_dimer"] is True
    assert self_pair["name_top"] == self_pair["name_bottom"] == "hairpin"
    assert self_pair["risk_level"] == "high"
    assert self_pair["counterpart_position"] is None
    assert self_pair["continuous_complementary_length"] == 10
    assert self_pair["three_prime_run_length"] == 10
    # 风险指标字段齐备
    for p in hp["blocked_pairs"]:
        assert p["risk_level"] in {"high", "medium"}
        assert p["risk_reasons"]
        assert p["alignment"]["top"]


def test_batch_conflict_lists_original_primer_blocker(client, batch_id):
    # TTTTTCCCCC 与 R=GGGGGTTTTT 在 3′ 端连续互补 5 → high（预先淘汰）
    cands = [
        {"name": "g1", "sequence": "GGG"},
        {"name": "bad-vs-r", "sequence": "TTTTTCCCCC"},
    ]
    data = _plan(client, batch_id, cands, max_accepted=18).json()
    assert [s["name"] for s in data["selected"]] == ["g1"]
    bad = next(r for r in data["rejected"] if r["name"] == "bad-vs-r")
    assert bad["reject_reason"] == "risk"
    batch_blockers = [
        p for p in bad["blocked_pairs"] if p["counterpart_kind"] == "batch"
    ]
    assert {p["counterpart_name"] for p in batch_blockers} <= {"F", "R"}
    r_block = next(p for p in batch_blockers if p["counterpart_name"] == "R")
    assert r_block["counterpart_position"] == 1
    assert r_block["risk_level"] == "high"
    assert r_block["three_prime_run_length"] == 5
    # 与独立重算结果一致
    expected = analyze_pair(
        "R", "GGGGGTTTTT", "bad-vs-r", "TTTTTCCCCC", 4, 3
    )
    assert r_block["best_shift"] == expected.best_shift
    assert r_block["continuous_complementary_length"] == expected.max_complementary_run
    assert r_block["risk_reasons"] == list(expected.risk_reasons)


def test_candidate_conflict_graph_exact_mis(client, batch_id):
    # 需要一个候选间冲突而与原批次无冲突的场景：使用纯 G 原批次
    g_batch = client.post(
        "/api/batches",
        json={
            "primers": [{"name": "Bg", "sequence": "ACGTACGT"}],
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    ).json()["id"]

    # 链 a1—t1—a2（10A 与 10T high；同字母 low）；MIS={a1,a2}
    chain = [
        {"name": "a1", "sequence": "AAAAAAAAAA"},
        {"name": "t1", "sequence": "TTTTTTTTTT"},
        {"name": "a2", "sequence": "AAAAAA"},
    ]
    data = _plan(client, g_batch, chain, max_accepted=18).json()
    assert [s["name"] for s in data["selected"]] == ["a1", "a2"]
    t1 = next(r for r in data["rejected"] if r["name"] == "t1")
    assert t1["reject_reason"] == "risk"
    assert {p["counterpart_name"] for p in t1["blocked_pairs"]} == {"a1", "a2"}
    assert all(p["counterpart_kind"] == "candidate" for p in t1["blocked_pairs"])
    assert {p["counterpart_position"] for p in t1["blocked_pairs"]} == {0, 2}

    # 星 a1—t1、a1—t2：{t1,t2} 大小 2 严格优于 {a1} 大小 1
    star = [
        {"name": "a1", "sequence": "AAAAAAAAAA"},
        {"name": "t1", "sequence": "TTTTTTTTTT"},
        {"name": "t2", "sequence": "TTTTT"},
    ]
    data2 = _plan(client, g_batch, star, max_accepted=18).json()
    assert [s["name"] for s in data2["selected"]] == ["t1", "t2"]
    a1 = next(r for r in data2["rejected"] if r["name"] == "a1")
    assert a1["reject_reason"] == "risk"
    assert {p["counterpart_name"] for p in a1["blocked_pairs"]} == {"t1", "t2"}


def test_lex_tie_break_on_equal_size_mis(client):
    # 纯 G 批次保证与候选无关；两条不相交 high 边：
    # c0=8C — c1=8G、c2=5T — c3=5A（交叉配对与各自自二聚体均 low）。
    # MIS 大小 2 且有多解；字典序最小取下标序列 [0,2]
    g_batch = client.post(
        "/api/batches",
        json={
            "primers": [{"name": "Bg2", "sequence": "ACGTACGT"}],
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    ).json()["id"]
    cands = [
        {"name": "c0", "sequence": "CCCCCCCC"},
        {"name": "c1", "sequence": "GGGGGGGG"},
        {"name": "c2", "sequence": "TTTTT"},
        {"name": "c3", "sequence": "AAAAA"},
    ]
    data = _plan(client, g_batch, cands, max_accepted=18).json()
    assert [s["name"] for s in data["selected"]] == ["c0", "c2"]
    # 落选者均因与入选候选冲突而 risk
    by_name = {r["name"]: r for r in data["rejected"]}
    assert by_name["c1"]["reject_reason"] == "risk"
    assert [p["counterpart_name"] for p in by_name["c1"]["blocked_pairs"]] == ["c0"]
    assert by_name["c3"]["reject_reason"] == "risk"
    assert [p["counterpart_name"] for p in by_name["c3"]["blocked_pairs"]] == ["c2"]


def test_thresholds_taken_from_batch_not_request(client, batch_id):
    # 请求体中没有阈值字段；高阈值批次下原本 high 的配对应降级
    strict = client.post(
        "/api/batches",
        json={
            "primers": [
                {"name": "F", "sequence": "GGGGGAAAAA"},
                {"name": "R", "sequence": "GGGGGTTTTT"},
            ],
            "continuous_threshold": 9,
            "three_prime_threshold": 9,
        },
    ).json()["id"]
    # GGGGGTTTTT：自二聚体 low，与 F/R 在 (9,9) 阈值下均非 high/medium
    data = _plan(
        client,
        strict,
        [
            {"name": "x", "sequence": "GGGGGTTTTT"},
            {"name": "y", "sequence": "GGG"},
        ],
        max_accepted=18,
    ).json()
    assert [s["name"] for s in data["selected"]] == ["x", "y"]


def test_deterministic_repeatable(client, batch_id):
    a = _plan(client, batch_id, SAFE_G, max_accepted=2).json()
    b = _plan(client, batch_id, SAFE_G, max_accepted=2).json()
    assert a["selected"] == b["selected"]
    assert a["rejected"] == b["rejected"]


def test_in_memory_only_no_database_write(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    resp = _plan(
        client,
        batch_id,
        SAFE_G + [{"name": "hairpin", "sequence": "CCCCCGGGGG"}],
        max_accepted=3,
    )
    assert resp.status_code == 200
    after = client.get(f"/api/batches/{batch_id}").json()
    assert after["pair_count"] == before["pair_count"] == 3
    assert [p["name"] for p in after["primers"]] == ["F", "R"]
    assert all("CCC" not in p["sequence"] for p in after["primers"])
    # 批次列表数量不增加
    listing = client.get("/api/batches").json()
    assert sum(1 for b in listing["batches"] if b["id"] == batch_id) == 1


# ---------- 错误处理 ----------

def test_batch_not_found_404(client):
    resp = _plan(
        client,
        "no-such-batch",
        [{"name": "a", "sequence": "ACGT"}, {"name": "b", "sequence": "TTTT"}],
        max_accepted=1,
    )
    assert resp.status_code == 404
    assert "不存在" in resp.json()["detail"]


def test_duplicate_candidate_names_422(client, batch_id):
    resp = _plan(
        client,
        batch_id,
        [{"name": "dup", "sequence": "ACGT"}, {"name": "dup", "sequence": "TTTT"}],
    )
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"]


def test_candidate_name_clashes_with_batch_primer_422(client, batch_id):
    resp = _plan(
        client,
        batch_id,
        [{"name": "F", "sequence": "ACGT"}, {"name": "n1", "sequence": "TTTT"}],
    )
    assert resp.status_code == 422
    assert "重名" in resp.json()["detail"]
    assert "F" in resp.json()["detail"]


def test_candidate_count_bounds_422(client, batch_id):
    one = _plan(client, batch_id, [{"name": "only", "sequence": "ACGT"}])
    assert one.status_code == 422

    nineteen = _plan(
        client,
        batch_id,
        [{"name": f"c{i}", "sequence": "ACGT"} for i in range(19)],
    )
    assert nineteen.status_code == 422


def test_max_accepted_bounds_422(client, batch_id):
    cands = [
        {"name": "a", "sequence": "ACGT"},
        {"name": "b", "sequence": "TTTT"},
    ]
    assert _plan(client, batch_id, cands, max_accepted=0).status_code == 422
    assert _plan(client, batch_id, cands, max_accepted=19).status_code == 422


def test_invalid_iupac_and_empty_name_422(client, batch_id):
    bad_seq = _plan(
        client,
        batch_id,
        [{"name": "a", "sequence": "ACGX"}, {"name": "b", "sequence": "TTTT"}],
    )
    assert bad_seq.status_code == 422

    bad_name = _plan(
        client,
        batch_id,
        [{"name": "  ", "sequence": "ACGT"}, {"name": "b", "sequence": "TTTT"}],
    )
    assert bad_name.status_code == 422


# ---------- 分支限界最大独立集单测 ----------

def test_bb_triangle_picks_single_lex_smallest():
    # 三角形（K3）任意两点相邻，MIS 大小 1；字典序最小取下标 [0]
    assert maximum_conflict_free_set(
        candidate_count=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        limit=18,
    ) == [0]
    # 上限 1 时同为 [0]
    assert maximum_conflict_free_set(
        candidate_count=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        limit=1,
    ) == [0]


def test_bb_five_cycle_picks_size_two_lex():
    # 5 环 0-1-2-3-4-0：MIS 大小 2；多解 {0,2}/{0,3}/{1,3}/{1,4}/{2,4}，
    # 字典序最小为 [0,2]
    assert maximum_conflict_free_set(
        candidate_count=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        limit=18,
    ) == [0, 2]


def test_bb_empty_graph_lex_prefix():
    # 空图：上限截断时取字典序最小的前 limit 个
    assert maximum_conflict_free_set(
        candidate_count=5, edges=[], limit=3
    ) == [0, 1, 2]
    assert maximum_conflict_free_set(
        candidate_count=3, edges=[], limit=18
    ) == [0, 1, 2]


def test_bb_preblocked_vertices_excluded():
    # 0、2 被预淘汰（自二聚体/原批次冲突）：只剩 1
    assert maximum_conflict_free_set(
        candidate_count=3, blocked=[0, 2], edges=[(0, 1)], limit=18
    ) == [1]
    # 与 blocked 相连的边在图构建时忽略
    assert maximum_conflict_free_set(
        candidate_count=4, blocked=[3], edges=[(3, 0), (3, 1)], limit=18
    ) == [0, 1, 2]


def test_bb_limit_caps_result_size():
    # 完全无边但只允许 2 个
    result = maximum_conflict_free_set(
        candidate_count=6, edges=[], limit=2
    )
    assert result == [0, 1]


def test_bb_lex_smallest_among_equal_sizes():
    # 两条不相连的边 (0,1)、(2,3)：MIS 大小 2，多解 {0,2}/{0,3}/{1,2}/{1,3}
    # 字典序最小为 [0,2]
    assert maximum_conflict_free_set(
        candidate_count=4, edges=[(0, 1), (2, 3)], limit=18
    ) == [0, 2]
    # 链 0-1-2-3：MIS={0,2} 与 {1,3} 等大，字典序 [0,2]
    assert maximum_conflict_free_set(
        candidate_count=4, edges=[(0, 1), (1, 2), (2, 3)], limit=18
    ) == [0, 2]


def test_bb_deterministic_and_edge_order_insensitive():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    a = maximum_conflict_free_set(
        candidate_count=4, edges=edges, limit=18
    )
    b = maximum_conflict_free_set(
        candidate_count=4, edges=list(reversed(edges)), limit=18
    )
    assert a == b == [0, 3]


def test_bb_all_blocked_returns_empty():
    assert maximum_conflict_free_set(
        candidate_count=2, blocked=[0, 1], edges=[(0, 1)], limit=18
    ) == []

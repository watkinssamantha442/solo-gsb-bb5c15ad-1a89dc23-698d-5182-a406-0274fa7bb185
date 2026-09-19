"""批次计算编排：枚举全部自二聚体与两两异二聚体并落库。"""

from __future__ import annotations

import json
import uuid

from . import db
from .compare import PairDiff, compare_pair_risks
from .dimer import PairResult, analyze_pair
from .pooling import build_pool_plan
from .ranking import non_dominated_layers
from .threshold_map import build_threshold_map


def _pair_to_row(pair_index: int, r: PairResult) -> tuple:
    return (
        pair_index,
        r.name_top,
        r.name_bottom,
        1 if r.is_self_dimer else 0,
        r.alignments_checked,
        r.best_shift,
        r.overlap_len,
        r.max_complementary_run,
        r.top3_run,
        r.bottom3_run,
        r.terminal_run,
        json.dumps(list(r.involved_three_prime_ends), ensure_ascii=False),
        r.risk_level,
        json.dumps(list(r.risk_reasons), ensure_ascii=False),
        r.top_line,
        r.match_line,
        r.bottom_line,
    )


def run_batch(
    *,
    primers: list[tuple[str, str]],
    continuous_threshold: int,
    three_prime_threshold: int,
) -> str:
    """计算一个批次。

    primers 为 (name, normalized_sequence) 列表。
    枚举内容：N 个自二聚体 + N*(N-1)/2 个无序异二聚体。
    """
    batch_id = str(uuid.uuid4())
    created_at = db.utc_now_iso()

    results: list[PairResult] = []
    n = len(primers)

    # 自二聚体
    for name, seq in primers:
        results.append(
            analyze_pair(
                name, seq, name, seq,
                continuous_threshold, three_prime_threshold,
                is_self_dimer=True,
            )
        )
    # 两两异二聚体（无序对；top 固定为提交顺序中靠前者）
    for i in range(n):
        for j in range(i + 1, n):
            name_i, seq_i = primers[i]
            name_j, seq_j = primers[j]
            results.append(
                analyze_pair(
                    name_i, seq_i, name_j, seq_j,
                    continuous_threshold, three_prime_threshold,
                    is_self_dimer=False,
                )
            )

    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for r in results:
        risk_counts[r.risk_level] += 1

    pair_rows = [
        _pair_to_row(idx, r) for idx, r in enumerate(results)
    ]
    primer_rows = [
        (pos, name, seq) for pos, (name, seq) in enumerate(primers)
    ]

    with db.get_conn() as conn:
        db.insert_batch(
            conn,
            batch_id=batch_id,
            created_at=created_at,
            continuous_threshold=continuous_threshold,
            three_prime_threshold=three_prime_threshold,
            primer_count=n,
            pair_count=len(results),
            risk_counts=risk_counts,
            primers=primer_rows,
            pair_rows=pair_rows,
        )
    return batch_id


def plan_batch_pools(
    *,
    batch_id: str,
    pool_count: int,
    pool_capacity: int,
    isolated_risk_levels: list[str],
) -> dict:
    """基于已保存批次的异二聚体结果构建冲突图并确定性分池。

    批次不存在返回 None（由路由层转 404）；容量总和不足抛出
    ValueError（由路由层转 422）。
    """
    with db.get_conn() as conn:
        batch = db.get_batch(conn, batch_id)
        if batch is None:
            return None
        primers = db.get_primers(conn, batch_id)
        pairs = db.get_pairs(conn, batch_id, risky_only=False)

    n = len(primers)
    if pool_count * pool_capacity < n:
        raise ValueError(
            f"池容量总和 {pool_count}×{pool_capacity}="
            f"{pool_count * pool_capacity} 不足以容纳 {n} 条引物"
        )

    # 批次内引物名称唯一（入库时已保证）
    position_by_name = {row["name"]: row["position"] for row in primers}
    names_by_position = {pos: name for name, pos in position_by_name.items()}

    # 仅取异二聚体（忽略自二聚体），且风险等级在隔离集合内
    levels = set(isolated_risk_levels)
    edges: list[tuple[int, int, str]] = []
    for row in pairs:
        if row["is_self_dimer"] or row["risk_level"] not in levels:
            continue
        edges.append(
            (
                position_by_name[row["name_top"]],
                position_by_name[row["name_bottom"]],
                row["risk_level"],
            )
        )

    plan = build_pool_plan(
        primer_count=n,
        edges=edges,
        pool_count=pool_count,
        pool_capacity=pool_capacity,
    )

    # 每个池的冲突边数
    conflict_count_by_pool = [0] * pool_count
    conflicts_out: list[dict] = []
    for pool_id, edge in plan.conflicts:
        conflict_count_by_pool[pool_id] += 1
        conflicts_out.append(
            {
                "pool": pool_id,
                "position_a": edge.position_a,
                "position_b": edge.position_b,
                "name_a": names_by_position[edge.position_a],
                "name_b": names_by_position[edge.position_b],
                "risk_level": edge.risk_level,
            }
        )

    pools_out = [
        {
            "pool": pool_id,
            "capacity": pool_capacity,
            "members": [names_by_position[pos] for pos in members],
            "member_positions": members,
            "conflict_count": conflict_count_by_pool[pool_id],
        }
        for pool_id, members in enumerate(plan.pool_members)
    ]

    return {
        "batch_id": batch_id,
        "pool_count": pool_count,
        "pool_capacity": pool_capacity,
        "isolated_risk_levels": isolated_risk_levels,
        "primer_count": n,
        "assigned_count": sum(len(m) for m in plan.pool_members),
        "total_conflicts": plan.total_conflicts,
        "pools": pools_out,
        "conflicts": conflicts_out,
    }


def _pair_result_to_out(pair_index: int, r: PairResult) -> dict:
    """内存 PairResult -> PairOut 风格字典（不落库、不经过 db.serialize_pair）。"""
    return {
        "pair_index": pair_index,
        "name_top": r.name_top,
        "name_bottom": r.name_bottom,
        "is_self_dimer": r.is_self_dimer,
        "alignments_checked": r.alignments_checked,
        "best_shift": r.best_shift,
        "overlap_len": r.overlap_len,
        "continuous_complementary_length": r.max_complementary_run,
        "three_prime_run_length": r.terminal_run,
        "top3_run_length": r.top3_run,
        "bottom3_run_length": r.bottom3_run,
        "involved_three_prime_ends": list(r.involved_three_prime_ends),
        "risk_level": r.risk_level,
        "risk_reasons": list(r.risk_reasons),
        "alignment": {
            "top": r.top_line,
            "match": r.match_line,
            "bottom": r.bottom_line,
        },
    }


def rank_replacements(
    *,
    batch_id: str,
    target_name: str,
    candidates: list[tuple[str, str]],
) -> dict | None:
    """对已保存批次做候选替换引物的内存模拟与非支配分层。

    纯读操作：读取批次阈值与引物序列后，仅在内存中复用 analyze_pair 计算
    “候选自二聚体 + 候选与其余每条引物的异二聚体”，不写库、不重算任何
    与目标无关的既有配对。

    批次不存在返回 None（路由层转 404）；目标引物不存在抛出 ValueError
    （路由层转 422）。candidates 为 (label, normalized_sequence) 列表，
    标签唯一性由路由层保证。
    """
    with db.get_conn() as conn:
        batch = db.get_batch(conn, batch_id)
        if batch is None:
            return None
        primers = db.get_primers(conn, batch_id)

    continuous_threshold = batch["continuous_threshold"]
    three_prime_threshold = batch["three_prime_threshold"]
    n = len(primers)

    target_pos = next(
        (row["position"] for row in primers if row["name"] == target_name), None
    )
    if target_pos is None:
        raise ValueError(f"批次内不存在名为 {target_name!r} 的引物")

    # 自二聚体在结果序列中占据 position 索引；异二聚体排在全部自二聚体之后，
    # 与 run_batch 的落库顺序（先 N 个自二聚体，再按 (i,j) i<j 枚举）保持一致
    self_index = target_pos

    def hetero_index(i: int, j: int) -> int:
        # 前 i 行异二聚体数 i*(2n-i-1)/2 + 第 i 行内 (i,j) 之前的 j-i-1 个
        return n + i * (2 * n - i - 1) // 2 + (j - i - 1)

    simulated: list[dict] = []
    for label, seq in candidates:
        pairs: list[tuple[int, PairResult]] = []

        # 候选自二聚体（沿用目标引物名，与批次落库命名一致）
        pairs.append(
            (
                self_index,
                analyze_pair(
                    target_name, seq, target_name, seq,
                    continuous_threshold, three_prime_threshold,
                    is_self_dimer=True,
                ),
            )
        )
        # 候选与其余引物的异二聚体：top 固定为提交顺序中靠前者
        for row in primers:
            other_pos = row["position"]
            if other_pos == target_pos:
                continue
            other_name = row["name"]
            if target_pos < other_pos:
                result = analyze_pair(
                    target_name, seq, other_name, row["sequence"],
                    continuous_threshold, three_prime_threshold,
                    is_self_dimer=False,
                )
                index = hetero_index(target_pos, other_pos)
            else:
                result = analyze_pair(
                    other_name, row["sequence"], target_name, seq,
                    continuous_threshold, three_prime_threshold,
                    is_self_dimer=False,
                )
                index = hetero_index(other_pos, target_pos)
            pairs.append((index, result))

        pairs.sort(key=lambda item: item[0])
        pair_out = [
            _pair_result_to_out(index, result) for index, result in pairs
        ]

        high_pairs = [p for p in pair_out if p["risk_level"] == "high"]
        max_three = max(p["three_prime_run_length"] for p in pair_out)
        max_continuous = max(
            p["continuous_complementary_length"] for p in pair_out
        )
        max_three_pairs = [
            p for p in pair_out if p["three_prime_run_length"] == max_three
        ]
        max_continuous_pairs = [
            p for p in pair_out
            if p["continuous_complementary_length"] == max_continuous
        ]

        simulated.append(
            {
                "label": label,
                "sequence": seq,
                "high_pair_count": len(high_pairs),
                "max_three_prime_run": max_three,
                "max_continuous_run": max_continuous,
                "high_pairs": high_pairs,
                "max_three_prime_pairs": max_three_pairs,
                "max_continuous_pairs": max_continuous_pairs,
            }
        )

    # 非支配分层：三维指标均越小越优；分层结果天然按输入顺序稳定
    layers = non_dominated_layers(
        simulated,
        lambda c: (
            c["high_pair_count"],
            c["max_three_prime_run"],
            c["max_continuous_run"],
        ),
    )
    ranked: list[dict] = []
    rank = 1
    for layer_no, layer in enumerate(layers, start=1):
        for candidate in layer:
            ranked.append({**candidate, "layer": layer_no, "rank": rank})
            rank += 1

    return {
        "batch_id": batch_id,
        "target_name": target_name,
        "continuous_threshold": continuous_threshold,
        "three_prime_threshold": three_prime_threshold,
        "primer_count": n,
        "candidate_count": len(candidates),
        "candidates": ranked,
    }


def build_batch_threshold_map(
    *,
    batch_id: str,
    continuous_start: int,
    continuous_end: int,
    three_prime_start: int,
    three_prime_end: int,
) -> dict | None:
    """已保存批次的阈值扫描地图（纯读，不重算对齐、不写库）。

    只读取每个配对已落库的最长连续互补与 3′端连续互补两个标量指标，
    用二维差分在请求阈值网格上聚合三级配对数量；单元按 3′端阈值、
    连续互补阈值升序输出，并标记批次原阈值所在单元。
    批次不存在返回 None（由路由层转 404）。
    """
    with db.get_conn() as conn:
        batch = db.get_batch(conn, batch_id)
        if batch is None:
            return None
        metrics = db.get_pair_metrics(conn, batch_id)

    grid = build_threshold_map(
        metrics,
        continuous_start=continuous_start,
        continuous_end=continuous_end,
        three_prime_start=three_prime_start,
        three_prime_end=three_prime_end,
    )

    batch_ct = batch["continuous_threshold"]
    batch_tt = batch["three_prime_threshold"]

    cells: list[dict] = []
    for r, tt in enumerate(range(three_prime_start, three_prime_end + 1)):
        for c, ct in enumerate(range(continuous_start, continuous_end + 1)):
            high, medium, low = grid[r][c]
            cells.append(
                {
                    "continuous_threshold": ct,
                    "three_prime_threshold": tt,
                    "high_count": high,
                    "medium_count": medium,
                    "low_count": low,
                    "is_batch_threshold": ct == batch_ct and tt == batch_tt,
                }
            )

    return {
        "batch_id": batch_id,
        "continuous_start": continuous_start,
        "continuous_end": continuous_end,
        "three_prime_start": three_prime_start,
        "three_prime_end": three_prime_end,
        "cell_count": len(cells),
        "pair_count": len(metrics),
        "batch_continuous_threshold": batch_ct,
        "batch_three_prime_threshold": batch_tt,
        "cells": cells,
    }


def _diff_to_out(d: PairDiff) -> dict:
    return {
        "pair_key": d.pair_key,
        "name_a": d.name_a,
        "name_b": d.name_b,
        "is_self_dimer": d.is_self_dimer,
        "sequence_changed": d.sequence_changed,
        "base_risk_level": d.base_risk_level,
        "target_risk_level": d.target_risk_level,
    }


def compare_batches(*, base_batch_id: str, target_batch_id: str) -> dict | None:
    """对比两个已保存批次的配对风险（base → target）。

    按引物名称构造稳定配对键（自二聚体保留单名，异二聚体忽略批次内
    提交顺序），比较风险等级 low<medium<high，输出 escalated /
    deescalated / introduced / resolved 四类明细与汇总。任一批次
    不存在返回 None（由路由层转 404）。
    """
    with db.get_conn() as conn:
        base = db.get_batch(conn, base_batch_id)
        target = db.get_batch(conn, target_batch_id)
        if base is None or target is None:
            return None
        base_primers = db.get_primers(conn, base_batch_id)
        target_primers = db.get_primers(conn, target_batch_id)
        base_pairs = db.get_pairs(conn, base_batch_id, risky_only=False)
        target_pairs = db.get_pairs(conn, target_batch_id, risky_only=False)

    result = compare_pair_risks(
        base_pairs=[
            (r["name_top"], r["name_bottom"], bool(r["is_self_dimer"]), r["risk_level"])
            for r in base_pairs
        ],
        target_pairs=[
            (r["name_top"], r["name_bottom"], bool(r["is_self_dimer"]), r["risk_level"])
            for r in target_pairs
        ],
        base_sequences={r["name"]: r["sequence"] for r in base_primers},
        target_sequences={r["name"]: r["sequence"] for r in target_primers},
    )

    return {
        "summary": {
            "base_batch_id": base_batch_id,
            "target_batch_id": target_batch_id,
            "base_pair_count": len(base_pairs),
            "target_pair_count": len(target_pairs),
            "common_pair_count": result.common_pair_count,
            "unchanged_pair_count": result.unchanged_pair_count,
            "escalated_count": len(result.escalated),
            "deescalated_count": len(result.deescalated),
            "introduced_count": len(result.introduced),
            "resolved_count": len(result.resolved),
        },
        "escalated": [_diff_to_out(d) for d in result.escalated],
        "deescalated": [_diff_to_out(d) for d in result.deescalated],
        "introduced": [_diff_to_out(d) for d in result.introduced],
        "resolved": [_diff_to_out(d) for d in result.resolved],
    }

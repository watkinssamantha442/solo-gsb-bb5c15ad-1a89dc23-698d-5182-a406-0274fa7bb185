"""冲突图上的最大无冲突候选集合（确定性分支限界精确求解）。

候选引物因 high/medium 风险的自二聚体或与原批次引物冲突而被预先淘汰，
其余候选之间的 high/medium 异二聚体构成一张无向冲突图：被同一条边连接
的两个候选不可同时入选。问题即在这张图上求**不超过容量上限 ``limit`` 的
最大独立集（maximum independent set）**。

输入规模固定为 2～18，采用经典的“含/不含分支顶点”分支限界即可精确求解，
无需任何启发式近似。

确定性（相同输入必得相同唯一解）
--------------------------------
* 每一步固定取候选集合中下标**最小**的顶点 v；
* 先递归“含 v”分支，再递归“不含 v”分支；
* 上界 ``当前已选 + 剩余顶点数``（且不超过 ``limit``）严格小于当前最优解
  大小时剪枝；大小相同不剪枝，并在更新时按下标序列字典序比较，
  双重保证唯一解。

含优先 + 最小下标分支保证：首个达到某大小的解就是同大小解中
**按下标序列字典序最小**的——较早的下标在递归树的更浅处被优先固定，
任何与之首处分歧的同大小解只会出现在更靠后的排除子树中。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def maximum_conflict_free_set(
    *,
    candidate_count: int,
    blocked: Iterable[int] = (),
    edges: Sequence[tuple[int, int]],
    limit: int,
) -> list[int]:
    """返回不超过 ``limit`` 的最大无冲突候选下标列表（升序，字典序唯一）。

    ``candidate_count`` 个候选下标为 0..n-1；``blocked`` 中的下标（因自二聚体
    或与原批次冲突而预先淘汰者）不参与求解；``edges`` 为候选间冲突边
    ``(u, v)``（u != v，可乱序、可重复，均按无向边处理）。
    无冲突可入选数不足 ``limit`` 时返回全部可入选顶点。
    """
    blocked_set = set(blocked)
    adjacency: list[set[int]] = [set() for _ in range(candidate_count)]
    for u, v in edges:
        if u == v or u in blocked_set or v in blocked_set:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    available = frozenset(
        i for i in range(candidate_count) if i not in blocked_set
    )
    best: list[int] = []

    def lex_less(a: list[int], b: list[int]) -> bool:
        """等长升序列表的字典序：首个分歧处下标小者优先；前缀关系下短者优先。"""
        for x, y in zip(a, b):
            if x != y:
                return x < y
        return len(a) < len(b)

    def search(chosen: set[int], remaining: frozenset) -> None:
        nonlocal best
        current = sorted(chosen)
        if len(chosen) > len(best) or (
            len(chosen) == len(best) and lex_less(current, best)
        ):
            best = current
        if len(chosen) >= limit:
            return
        if not remaining:
            return
        # 上界（含容量截断）小于当前最优大小则剪枝；等大时不剪枝，
        # 因为同大小仍可能存在字典序更小的解
        if min(len(chosen) + len(remaining), limit) < len(best):
            return

        v = min(remaining)
        rest = remaining - {v}

        # 含 v：删除 v 及其邻居（先搜含分支，配合最小下标分支，
        # 首个最优解即字典序最小解；比较保留仅作确定性双保险）
        search(chosen | {v}, rest - adjacency[v])
        # 不含 v
        search(chosen, rest)

    search(set(), available)
    return best

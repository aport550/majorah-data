import json
import math
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

MIN_CLUSTER_SIZE = 4
MAX_CLUSTER_SIZE = 12
MAX_REPAIR_PASSES = 8


def build_index_maps(tickers):
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    idx_to_ticker = {i: t for i, t in enumerate(tickers)}
    return ticker_to_idx, idx_to_ticker


def average_distance(cluster_a, cluster_b, dist_matrix, ticker_to_idx):
    ia = np.array([ticker_to_idx[t] for t in cluster_a], dtype=int)
    ib = np.array([ticker_to_idx[t] for t in cluster_b], dtype=int)
    vals = dist_matrix[np.ix_(ia, ib)]
    return float(vals.mean())


def intra_cluster_distance(cluster, dist_matrix, ticker_to_idx):
    if len(cluster) <= 1:
        return 0.0
    idx = np.array([ticker_to_idx[t] for t in cluster], dtype=int)
    sub = dist_matrix[np.ix_(idx, idx)]
    n = len(idx)
    total = sub.sum() - np.trace(sub)
    return float(total / (n * (n - 1)))


def labels_to_clusters(labels, tickers):
    clusters = {}
    for tkr, lab in zip(tickers, labels):
        clusters.setdefault(int(lab), []).append(str(tkr))
    return [sorted(v) for _, v in sorted(clusters.items(), key=lambda kv: kv[0])]


def split_oversized_cluster(cluster, dist_matrix, ticker_to_idx, max_size):
    """
    Recursively split an oversized cluster until all pieces are <= max_size.
    """
    out = []
    queue = [list(cluster)]

    while queue:
        current = queue.pop(0)

        if len(current) <= max_size:
            out.append(sorted(current))
            continue

        idx = np.array([ticker_to_idx[t] for t in current], dtype=int)
        sub_dist = dist_matrix[np.ix_(idx, idx)]

        if len(current) == 2:
            out.append([current[0]])
            out.append([current[1]])
            continue

        condensed = squareform(sub_dist, checks=False)
        Z_sub = linkage(condensed, method="average")
        labels = fcluster(Z_sub, t=2, criterion="maxclust")

        left = [t for t, lab in zip(current, labels) if lab == 1]
        right = [t for t, lab in zip(current, labels) if lab == 2]

        if len(left) == 0 or len(right) == 0:
            midpoint = len(current) // 2
            left = current[:midpoint]
            right = current[midpoint:]

        queue.append(left)
        queue.append(right)

    return out


def merge_smallest_cluster_once(clusters, dist_matrix, ticker_to_idx, min_size, max_size):
    """
    Merge one undersized cluster into the nearest feasible cluster.
    Returns (new_clusters, changed_flag).
    """
    small_indices = [i for i, c in enumerate(clusters) if len(c) < min_size]
    if not small_indices:
        return clusters, False

    # Start with the smallest cluster first
    i = min(small_indices, key=lambda k: len(clusters[k]))
    c_small = clusters[i]

    best_j = None
    best_dist = float("inf")

    # Prefer feasible merge that stays <= max_size
    for j, c_other in enumerate(clusters):
        if i == j:
            continue
        if len(c_small) + len(c_other) <= max_size:
            d = average_distance(c_small, c_other, dist_matrix, ticker_to_idx)
            if d < best_dist:
                best_dist = d
                best_j = j

    # If no feasible merge exists, merge with nearest cluster anyway,
    # then split later if needed.
    if best_j is None:
        for j, c_other in enumerate(clusters):
            if i == j:
                continue
            d = average_distance(c_small, c_other, dist_matrix, ticker_to_idx)
            if d < best_dist:
                best_dist = d
                best_j = j

    if best_j is None:
        return clusters, False

    merged = sorted(clusters[i] + clusters[best_j])
    new_clusters = []
    for k, c in enumerate(clusters):
        if k not in (i, best_j):
            new_clusters.append(c)
    new_clusters.append(merged)

    return new_clusters, True


def repair_clusters(clusters, dist_matrix, ticker_to_idx, min_size, max_size):
    """
    Enforce hard bounds 4 <= size <= 12.
    """
    clusters = [sorted(c) for c in clusters]

    for _ in range(MAX_REPAIR_PASSES):
        changed = False

        # Split oversized clusters
        new_clusters = []
        for c in clusters:
            if len(c) > max_size:
                parts = split_oversized_cluster(c, dist_matrix, ticker_to_idx, max_size)
                new_clusters.extend(parts)
                changed = True
            else:
                new_clusters.append(c)
        clusters = new_clusters

        # Merge undersized clusters one at a time
        while True:
            clusters, did_merge = merge_smallest_cluster_once(
                clusters, dist_matrix, ticker_to_idx, min_size, max_size
            )
            if not did_merge:
                break
            changed = True

        # Split again in case a forced merge created oversized clusters
        new_clusters = []
        for c in clusters:
            if len(c) > max_size:
                parts = split_oversized_cluster(c, dist_matrix, ticker_to_idx, max_size)
                new_clusters.extend(parts)
                changed = True
            else:
                new_clusters.append(c)
        clusters = new_clusters

        if not changed:
            break

    bad = [len(c) for c in clusters if not (min_size <= len(c) <= max_size)]
    if bad:
        raise RuntimeError(f"Could not satisfy hard cluster-size rule. Bad sizes: {bad}")

    return sorted(clusters, key=lambda c: (-len(c), c))


def score_solution(clusters, dist_matrix, ticker_to_idx):
    sizes = np.array([len(c) for c in clusters], dtype=float)
    midpoint = (MIN_CLUSTER_SIZE + MAX_CLUSTER_SIZE) / 2.0

    balance_penalty = float(np.sum((sizes - midpoint) ** 2))
    cohesion_penalty = float(
        sum(intra_cluster_distance(c, dist_matrix, ticker_to_idx) for c in clusters)
    )

    score = balance_penalty + cohesion_penalty

    diagnostics = {
        "min_size": int(sizes.min()) if len(sizes) else 0,
        "max_size": int(sizes.max()) if len(sizes) else 0,
        "balance_penalty": balance_penalty,
        "cohesion_penalty": cohesion_penalty,
        "cluster_count": len(clusters),
    }
    return score, diagnostics


def choose_cluster_count_candidates(n_tickers):
    """
    Small candidate set around the midpoint-based target.
    Much faster than sweeping 96 distance thresholds.
    """
    min_k = math.ceil(n_tickers / MAX_CLUSTER_SIZE)
    max_k = math.floor(n_tickers / MIN_CLUSTER_SIZE)

    if min_k > max_k:
        raise RuntimeError(
            f"Infeasible constraints for {n_tickers} tickers with "
            f"min={MIN_CLUSTER_SIZE}, max={MAX_CLUSTER_SIZE}"
        )

    midpoint = (MIN_CLUSTER_SIZE + MAX_CLUSTER_SIZE) / 2.0
    target_k = int(round(n_tickers / midpoint))
    target_k = max(min_k, min(max_k, target_k))

    candidates = sorted(set(
        k for k in [target_k - 2, target_k - 1, target_k, target_k + 1, target_k + 2]
        if min_k <= k <= max_k
    ))

    return candidates


def main():
    corr_path = "data/correlation_matrix.csv"
    out_path = "public/data/clusters.json"

    print(f"CWD: {os.getcwd()}")
    print(f"corr_path: {corr_path} (exists={os.path.exists(corr_path)})")
    print(f"out_path: {out_path}")

    if not os.path.exists(corr_path):
        raise RuntimeError(
            f"Missing {corr_path}. Did daily returns step run and create correlation_matrix.csv?"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    corr = pd.read_csv(corr_path, index_col=0)
    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    corr = corr.apply(pd.to_numeric, errors="coerce")

    idx = corr.index.astype(str)
    cols = corr.columns.astype(str)
    common = [x for x in idx if x in set(cols)]
    if not common:
        raise RuntimeError("Correlation matrix has no overlapping row/column labels.")

    corr = corr.loc[common, common]

    corr_values = np.nan_to_num(corr.values.astype(float), nan=0.0)
    corr_values = np.clip(corr_values, -1.0, 1.0)

    dist_matrix = 1.0 - corr_values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

    tickers = list(map(str, corr.index.tolist()))
    n = len(tickers)

    if n < MIN_CLUSTER_SIZE:
        raise RuntimeError(
            f"Only {n} tickers available, cannot form a valid cluster of size >= {MIN_CLUSTER_SIZE}."
        )

    ticker_to_idx, idx_to_ticker = build_index_maps(tickers)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")

    candidate_ks = choose_cluster_count_candidates(n)
    print(f"Trying candidate cluster counts: {candidate_ks}")

    best = None

    for k in candidate_ks:
        labels = fcluster(Z, t=k, criterion="maxclust")
        raw_clusters = labels_to_clusters(labels, tickers)

        try:
            repaired_clusters = repair_clusters(
                raw_clusters,
                dist_matrix,
                ticker_to_idx,
                MIN_CLUSTER_SIZE,
                MAX_CLUSTER_SIZE,
            )
        except RuntimeError:
            continue

        score, diagnostics = score_solution(repaired_clusters, dist_matrix, ticker_to_idx)

        candidate = {
            "cluster_count_target": k,
            "clusters": repaired_clusters,
            "score": score,
            "diagnostics": diagnostics,
        }

        print(
            f"k={k}: clusters={len(repaired_clusters)} "
            f"min={diagnostics['min_size']} max={diagnostics['max_size']} "
            f"score={score:.4f}"
        )

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No feasible clustering solution found.")

    clusters_sorted = best["clusters"]
    diagnostics = best["diagnostics"]

    payload = {
        "cluster_count_target": best["cluster_count_target"],
        "min_cluster_size_target": MIN_CLUSTER_SIZE,
        "max_cluster_size_target": MAX_CLUSTER_SIZE,
        "ticker_count_used": len(tickers),
        "cluster_count": len(clusters_sorted),
        "solution_quality": {
            "min_size_found": diagnostics["min_size"],
            "max_size_found": diagnostics["max_size"],
            "balance_penalty": diagnostics["balance_penalty"],
            "cohesion_penalty": diagnostics["cohesion_penalty"],
            "cluster_count": diagnostics["cluster_count"],
            "hard_rule_satisfied": True,
        },
        "clusters": [
            {
                "cluster_rank": i + 1,
                "cluster_id": i + 1,
                "size": len(members),
                "members": members,
            }
            for i, members in enumerate(clusters_sorted)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"✅ Wrote {out_path} | "
        f"clusters={len(clusters_sorted)} | "
        f"min={diagnostics['min_size']} max={diagnostics['max_size']}"
    )


if __name__ == "__main__":
    main()

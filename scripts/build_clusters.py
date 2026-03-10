import json
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

MIN_CLUSTER_SIZE = 4
MAX_CLUSTER_SIZE = 12
THRESHOLD_GRID = np.arange(0.05, 1.01, 0.01)


def cluster_distance(cluster_a, cluster_b, dist_df):
    """
    Average pairwise distance between two clusters.
    Lower means more similar.
    """
    vals = dist_df.loc[cluster_a, cluster_b].values
    return float(np.mean(vals))


def internal_cluster_score(cluster, dist_df):
    """
    Average intra-cluster distance.
    Lower is tighter / better.
    """
    if len(cluster) <= 1:
        return 0.0
    sub = dist_df.loc[cluster, cluster].values
    n = len(cluster)
    # exclude diagonal
    total = np.sum(sub) - np.trace(sub)
    denom = n * (n - 1)
    return float(total / denom)


def build_clusters_from_threshold(Z, tickers, threshold):
    labels = fcluster(Z, t=float(threshold), criterion="distance")
    clusters = {}
    for tkr, lab in zip(tickers, labels):
        clusters.setdefault(int(lab), []).append(str(tkr))
    return [sorted(members) for _, members in sorted(clusters.items(), key=lambda kv: kv[0])]


def split_oversized_cluster(cluster, dist_df, max_size):
    """
    Recursively split an oversized cluster until all pieces are <= max_size.
    Uses hierarchical clustering on the subcluster only.
    """
    results = []
    queue = [list(cluster)]

    while queue:
        current = queue.pop(0)

        if len(current) <= max_size:
            results.append(sorted(current))
            continue

        sub_dist = dist_df.loc[current, current].values
        condensed = squareform(sub_dist, checks=False)
        Z_sub = linkage(condensed, method="average")

        # Force a 2-way split first, then recurse if needed
        labels = fcluster(Z_sub, t=2, criterion="maxclust")
        left = [t for t, lab in zip(current, labels) if lab == 1]
        right = [t for t, lab in zip(current, labels) if lab == 2]

        # Safety fallback in pathological cases
        if len(left) == 0 or len(right) == 0:
            midpoint = len(current) // 2
            left = current[:midpoint]
            right = current[midpoint:]

        queue.append(left)
        queue.append(right)

    return results


def merge_small_clusters(clusters, dist_df, min_size, max_size):
    """
    Merge undersized clusters into the nearest cluster that can absorb them
    without exceeding max_size. If none can absorb directly, merge small-small.
    """
    clusters = [list(c) for c in clusters]

    changed = True
    while changed:
        changed = False

        small_idxs = [i for i, c in enumerate(clusters) if len(c) < min_size]
        if not small_idxs:
            break

        for i in small_idxs:
            if i >= len(clusters):
                continue
            if len(clusters[i]) >= min_size:
                continue

            c_small = clusters[i]
            best_j = None
            best_dist = float("inf")

            # First try merging into a cluster that stays within max_size
            for j, c_other in enumerate(clusters):
                if i == j:
                    continue
                if len(c_small) + len(c_other) <= max_size:
                    d = cluster_distance(c_small, c_other, dist_df)
                    if d < best_dist:
                        best_dist = d
                        best_j = j

            # If impossible, allow merge anyway with nearest small cluster,
            # and we will split later if needed
            if best_j is None:
                for j, c_other in enumerate(clusters):
                    if i == j:
                        continue
                    d = cluster_distance(c_small, c_other, dist_df)
                    if d < best_dist:
                        best_dist = d
                        best_j = j

            if best_j is not None:
                merged = sorted(clusters[i] + clusters[best_j])
                keep = []
                for k, c in enumerate(clusters):
                    if k not in (i, best_j):
                        keep.append(c)
                keep.append(merged)
                clusters = keep
                changed = True
                break

    return [sorted(c) for c in clusters]


def repair_clusters(raw_clusters, dist_df, min_size, max_size):
    """
    Enforce hard cluster size constraints:
    1. Split oversized clusters
    2. Merge undersized clusters
    3. Re-split if merges created oversized clusters
    Repeat until stable
    """
    clusters = [sorted(c) for c in raw_clusters]

    for _ in range(20):  # safety cap
        prev_signature = sorted(sorted(c) for c in clusters)

        # Step 1: split oversized clusters
        new_clusters = []
        for c in clusters:
            if len(c) > max_size:
                pieces = split_oversized_cluster(c, dist_df, max_size)
                new_clusters.extend(pieces)
            else:
                new_clusters.append(sorted(c))
        clusters = new_clusters

        # Step 2: merge undersized clusters
        clusters = merge_small_clusters(clusters, dist_df, min_size, max_size)

        # Step 3: if merging caused oversized clusters, split again
        new_clusters = []
        for c in clusters:
            if len(c) > max_size:
                pieces = split_oversized_cluster(c, dist_df, max_size)
                new_clusters.extend(pieces)
            else:
                new_clusters.append(sorted(c))
        clusters = new_clusters

        # Stable?
        curr_signature = sorted(sorted(c) for c in clusters)
        if curr_signature == prev_signature:
            break

    # Final validation
    bad = [len(c) for c in clusters if not (min_size <= len(c) <= max_size)]
    if bad:
        raise RuntimeError(
            f"Unable to satisfy hard size constraints for all clusters. Bad sizes: {bad}. "
            f"This can happen if total ticker count or cluster geometry makes the constraints infeasible."
        )

    return sorted(clusters, key=lambda members: (-len(members), members))


def score_cluster_solution(clusters, dist_df):
    """
    Lower score is better.
    Since repaired clusters already satisfy hard bounds, we just optimize cohesion + balance.
    """
    sizes = np.array([len(c) for c in clusters], dtype=float)
    midpoint = (MIN_CLUSTER_SIZE + MAX_CLUSTER_SIZE) / 2.0

    balance_penalty = float(np.sum((sizes - midpoint) ** 2))
    cohesion_penalty = float(sum(internal_cluster_score(c, dist_df) for c in clusters))
    cluster_count_penalty = float(len(clusters) * 0.01)

    total_score = balance_penalty + cohesion_penalty + cluster_count_penalty

    diagnostics = {
        "min_size": int(np.min(sizes)) if len(sizes) else 0,
        "max_size": int(np.max(sizes)) if len(sizes) else 0,
        "balance_penalty": balance_penalty,
        "cohesion_penalty": cohesion_penalty,
        "cluster_count": int(len(clusters)),
    }
    return total_score, diagnostics


def choose_best_threshold(Z, tickers, dist_df):
    best = None

    for threshold in THRESHOLD_GRID:
        raw_clusters = build_clusters_from_threshold(Z, tickers, threshold)

        try:
            repaired_clusters = repair_clusters(
                raw_clusters=raw_clusters,
                dist_df=dist_df,
                min_size=MIN_CLUSTER_SIZE,
                max_size=MAX_CLUSTER_SIZE,
            )
        except RuntimeError:
            continue

        score, diagnostics = score_cluster_solution(repaired_clusters, dist_df)

        candidate = {
            "threshold": float(round(threshold, 4)),
            "clusters": repaired_clusters,
            "score": score,
            "diagnostics": diagnostics,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError(
            "No feasible clustering found that satisfies the hard rule "
            f"{MIN_CLUSTER_SIZE} <= cluster size <= {MAX_CLUSTER_SIZE}."
        )

    return best


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

    out_dir = os.path.dirname(out_path)
    if not out_dir:
        raise RuntimeError(f"Invalid out_path (no directory part): {out_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"out_dir: {out_dir} (exists={os.path.exists(out_dir)})")

    corr = pd.read_csv(corr_path, index_col=0)
    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    corr = corr.apply(pd.to_numeric, errors="coerce")

    idx = corr.index.astype(str)
    cols = corr.columns.astype(str)
    common = [x for x in idx if x in set(cols)]
    if len(common) == 0:
        raise RuntimeError("Correlation matrix has no overlapping row/column labels.")

    corr = corr.loc[common, common]

    corr_values = np.nan_to_num(corr.values.astype(float), nan=0.0)
    corr_values = np.clip(corr_values, -1.0, 1.0)

    # correlation -> distance
    dist = 1.0 - corr_values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    dist_df = pd.DataFrame(dist, index=common, columns=common)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    tickers = list(map(str, corr.index.tolist()))

    # Feasibility check
    if len(tickers) < MIN_CLUSTER_SIZE:
        raise RuntimeError(
            f"Only {len(tickers)} tickers available, cannot form even one cluster "
            f"of minimum size {MIN_CLUSTER_SIZE}."
        )

    best = choose_best_threshold(Z, tickers, dist_df)

    clusters_sorted = best["clusters"]
    chosen_threshold = best["threshold"]
    diagnostics = best["diagnostics"]

    # final hard validation
    bad_sizes = [len(c) for c in clusters_sorted if not (MIN_CLUSTER_SIZE <= len(c) <= MAX_CLUSTER_SIZE)]
    if bad_sizes:
        raise RuntimeError(f"Hard rule violated after optimization. Bad sizes: {bad_sizes}")

    payload = {
        "distance_threshold": chosen_threshold,
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

    print(f"Chosen threshold: {chosen_threshold}")
    print(
        f"Cluster size range: min={diagnostics['min_size']} "
        f"max={diagnostics['max_size']} "
        f"cluster_count={diagnostics['cluster_count']}"
    )
    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print("public/:", os.path.exists("public"), "public/data/:", os.path.exists("public/data"))


if __name__ == "__main__":
    main()

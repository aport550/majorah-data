import json
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

MIN_CLUSTER_SIZE = 4
MAX_CLUSTER_SIZE = 12

# Search a range of thresholds instead of hardcoding one
THRESHOLD_GRID = np.arange(0.20, 1.51, 0.01)


def build_clusters_from_threshold(Z, tickers, threshold):
    labels = fcluster(Z, t=float(threshold), criterion="distance")

    clusters = {}
    for tkr, lab in zip(tickers, labels):
        clusters.setdefault(int(lab), []).append(tkr)

    clusters_sorted = sorted(
        [(cid, sorted(members)) for cid, members in clusters.items()],
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    return clusters_sorted


def score_cluster_solution(clusters_sorted):
    """
    Lower score is better.

    We prefer:
    - all clusters within [MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE]
    - cluster sizes close to the midpoint of the allowed range
    - fewer violations if no exact solution exists
    """
    sizes = np.array([len(members) for _, members in clusters_sorted], dtype=float)
    if len(sizes) == 0:
        return float("inf"), {
            "min_size": 0,
            "max_size": 0,
            "violation_count": 999999,
            "balance_penalty": float("inf"),
        }

    midpoint = (MIN_CLUSTER_SIZE + MAX_CLUSTER_SIZE) / 2.0

    below = np.maximum(0, MIN_CLUSTER_SIZE - sizes)
    above = np.maximum(0, sizes - MAX_CLUSTER_SIZE)

    violation_count = int(np.sum((sizes < MIN_CLUSTER_SIZE) | (sizes > MAX_CLUSTER_SIZE)))
    violation_magnitude = float(np.sum(below + above))
    balance_penalty = float(np.sum((sizes - midpoint) ** 2))

    # Heavily prioritize satisfying the hard bounds
    total_score = (
        violation_count * 1_000_000
        + violation_magnitude * 100_000
        + balance_penalty
    )

    diagnostics = {
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
        "violation_count": violation_count,
        "violation_magnitude": violation_magnitude,
        "balance_penalty": balance_penalty,
    }
    return total_score, diagnostics


def choose_best_threshold(Z, tickers):
    best = None

    for threshold in THRESHOLD_GRID:
        clusters_sorted = build_clusters_from_threshold(Z, tickers, threshold)
        score, diagnostics = score_cluster_solution(clusters_sorted)

        candidate = {
            "threshold": float(round(threshold, 4)),
            "clusters_sorted": clusters_sorted,
            "score": score,
            "diagnostics": diagnostics,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

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

    # correlation -> distance
    dist = 1.0 - corr_values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    tickers = list(map(str, corr.index.tolist()))
    best = choose_best_threshold(Z, tickers)

    clusters_sorted = best["clusters_sorted"]
    chosen_threshold = best["threshold"]
    diagnostics = best["diagnostics"]

    payload = {
        "distance_threshold": chosen_threshold,
        "min_cluster_size_target": MIN_CLUSTER_SIZE,
        "max_cluster_size_target": MAX_CLUSTER_SIZE,
        "ticker_count_used": len(tickers),
        "cluster_count": len(clusters_sorted),
        "solution_quality": {
            "min_size_found": diagnostics["min_size"],
            "max_size_found": diagnostics["max_size"],
            "violation_count": diagnostics["violation_count"],
            "violation_magnitude": diagnostics["violation_magnitude"],
            "balance_penalty": diagnostics["balance_penalty"],
        },
        "clusters": [
            {
                "cluster_rank": i + 1,
                "cluster_id": cid,
                "size": len(members),
                "members": members,
            }
            for i, (cid, members) in enumerate(clusters_sorted)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Chosen threshold: {chosen_threshold}")
    print(
        f"Cluster size range: min={diagnostics['min_size']} max={diagnostics['max_size']} "
        f"violations={diagnostics['violation_count']}"
    )
    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print("public/:", os.path.exists("public"), "public/data/:", os.path.exists("public/data"))


if __name__ == "__main__":
    main()

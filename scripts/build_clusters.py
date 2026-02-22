import json
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

DIST_THRESHOLD = 0.65  # tweak later


def main():
    corr_path = "data/correlation_matrix.csv"
    out_path = "public/data/clusters.json"

    # --- sanity/debug (helps in GitHub Actions logs) ---
    print(f"CWD: {os.getcwd()}")
    print(f"corr_path: {corr_path} (exists={os.path.exists(corr_path)})")
    print(f"out_path: {out_path}")

    if not os.path.exists(corr_path):
        raise RuntimeError(
            f"Missing {corr_path}. Did daily returns step run and create correlation_matrix.csv?"
        )

    # --- ensure output folder exists (public/data) ---
    out_dir = os.path.dirname(out_path)  # "public/data"
    if not out_dir:
        raise RuntimeError(f"Invalid out_path (no directory part): {out_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"out_dir: {out_dir} (exists={os.path.exists(out_dir)})")

    # --- read correlation matrix ---
    corr = pd.read_csv(corr_path, index_col=0)
    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    # ensure numeric + square
    corr = corr.apply(pd.to_numeric, errors="coerce")

    # keep only intersection of rows/cols to enforce squareness
    idx = corr.index.astype(str)
    cols = corr.columns.astype(str)
    common = [x for x in idx if x in set(cols)]
    if len(common) == 0:
        raise RuntimeError("Correlation matrix has no overlapping row/column labels.")

    corr = corr.loc[common, common]

    corr_values = np.nan_to_num(corr.values.astype(float), nan=0.0)

    # corr -> distance
    dist = 1.0 - corr_values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=DIST_THRESHOLD, criterion="distance")

    clusters = {}
    tickers = list(map(str, corr.index.tolist()))
    for tkr, lab in zip(tickers, labels):
        clusters.setdefault(int(lab), []).append(tkr)

    clusters_sorted = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    payload = {
        "distance_threshold": DIST_THRESHOLD,
        "ticker_count_used": len(tickers),
        "cluster_count": len(clusters_sorted),
        "clusters": [
            {
                "cluster_rank": i + 1,
                "cluster_id": cid,
                "size": len(members),
                "members": sorted(members),
            }
            for i, (cid, members) in enumerate(clusters_sorted)
        ],
    }

    # --- write output ---
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    # helpful extra confirmation
    print("public/:", os.path.exists("public"), "public/data/:", os.path.exists("public/data"))


if __name__ == "__main__":
    main()

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

    if not os.path.exists(corr_path):
        raise RuntimeError(f"Missing {corr_path}. Did daily returns step run and create correlation_matrix.csv?")

    os.makedirs("data", exist_ok=True)

    corr = pd.read_csv(corr_path, index_col=0)

    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    # ensure numeric + square
    corr = corr.apply(pd.to_numeric, errors="coerce")
    corr = corr.loc[corr.index, corr.index]
    corr_values = np.nan_to_num(corr.values.astype(float), nan=0.0)

    # corr -> distance
    dist = 1.0 - corr_values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=DIST_THRESHOLD, criterion="distance")

    clusters = {}
    tickers = corr.index.astype(str).tolist()
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()

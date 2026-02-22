import json
import numpy as np
import pandas as pd

# pip: scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ---- Settings you can tweak ----
# Distance = 1 - corr
# Clustering threshold: smaller => more clusters; larger => fewer clusters.
# Typical good starting range: 0.55 to 0.75
DIST_THRESHOLD = 0.65

# Only keep tickers that have enough data (optional)
MIN_PAIRWISE_NON_NAN = 50


def main():
    corr = pd.read_csv("data/correlation_matrix.csv", index_col=0)

    # Ensure symmetry + consistent ordering
    corr = corr.loc[corr.index, corr.index]

    # Replace NaNs with 0 correlation (neutral)
    # (Alternative: drop tickers with too many NaNs)
    corr_values = corr.values.astype(float)

    # Optional filter: drop tickers with too little correlation info
    # (counts non-nan in each row)
    non_nan_counts = np.sum(~np.isnan(corr_values), axis=1)
    keep_mask = non_nan_counts >= MIN_PAIRWISE_NON_NAN

    kept = corr.index[keep_mask].tolist()
    dropped = corr.index[~keep_mask].tolist()

    corr = corr.loc[kept, kept]
    corr_values = corr.values.astype(float)
    corr_values = np.nan_to_num(corr_values, nan=0.0)

    # Correlation -> distance
    dist = 1.0 - corr_values

    # Force distance matrix validity
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    # Convert to condensed form for scipy
    condensed = squareform(dist, checks=False)

    # Hierarchical clustering (Ward expects Euclidean; average is safer for correlation-distance)
    Z = linkage(condensed, method="average")

    # Cut the dendrogram into clusters based on threshold
    labels = fcluster(Z, t=DIST_THRESHOLD, criterion="distance")

    # Build clusters dict: cluster_id -> [tickers]
    clusters = {}
    for tkr, lab in zip(kept, labels):
        clusters.setdefault(int(lab), []).append(tkr)

    # Sort clusters by size (largest first) and give them stable names
    clusters_sorted = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    cluster_list = []
    for rank, (cid, members) in enumerate(clusters_sorted, start=1):
        members_sorted = sorted(members)
        cluster_list.append({
            "cluster_rank": rank,
            "cluster_id": cid,
            "size": len(members_sorted),
            "members": members_sorted
        })

    payload = {
        "distance_threshold": DIST_THRESHOLD,
        "min_pairwise_non_nan": MIN_PAIRWISE_NON_NAN,
        "ticker_count_used": len(kept),
        "ticker_count_dropped": len(dropped),
        "dropped_tickers_sample": dropped[:50],
        "clusters": cluster_list,
    }

    with open("data/clusters.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote data/clusters.json with {len(cluster_list)} clusters.")


if __name__ == "__main__":
    main()

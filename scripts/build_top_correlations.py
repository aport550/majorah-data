import json
import os
from datetime import date

import numpy as np
import pandas as pd

TOP_K = 10

def main():
    corr_path = "data/correlation_matrix.csv"
    out_path = "public/data/top_correlations.json"

    if not os.path.exists(corr_path):
        raise RuntimeError(f"Missing {corr_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    corr = pd.read_csv(corr_path, index_col=0)
    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    # ensure numeric + square on common labels
    corr = corr.apply(pd.to_numeric, errors="coerce")
    idx = corr.index.astype(str)
    cols = corr.columns.astype(str)
    common = [x for x in idx if x in set(cols)]
    if not common:
        raise RuntimeError("No overlapping row/column labels in correlation matrix.")
    corr = corr.loc[common, common]

    items = {}
    tickers = list(map(str, corr.index.tolist()))

    for t in tickers:
        s = corr.loc[t].copy()
        s = s.drop(labels=[t], errors="ignore")  # remove self
        s = s.replace([np.inf, -np.inf], np.nan).dropna()

        # sort highest corr first, take top K
        top = s.sort_values(ascending=False).head(TOP_K)

        items[t] = [
            {"ticker": str(other), "corr": float(val)}
            for other, val in top.items()
        ]

    payload = {
        "as_of": date.today().isoformat(),
        "top_k": TOP_K,
        "items": items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()

import json
import os
from datetime import date

import numpy as np
import pandas as pd

TOP_K = 20

# 🔥 Your custom exclusion list
EXCLUDE = {
    "VOO","VUG","VOOV","VOOG","VTI","VT",
    "SH","PSQ","DOG","SDOW","SQQQ","TZA","SPXS",
    "TSLZ","PLTZ","NVDZ","NVDQ","NVDW","NVDY","ULTY",
    "SOXS","SOXL","IONZ","PLTD","METD","AAPD",
    "VB","SPYI","QQQI","JEPQ","DGRO","TQQQ",
    "SCHG","NANC","SVXY","VXX","UVIX","UVXY",
    "XAIX","ETHD","YMAX"
}

# Explicitly KEEP these even if future edits touch EXCLUDE
FORCE_INCLUDE = {"SPY", "QQQ", "DIA", "IWM"}


def main():
    corr_path = "data/correlation_matrix.csv"
    out_path = "public/data/top_correlations.json"

    if not os.path.exists(corr_path):
        raise RuntimeError(f"Missing {corr_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    corr = pd.read_csv(corr_path, index_col=0)
    if corr.empty:
        raise RuntimeError("Correlation matrix is empty.")

    # ensure numeric + square
    corr = corr.apply(pd.to_numeric, errors="coerce")

    idx = corr.index.astype(str)
    cols = corr.columns.astype(str)

    common = [x for x in idx if x in set(cols)]
    if not common:
        raise RuntimeError("No overlapping row/column labels in correlation matrix.")

    corr = corr.loc[common, common]

    tickers = list(map(str, corr.index.tolist()))

    items = {}

    for t in tickers:
        s = corr.loc[t].copy()

        # remove self
        s = s.drop(labels=[t], errors="ignore")

        # clean invalid values
        s = s.replace([np.inf, -np.inf], np.nan).dropna()

        # 🔥 apply exclusion (but never exclude FORCE_INCLUDE)
        to_exclude = [
            x for x in s.index
            if x in EXCLUDE and x not in FORCE_INCLUDE
        ]
        if to_exclude:
            s = s.drop(labels=to_exclude, errors="ignore")

        if s.empty:
            items[t] = {"positive": [], "negative": []}
            continue

        # strongest positive
        pos = s.sort_values(ascending=False).head(TOP_K)

        # most inverse (most negative)
        neg = s.sort_values(ascending=True).head(TOP_K)

        items[t] = {
            "positive": [
                {"ticker": str(o), "corr": float(v)}
                for o, v in pos.items()
            ],
            "negative": [
                {"ticker": str(o), "corr": float(v)}
                for o, v in neg.items()
            ],
        }

    payload = {
        "as_of": date.today().isoformat(),
        "top_k": TOP_K,
        "excluded_count": len(EXCLUDE),
        "items": items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()

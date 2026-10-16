"""
Reusable WOE (Weight of Evidence) binning utility.

Why a class instead of one-off code: WOE bin edges must be learned ONLY from
training data, then applied unchanged to the test set (and to any future
applicant). If we re-computed bins on the test set, we'd be leaking test-set
information into how we group values - a classic beginner mistake.
"""

import pandas as pd
import numpy as np


class WOEBinner:
    def __init__(self):
        self.bin_edges_ = {}   # feature -> bin edges (for numeric qcut features)
        self.woe_maps_ = {}    # feature -> {bin_label: woe_value}
        self.iv_ = {}          # feature -> IV
        self.discrete_features_ = set()

    def _make_bins(self, series):
        top_share = series.value_counts(normalize=True).iloc[0]
        if series.nunique() <= 20 or top_share > 0.4:
            cap = series[series > 0].quantile(0.99) if (series > 0).any() else 0
            cap = max(int(cap), 3)
            return None, cap  # discrete: no qcut edges, just a cap value
        _, edges = pd.qcut(series, q=10, duplicates="drop", retbins=True)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges, None

    def _apply_bins(self, series, edges, cap):
        if edges is None:
            capped = series.clip(upper=cap)
            return capped.astype(str).where(capped < cap, f"{cap}+")
        return pd.cut(series, bins=edges)

    def fit(self, df, features, target):
        total_good = (df[target] == 0).sum()
        total_bad = (df[target] == 1).sum()

        for feat in features:
            edges, cap = self._make_bins(df[feat])
            self.bin_edges_[feat] = (edges, cap)
            binned = self._apply_bins(df[feat], edges, cap)

            grouped = pd.DataFrame({"bin": binned, "target": df[target]})
            summary = grouped.groupby("bin", observed=True)["target"].agg(
                ["count", "sum"]
            )
            summary.columns = ["total", "bad"]
            summary["good"] = summary["total"] - summary["bad"]
            summary["pct_good"] = (summary["good"] + 0.5) / (total_good + 0.5)
            summary["pct_bad"] = (summary["bad"] + 0.5) / (total_bad + 0.5)
            summary["woe"] = np.log(summary["pct_good"] / summary["pct_bad"])
            summary["iv_contribution"] = (
                summary["pct_good"] - summary["pct_bad"]
            ) * summary["woe"]

            self.woe_maps_[feat] = summary["woe"].to_dict()
            self.iv_[feat] = summary["iv_contribution"].sum()
        return self

    def transform(self, df, features):
        out = pd.DataFrame(index=df.index)
        for feat in features:
            edges, cap = self.bin_edges_[feat]
            binned = self._apply_bins(df[feat], edges, cap)
            woe_map = self.woe_maps_[feat]
            # unseen bins (rare, at the edges) fall back to WOE = 0 (neutral)
            out[f"{feat}_woe"] = binned.map(woe_map).astype(float).fillna(0.0)
        return out

    def iv_table(self):
        return (
            pd.Series(self.iv_, name="IV")
            .sort_values(ascending=False)
            .to_frame()
        )

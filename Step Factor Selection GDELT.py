#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Factor Selection GDELT.py
=============================================================================

DESCRIPTION:
    Scores every GDELT factor portfolio on three pillars to produce a
    keep/drop recommendation for downstream pipeline steps.
      1. Long-term alpha: cumulative return, annualised Sharpe, hit rate,
         max drawdown, t-stat.
      2. Episodic alpha: peak trailing Sharpe, Sharpe volatility, fraction
         of months with |Sharpe| > 1, best-horizon indicator, AR(1)
         auto-predictability, and information coefficient — computed across
         6 rolling windows (3, 6, 12, 24, 36, 60 months).
      3. Distinctiveness / redundancy: pairwise return correlation,
         hierarchical clustering, within-cluster ranking.
    Pillar scores are merged into a weighted composite, percentile-ranked,
    and used to flag redundant or low-scoring factors for removal.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Optimizer.xlsx
        Monthly net returns for each GDELT factor portfolio (percentage
        points), read from the Monthly_Net_Returns sheet.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/Step Factor Categories GDELT.xlsx
        Optional file mapping factor names to category labels for output
        tagging only.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Factor_Selection.xlsx
        Multi-sheet workbook: All_Factors_Scored, Kept_Factors,
        Dropped_Factors, Correlation_Clusters.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Factor_Selection_Summary.pdf
        One-page summary figure: scatter (long-term vs episodic score),
        composite-score histogram, top-30 bar chart, best-horizon
        distribution.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - numpy
    - scipy
    - matplotlib
    - seaborn
    - xlsxwriter
    - openpyxl

USAGE:
    python "Step Factor Selection GDELT.py"

NOTES:
    - Composite score = equal weights of long-term, episodic, and
      distinctiveness pillar scores (1/3 each).
    - Redundant factors in correlation clusters (distance < 0.30) are
      dropped unless they are the cluster's top scorer.
    - A minimum composite percentile threshold (default 0) can be
      configured via MIN_COMPOSITE_PCTILE.
=============================================================================
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input / output paths
OPTIMIZER_PATH = "GDELT_Optimizer.xlsx"
FACTOR_CAT_PATH = "Step Factor Categories GDELT.xlsx"
OUTPUT_XLSX = "GDELT_Factor_Selection.xlsx"
OUTPUT_PDF = "GDELT_Factor_Selection_Summary.pdf"

# Pillar weights for composite score (must sum to 1)
W_LONGTERM = 1.0 / 3.0
W_EPISODIC = 1.0 / 3.0
W_DISTINCT = 1.0 / 3.0

# Episodic alpha rolling windows (months)
ROLLING_WINDOWS = [3, 6, 12, 24, 36, 60]

# Redundancy: correlation threshold for "high correlation" flag
CORR_THRESHOLD = 0.85

# Hierarchical clustering: max intra-cluster distance (1 - corr)
CLUSTER_DISTANCE_THRESHOLD = 0.30  # ≈ corr ≥ 0.70 within cluster

# Minimum composite percentile to keep (0-100).  Set to 0 to keep all and
# rely only on redundancy filtering.
MIN_COMPOSITE_PCTILE = 0

# =============================================================================
# PILLAR 1 — LONG-TERM ALPHA
# =============================================================================

def compute_longterm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute persistent-signal quality metrics for each factor column.

    Parameters
    ----------
    df : DataFrame
        Monthly net returns (percentage points), index = dates.

    Returns
    -------
    DataFrame indexed by factor name with columns:
        cum_ret, ann_mean, ann_vol, ann_sharpe, hit_rate, max_dd, tstat
    """
    results = {}
    n = len(df)
    sqrt12 = np.sqrt(12)

    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 6:
            continue

        m = s.mean()
        sd = s.std(ddof=1)
        cum = s.sum()
        ann_mean = m * 12
        ann_vol = sd * sqrt12
        sharpe = (m / sd * sqrt12) if sd > 0 else 0.0
        # Hit rate: fraction in the direction of cumulative alpha
        if cum >= 0:
            hit = (s > 0).mean()
        else:
            hit = (s < 0).mean()
        # t-stat for mean ≠ 0
        tstat = m / (sd / np.sqrt(len(s))) if sd > 0 else 0.0
        # Max drawdown on cumulative return path
        cum_path = s.cumsum()
        running_max = cum_path.cummax()
        dd = cum_path - running_max
        max_dd = dd.min()

        results[col] = {
            "cum_ret": cum,
            "ann_mean": ann_mean,
            "ann_vol": ann_vol,
            "ann_sharpe": sharpe,
            "hit_rate": hit,
            "max_dd": max_dd,
            "tstat": tstat,
        }

    out = pd.DataFrame(results).T
    out.index.name = "factor"
    return out


def _score_longterm(lt: pd.DataFrame) -> pd.Series:
    """
    Normalised 0-1 composite score for Pillar 1.
    Uses |ann_sharpe|, |tstat|, and deviation of hit_rate from 50%.
    """
    abs_sharpe_rank = lt["ann_sharpe"].abs().rank(pct=True)
    abs_tstat_rank = lt["tstat"].abs().rank(pct=True)
    hit_dev_rank = (lt["hit_rate"] - 0.50).abs().rank(pct=True)
    return abs_sharpe_rank * 0.50 + abs_tstat_rank * 0.30 + hit_dev_rank * 0.20


# =============================================================================
# PILLAR 2 — EPISODIC ALPHA
# =============================================================================

def compute_episodic_metrics(
    df: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Compute trailing-window Sharpe-based episodic metrics.

    For each factor, for each window w:
      - rolling_sharpe_wM  = rolling(w).mean / rolling(w).std * sqrt(12)
      - peak_sharpe_wM     = max |rolling_sharpe_wM|
      - sharpe_vol_wM      = std(rolling_sharpe_wM)
      - frac_gt1_wM        = fraction of months where |rolling Sharpe| > 1

    Then across windows:
      - best_horizon        = w with highest peak_sharpe_wM
      - best_peak_sharpe    = peak_sharpe at best horizon
      - ar1_best            = AR(1) coefficient of rolling Sharpe at best horizon

    Returns
    -------
    DataFrame indexed by factor name.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    # Pre-compute rolling Sharpe for all factors × windows
    # Annualize with √(12/w) so each window's Sharpe is comparable on an
    # annual basis.  Cap at ±5 to prevent degenerate extreme values from
    # very sparse factors dominating percentile ranks.
    SHARPE_CAP = 5.0
    rolling_sharpes: dict[int, pd.DataFrame] = {}
    for w in windows:
        rm = df.rolling(w, min_periods=w).mean()
        rs = df.rolling(w, min_periods=w).std(ddof=1)
        annualisation = np.sqrt(12.0 / w)
        raw = (rm / rs.replace(0, np.nan)) * annualisation
        rolling_sharpes[w] = raw.clip(-SHARPE_CAP, SHARPE_CAP)

    records = {}
    for col in df.columns:
        rec: dict = {}
        best_ic = -1.0
        best_w = windows[0]
        best_sharpe_series: pd.Series | None = None

        factor_rets = df[col].dropna()

        for w in windows:
            rs = rolling_sharpes[w][col].dropna()
            if len(rs) < 2:
                rec[f"peak_sharpe_{w}M"] = np.nan
                rec[f"sharpe_vol_{w}M"] = np.nan
                rec[f"frac_gt1_{w}M"] = np.nan
                rec[f"ic_{w}M"] = np.nan
                continue

            peak = rs.abs().max()
            vol = rs.std(ddof=1)
            frac = (rs.abs() > 1.0).mean()

            rec[f"peak_sharpe_{w}M"] = peak
            rec[f"sharpe_vol_{w}M"] = vol
            rec[f"frac_gt1_{w}M"] = frac

            # Information Coefficient: correlation of trailing Sharpe with
            # next month's factor return.  This is the key metric — does
            # looking at the last w months of factor performance predict
            # next month's factor return?
            # Shift rolling Sharpe forward by 1 to align with next return
            rs_shifted = rs.shift(1)
            common_idx = rs_shifted.dropna().index.intersection(factor_rets.index)
            if len(common_idx) > 10:
                ic = np.corrcoef(
                    rs_shifted.loc[common_idx].values,
                    factor_rets.loc[common_idx].values,
                )[0, 1]
                ic = ic if np.isfinite(ic) else 0.0
            else:
                ic = 0.0
            rec[f"ic_{w}M"] = ic

            # Best horizon = window with highest |IC| (predictive power).
            # Tie-break: prefer longer window (more robust).
            abs_ic = abs(ic)
            if abs_ic > best_ic or (abs_ic == best_ic and w > best_w):
                best_ic = abs_ic
                best_w = w
                best_sharpe_series = rs

        rec["best_horizon"] = best_w
        rec["best_ic"] = best_ic
        # Best peak sharpe is taken at the best-horizon window
        rec["best_peak_sharpe"] = rec.get(f"peak_sharpe_{best_w}M", np.nan)
        # Best fraction > 1 across all windows (for scoring)
        all_fracs = [rec.get(f"frac_gt1_{w}M", 0) for w in windows]
        valid_fracs = [f for f in all_fracs if not np.isnan(f)]
        rec["best_frac_gt1"] = max(valid_fracs) if valid_fracs else 0.0

        # AR(1) of rolling Sharpe at best horizon
        if best_sharpe_series is not None and len(best_sharpe_series) > 3:
            y = best_sharpe_series.values
            ar1_corr = np.corrcoef(y[:-1], y[1:])[0, 1]
            rec["ar1_best"] = ar1_corr if np.isfinite(ar1_corr) else 0.0
        else:
            rec["ar1_best"] = 0.0

        records[col] = rec

    out = pd.DataFrame(records).T
    out.index.name = "factor"
    return out


def _score_episodic(ep: pd.DataFrame) -> pd.Series:
    """
    Normalised 0-1 composite score for Pillar 2.
    best peak Sharpe (at best horizon) × 0.3
    + best IC (predictive power) × 0.3
    + best fraction > 1 (across all windows) × 0.2
    + AR(1) at best horizon (persistence) × 0.2
    """
    peak_rank = ep["best_peak_sharpe"].rank(pct=True)
    ic_rank = ep["best_ic"].rank(pct=True)
    frac_rank = ep["best_frac_gt1"].rank(pct=True)
    ar1_rank = ep["ar1_best"].abs().rank(pct=True)

    return peak_rank * 0.30 + ic_rank * 0.30 + frac_rank * 0.20 + ar1_rank * 0.20


# =============================================================================
# PILLAR 3 — DISTINCTIVENESS / REDUNDANCY
# =============================================================================

def compute_redundancy_metrics(
    df: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
    cluster_dist: float = CLUSTER_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute pairwise return correlation, hierarchical clustering,
    and within-cluster ranking.

    Returns
    -------
    DataFrame indexed by factor name with columns:
        max_pairwise_corr, mean_pairwise_corr, cluster_id, cluster_size
    """
    print("  Computing pairwise correlation matrix (this may take a moment)...")
    corr = df.corr()

    # Fill any NaN correlations with 0 (factors with no overlap)
    corr = corr.fillna(0.0)

    max_corr = corr.abs().apply(
        lambda row: row.drop(row.name).max() if row.name in row.index else 0.0, axis=1
    )
    mean_corr = corr.abs().apply(
        lambda row: row.drop(row.name).mean() if row.name in row.index else 0.0, axis=1
    )

    # Hierarchical clustering on 1 - |corr|
    print("  Running hierarchical clustering...")
    dist_matrix = 1.0 - corr.abs().values
    np.fill_diagonal(dist_matrix, 0.0)
    # Make symmetric (numerical precision)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix = np.clip(dist_matrix, 0, None)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=cluster_dist, criterion="distance")

    out = pd.DataFrame(
        {
            "max_pairwise_corr": max_corr,
            "mean_pairwise_corr": mean_corr,
            "cluster_id": labels,
        },
        index=df.columns,
    )
    out.index.name = "factor"

    # Cluster sizes
    cluster_sizes = out["cluster_id"].value_counts().to_dict()
    out["cluster_size"] = out["cluster_id"].map(cluster_sizes)

    return out


def _score_distinctiveness(rd: pd.DataFrame) -> pd.Series:
    """
    Score: lower max_pairwise_corr → higher score (more distinct).
    We invert the rank so that unique factors score higher.
    """
    return (1.0 - rd["max_pairwise_corr"].rank(pct=True))


# =============================================================================
# COMPOSITE SCORING
# =============================================================================

def composite_score(
    lt: pd.DataFrame,
    ep: pd.DataFrame,
    rd: pd.DataFrame,
    w_lt: float = W_LONGTERM,
    w_ep: float = W_EPISODIC,
    w_rd: float = W_DISTINCT,
) -> pd.DataFrame:
    """
    Merge pillar scores, compute weighted composite, rank,
    and assign keep/drop labels.
    """
    # Pillar scores (0-1 percentile rank)
    s_lt = _score_longterm(lt).rename("score_longterm")
    s_ep = _score_episodic(ep).rename("score_episodic")
    s_rd = _score_distinctiveness(rd).rename("score_distinct")

    # Merge all metrics + scores
    scored = lt.join(ep, how="outer").join(rd, how="outer")
    scored = scored.join(s_lt).join(s_ep).join(s_rd)

    # Composite
    scored["composite_score"] = (
        w_lt * scored["score_longterm"].fillna(0)
        + w_ep * scored["score_episodic"].fillna(0)
        + w_rd * scored["score_distinct"].fillna(0)
    )
    scored["composite_rank"] = scored["composite_score"].rank(ascending=False).astype(int)

    # Within-cluster ranking: best composite in each cluster
    scored["cluster_rank"] = scored.groupby("cluster_id")["composite_score"].rank(
        ascending=False
    )
    scored["is_cluster_best"] = scored["cluster_rank"] == 1.0

    # Keep / drop label
    pctile = scored["composite_score"].rank(pct=True) * 100
    scored["keep"] = True

    # Drop if below minimum percentile
    if MIN_COMPOSITE_PCTILE > 0:
        scored.loc[pctile < MIN_COMPOSITE_PCTILE, "keep"] = False

    # Drop redundant: in cluster with size > 1 and not the best
    redundant_mask = (scored["cluster_size"] > 1) & (~scored["is_cluster_best"])
    scored.loc[redundant_mask, "keep"] = False

    # Assign a drop reason
    scored["drop_reason"] = ""
    if MIN_COMPOSITE_PCTILE > 0:
        scored.loc[pctile < MIN_COMPOSITE_PCTILE, "drop_reason"] = "low_composite"
    scored.loc[redundant_mask, "drop_reason"] = scored.loc[
        redundant_mask, "drop_reason"
    ].apply(lambda x: (x + "; " if x else "") + "redundant_in_cluster")

    scored = scored.sort_values("composite_rank")
    return scored


# =============================================================================
# OUTPUT — EXCEL
# =============================================================================

def write_output(
    scored: pd.DataFrame,
    factor_categories: pd.DataFrame | None,
    output_path: str,
) -> None:
    """Write multi-sheet Excel workbook."""
    # Attach category labels if available
    if factor_categories is not None and "Category" in factor_categories.columns:
        cat_map = dict(
            zip(factor_categories["Factor Name"], factor_categories["Category"])
        )
        scored.insert(0, "category", scored.index.map(cat_map).fillna("Unknown"))

    kept = scored[scored["keep"]].copy()
    dropped = scored[~scored["keep"]].copy()

    # Cluster summary
    cluster_summary = (
        scored.groupby("cluster_id")
        .agg(
            size=("composite_score", "size"),
            best_factor=("composite_rank", "idxmin"),
            best_score=("composite_score", "max"),
            mean_score=("composite_score", "mean"),
            n_kept=("keep", "sum"),
        )
        .sort_values("best_score", ascending=False)
    )

    # Reorder columns for readability: put key summary columns first
    summary_cols_first = [
        "category",
        "composite_score",
        "composite_rank",
        "keep",
        "drop_reason",
        "score_longterm",
        "score_episodic",
        "score_distinct",
        "cum_ret",
        "ann_sharpe",
        "hit_rate",
        "tstat",
        "max_dd",
        "best_peak_sharpe",
        "best_horizon",
        "ar1_best",
        "max_pairwise_corr",
        "cluster_id",
        "cluster_size",
        "is_cluster_best",
    ]
    existing_first = [c for c in summary_cols_first if c in scored.columns]
    remaining = [c for c in scored.columns if c not in existing_first]
    col_order = existing_first + remaining

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        scored[col_order].to_excel(
            writer, sheet_name="All_Factors_Scored", index_label="Factor"
        )
        kept[col_order].to_excel(
            writer, sheet_name="Kept_Factors", index_label="Factor"
        )
        dropped[col_order].to_excel(
            writer, sheet_name="Dropped_Factors", index_label="Factor"
        )
        cluster_summary.to_excel(
            writer, sheet_name="Correlation_Clusters", index_label="Cluster_ID"
        )

        # Format numbers
        wb = writer.book
        num_fmt = wb.add_format({"num_format": "0.0000"})
        pct_fmt = wb.add_format({"num_format": "0.00%"})
        int_fmt = wb.add_format({"num_format": "0"})

        for sheet_name in [
            "All_Factors_Scored",
            "Kept_Factors",
            "Dropped_Factors",
        ]:
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 0, 45)  # Factor name column
            ws.set_column(1, len(col_order), 14, num_fmt)
            # Freeze top row and factor column
            ws.freeze_panes(1, 1)

    print(f"\n{'='*60}")
    print(f"Output written to {output_path}")
    print(f"  Total factors scored : {len(scored)}")
    print(f"  Kept                 : {len(kept)}")
    print(f"  Dropped              : {len(dropped)}")
    print(f"  Clusters             : {scored['cluster_id'].nunique()}")
    print(f"{'='*60}")


# =============================================================================
# OUTPUT — PDF SUMMARY
# =============================================================================

def plot_summary(scored: pd.DataFrame, pdf_path: str) -> None:
    """One-page PDF with three panels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GDELT Factor Selection Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    kept = scored["keep"]

    # --- Panel 1: Scatter — long-term vs episodic score ---
    ax = axes[0, 0]
    ax.scatter(
        scored.loc[~kept, "score_longterm"],
        scored.loc[~kept, "score_episodic"],
        c="#cccccc",
        s=12,
        alpha=0.5,
        label=f"Dropped ({(~kept).sum()})",
        edgecolors="none",
    )
    ax.scatter(
        scored.loc[kept, "score_longterm"],
        scored.loc[kept, "score_episodic"],
        c="#1a73e8",
        s=18,
        alpha=0.7,
        label=f"Kept ({kept.sum()})",
        edgecolors="none",
    )
    ax.set_xlabel("Long-Term Alpha Score", fontsize=11)
    ax.set_ylabel("Episodic Alpha Score", fontsize=11)
    ax.set_title("Long-Term vs Episodic Alpha", fontsize=12, fontweight="600")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)

    # --- Panel 2: Histogram of composite scores ---
    ax = axes[0, 1]
    ax.hist(
        scored.loc[~kept, "composite_score"],
        bins=40,
        color="#cccccc",
        alpha=0.7,
        label="Dropped",
    )
    ax.hist(
        scored.loc[kept, "composite_score"],
        bins=40,
        color="#1a73e8",
        alpha=0.7,
        label="Kept",
    )
    ax.set_xlabel("Composite Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Composite Score Distribution", fontsize=12, fontweight="600")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # --- Panel 3: Top-30 factors bar chart ---
    ax = axes[1, 0]
    top30 = scored.head(30)
    colors = ["#1a73e8" if k else "#e84040" for k in top30["keep"]]
    bars = ax.barh(
        range(len(top30)),
        top30["composite_score"],
        color=colors,
        height=0.7,
    )
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(
        [n[:40] for n in top30.index], fontsize=7, fontfamily="monospace"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Composite Score", fontsize=11)
    ax.set_title("Top 30 Factors", fontsize=12, fontweight="600")
    ax.grid(alpha=0.2, axis="x")

    # --- Panel 4: Best horizon distribution ---
    ax = axes[1, 1]
    if "best_horizon" in scored.columns:
        horizon_counts = scored["best_horizon"].value_counts().sort_index()
        horizon_kept = scored.loc[kept, "best_horizon"].value_counts().sort_index()
        bar_positions = range(len(horizon_counts))
        ax.bar(
            bar_positions,
            horizon_counts.values,
            color="#cccccc",
            label="All",
            width=0.4,
            align="edge",
        )
        kept_vals = [horizon_kept.get(h, 0) for h in horizon_counts.index]
        ax.bar(
            [p + 0.4 for p in bar_positions],
            kept_vals,
            color="#1a73e8",
            label="Kept",
            width=0.4,
            align="edge",
        )
        ax.set_xticks([p + 0.2 for p in bar_positions])
        ax.set_xticklabels([f"{int(h)}M" for h in horizon_counts.index])
        ax.set_xlabel("Best Horizon (months)", fontsize=11)
        ax.set_ylabel("Number of Factors", fontsize=11)
        ax.set_title("Best Horizon Distribution", fontsize=12, fontweight="600")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary PDF saved to {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("GDELT Factor Selection Framework")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading GDELT_Optimizer.xlsx ...")
    df = pd.read_excel(OPTIMIZER_PATH, sheet_name="Monthly_Net_Returns", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(f"  Shape: {df.shape[0]} months × {df.shape[1]} factors")
    print(f"  Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")

    # Load factor categories (optional, for labelling)
    factor_categories = None
    if os.path.isfile(FACTOR_CAT_PATH):
        factor_categories = pd.read_excel(FACTOR_CAT_PATH)
        print(f"  Loaded {len(factor_categories)} factor category labels")

    # ------------------------------------------------------------------
    # Pillar 1: Long-term alpha
    # ------------------------------------------------------------------
    print("\n[2/6] Computing long-term alpha metrics ...")
    lt = compute_longterm_metrics(df)
    print(f"  Scored {len(lt)} factors")
    print(f"  Top 5 by |ann_sharpe|:")
    top5 = lt["ann_sharpe"].abs().nlargest(5)
    for name, val in top5.items():
        direction = "+" if lt.loc[name, "ann_sharpe"] > 0 else "-"
        print(f"    {direction} {name}: Sharpe={lt.loc[name, 'ann_sharpe']:.3f}, "
              f"cum={lt.loc[name, 'cum_ret']:.2f}, hit={lt.loc[name, 'hit_rate']:.1%}")

    # ------------------------------------------------------------------
    # Pillar 2: Episodic alpha
    # ------------------------------------------------------------------
    print(f"\n[3/6] Computing episodic alpha metrics (windows: {ROLLING_WINDOWS}) ...")
    ep = compute_episodic_metrics(df, windows=ROLLING_WINDOWS)
    print(f"  Scored {len(ep)} factors")
    print(f"  Top 5 by best_peak_sharpe:")
    top5_ep = ep["best_peak_sharpe"].nlargest(5)
    for name, val in top5_ep.items():
        w = int(ep.loc[name, "best_horizon"])
        ar1 = ep.loc[name, "ar1_best"]
        print(f"    {name}: peak|Sharpe|={val:.2f} @ {w}M, AR(1)={ar1:.3f}")

    # ------------------------------------------------------------------
    # Pillar 3: Redundancy / distinctiveness
    # ------------------------------------------------------------------
    print(f"\n[4/6] Computing redundancy metrics (corr threshold={CORR_THRESHOLD}) ...")
    rd = compute_redundancy_metrics(df, threshold=CORR_THRESHOLD)
    n_clusters = rd["cluster_id"].nunique()
    print(f"  {n_clusters} clusters identified")
    largest_cluster = rd["cluster_id"].value_counts().iloc[0]
    print(f"  Largest cluster size: {largest_cluster}")
    singleton_count = (rd["cluster_size"] == 1).sum()
    print(f"  Singleton factors (unique): {singleton_count}")

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------
    print("\n[5/6] Computing composite scores ...")
    scored = composite_score(lt, ep, rd)
    n_kept = scored["keep"].sum()
    n_dropped = (~scored["keep"]).sum()
    print(f"  Kept: {n_kept} / {len(scored)}  ({n_kept/len(scored):.0%})")
    print(f"  Dropped: {n_dropped}  (redundant: {(scored['drop_reason'].str.contains('redundant')).sum()})")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print("\n[6/6] Writing outputs ...")
    write_output(scored, factor_categories, OUTPUT_XLSX)
    plot_summary(scored, OUTPUT_PDF)

    # Print top 20 for quick review
    print(f"\n{'='*60}")
    print("TOP 20 FACTORS BY COMPOSITE SCORE")
    print(f"{'='*60}")
    cols_show = ["composite_score", "score_longterm", "score_episodic",
                 "score_distinct", "ann_sharpe", "best_peak_sharpe",
                 "best_horizon", "max_pairwise_corr", "keep"]
    print(scored[cols_show].head(20).to_string())
    print(f"\nDone!")


if __name__ == "__main__":
    main()

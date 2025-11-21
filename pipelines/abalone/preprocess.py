#!/usr/bin/env python
"""
Clinical biomarker analysis script for SageMaker Processing.

- Downloads clinical.csv from an S3 path
- Basic descriptive stats
- Missingness & QC
- Change From Baseline (CFB) for biomarkers
- Differential analysis at D2 (DRUG vs PLACEBO) with BH-FDR
- Time-course plots for significant markers → saved to /opt/ml/processing/output/
"""
import argparse
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# Benjamini–Hochberg FDR
# =========================================================
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values (same shape as pvals).
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    q = np.empty(m, dtype=float)
    prev_q = 1.0
    # Iterate from largest p to smallest
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_i = ranked_p[i] * m / rank
        if q_i > prev_q:
            q_i = prev_q
        prev_q = q_i
        q[i] = q_i
    qvals = np.empty(m, dtype=float)
    qvals[order] = q
    return qvals


# =========================================================
# Core analysis steps
# =========================================================
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    marker_cols = [c for c in df.columns if c.startswith("MARKER_")]

    print("=== Basic Info ===")
    print("Shape:", df.shape)
    print("Number of biomarker columns:", len(marker_cols))
    print("Example biomarker columns:", marker_cols[:5])

    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    marker_cols = [c for c in df.columns if c.startswith("MARKER_")]

    print("\n=== Descriptive statistics (full dataset) ===")

    print("\nTREATMENT value counts:")
    print(df["TREATMENT"].value_counts(dropna=False))

    print("\nGENDER value counts:")
    print(df["GENDER"].value_counts(dropna=False))

    if "MARKER_TP53" in df.columns:
        print("\nMARKER_TP53 summary (full dataset):")
        print(df["MARKER_TP53"].describe())
    else:
        print("\nWARNING: MARKER_TP53 not found. Available markers example:")
        print(marker_cols[:10])


def qc_missingness(df: pd.DataFrame) -> pd.DataFrame:
    marker_cols = [c for c in df.columns if c.startswith("MARKER_")]

    # Row-level QC flag: any marker missing => measurement invalid at that timepoint
    df = df.copy()
    df["ROW_HAS_MISSING_MARKER"] = df[marker_cols].isna().any(axis=1)

    print("\n=== Missingness summary ===")
    print("Total rows:", len(df))
    print("Rows with any missing biomarker:", df["ROW_HAS_MISSING_MARKER"].sum())

    print("\nMissingness by TREATMENT:")
    print(
        df.groupby("TREATMENT")["ROW_HAS_MISSING_MARKER"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "rows",
                "sum": "rows_with_missing",
                "mean": "frac_with_missing",
            }
        )
    )

    print("\nMissingness by GENDER:")
    print(
        df.groupby("GENDER")["ROW_HAS_MISSING_MARKER"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "rows",
                "sum": "rows_with_missing",
                "mean": "frac_with_missing",
            }
        )
    )

    # Analysis dataset: drop rows with missing marker
    analysis_df = df[~df["ROW_HAS_MISSING_MARKER"]].copy()

    print("\n=== Analysis dataset size ===")
    print("Rows in analysis dataset:", len(analysis_df))

    print("\nTREATMENT (analysis dataset):")
    print(analysis_df["TREATMENT"].value_counts(dropna=False))

    print("\nGENDER (analysis dataset):")
    print(analysis_df["GENDER"].value_counts(dropna=False))

    if "MARKER_TP53" in analysis_df.columns:
        print("\nMARKER_TP53 (analysis dataset):")
        print(analysis_df["MARKER_TP53"].describe())

    return analysis_df


def compute_cfb(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Change From Baseline (D0) for each biomarker at D1, D2."""
    marker_cols = [c for c in analysis_df.columns if c.startswith("MARKER_")]

    pivot = analysis_df.pivot_table(
        index="USUBJID",
        columns="VISIT",
        values=marker_cols,
    )

    # Collect all new CFB columns in a list for concat
    cfb_frames = []

    for m in marker_cols:
        if "D0" not in pivot[m]:
            continue
        baseline = pivot[m]["D0"]

        for visit in ["D1", "D2"]:
            if visit in pivot[m]:
                cfb_series = pivot[m][visit] - baseline
                cfb_series.name = (m + "_CFB", visit)
                cfb_frames.append(cfb_series)

    if not cfb_frames:
        print("\nWARNING: No CFB columns could be computed (missing D0/D1/D2).")
        sub_meta = analysis_df[["USUBJID", "TREATMENT", "GENDER"]].drop_duplicates()
        return sub_meta

    # Concatenate all at once → avoids fragmentation
    cfb_wide = pd.concat(cfb_frames, axis=1)

    # Flatten multiindex columns
    cfb_wide.columns = [f"{col[0]}_{col[1]}" for col in cfb_wide.columns]

    # Merge with subject metadata
    sub_meta = analysis_df[["USUBJID", "TREATMENT", "GENDER"]].drop_duplicates()
    cfb_df = sub_meta.merge(cfb_wide, on="USUBJID", how="left")

    return cfb_df


def differential_analysis_d2(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """D2 differential analysis (DRUG vs PLACEBO) with BH-FDR."""
    marker_cols = [c for c in analysis_df.columns if c.startswith("MARKER_")]

    d2 = analysis_df[analysis_df["VISIT"] == "D2"].copy()

    print("\n=== D2 subset info (analysis dataset) ===")
    print("D2 rows:", len(d2))
    print("D2 TREATMENT counts:")
    print(d2["TREATMENT"].value_counts())

    drug_mask = d2["TREATMENT"] == "DRUG"
    plac_mask = d2["TREATMENT"] == "PLACEBO"

    pvals = []
    marker_list = []

    for m in marker_cols:
        vals_drug = d2.loc[drug_mask, m].dropna()
        vals_plac = d2.loc[plac_mask, m].dropna()
        # Require a minimum size to avoid silly tests
        if len(vals_drug) >= 3 and len(vals_plac) >= 3:
            stat, p = ttest_ind(vals_drug, vals_plac, equal_var=False)
            pvals.append(p)
            marker_list.append(m)

    if not pvals:
        print("\nWARNING: No markers met minimum sample size for t-test.")
        return pd.DataFrame(columns=["marker", "p_value", "q_value"])

    pvals = np.array(pvals)
    qvals = bh_fdr(pvals)

    stats_df = pd.DataFrame(
        {
            "marker": marker_list,
            "p_value": pvals,
            "q_value": qvals,
        }
    ).sort_values("q_value")

    print("\n=== Biomarker differential analysis at D2 ===")
    print("Total markers tested:", len(stats_df))

    print("\nTop 10 markers by q-value:")
    print(stats_df.head(10))

    # Significant markers at FDR <= 1%
    sig_df = stats_df[stats_df["q_value"] <= 0.01].copy()
    print("\nSignificant markers at FDR <= 1%:")
    print(sig_df)

    if len(sig_df) == 0:
        print("\nNo markers passed FDR <= 1%. Using top 3 markers by q-value instead:")
        print(stats_df.head(3)["marker"].tolist())

    return stats_df


def plot_time_courses(
    analysis_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    fdr_threshold: float = 0.01,
    save_dir: Optional[Path] = None,
) -> None:
    """Time-course plots for significant (or top) markers."""
    if stats_df.empty:
        print("\nNo stats available to plot.")
        return

    # Decide which markers to plot
    sig_df = stats_df[stats_df["q_value"] <= fdr_threshold].copy()
    if len(sig_df) > 0:
        sig_markers = sig_df["marker"].tolist()
    else:
        sig_markers = stats_df.head(3)["marker"].tolist()

    # helper lookup dicts for p/q values
    pval_lookup = dict(zip(stats_df["marker"], stats_df["p_value"]))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for m in sig_markers:
        plt.figure()

        # We'll use numeric x positions to make annotation easier
        visits_order = ["D0", "D1", "D2"]
        x_labels = [v for v in visits_order if v in analysis_df["VISIT"].unique()]
        x = np.arange(len(x_labels))

        for treatment in ["DRUG", "PLACEBO"]:
            sub = analysis_df[analysis_df["TREATMENT"] == treatment]
            means = sub.groupby("VISIT")[m].mean()

            # guard in case some visits are missing
            y_vals = [means.loc[v] if v in means.index else np.nan for v in x_labels]
            plt.plot(x, y_vals, marker="o", label=treatment)

        # axis labels/ticks
        plt.xlabel("VISIT")
        plt.ylabel(m)
        plt.title(f"Mean {m} over time by TREATMENT")
        plt.xticks(x, x_labels)
        plt.legend()

        # === Add p-value annotation at D2 if present ===
        p = pval_lookup.get(m, None)

        # Only show something if p <= 0.01
        if p is not None and p <= 0.01:
            ax = plt.gca()
            ax.text(
                0.65,
                0.98,  # near top-right
                "Significant (p ≤ 0.01)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    fc="white",
                    ec="black",
                    alpha=0.8,
                ),
            )

        plt.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"timecourse_{m}.png"
            plt.savefig(out_path)
            print(f"Saved plot for {m} to {out_path}")
            plt.close()
        else:
            plt.show()


# =========================================================
# Utilities
# =========================================================
def download_from_s3(s3_uri: str, destination: Path) -> Path:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    boto3.resource("s3").Bucket(bucket).download_file(key, str(destination))
    return destination


# =========================================================
# Main
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical biomarker analysis (QC, CFB, D2 diff, plots)."
    )
    parser.add_argument(
        "--csv-path",
        "--csv_path",
        "--input-data",
        dest="csv_path",
        type=str,
        required=True,
        help="Local or S3 path to clinical.csv file.",
    )
    parser.add_argument(
        "--plots-dir",
        dest="plots_dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Directory where plots will be stored (default: /opt/ml/processing/output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_arg = args.csv_path
    plots_dir = Path(args.plots_dir)

    if csv_arg.startswith("s3://"):
        local_csv = Path("/opt/ml/processing/input/clinical.csv")
        csv_path = download_from_s3(csv_arg, local_csv)
    else:
        csv_path = Path(csv_arg)

    # Always use output directory for plots inside processing output
    save_plots_dir = plots_dir

    # 1. Load data
    df = load_data(csv_path)

    # 2. Descriptive stats
    descriptive_stats(df)

    # 3. Missingness & QC
    analysis_df = qc_missingness(df)

    # 4. Compute CFB
    cfb_df = compute_cfb(analysis_df)
    print("\n=== CFB dataframe shape ===")
    print(cfb_df.shape)

    # 5. Differential analysis at D2
    stats_df = differential_analysis_d2(analysis_df)

    # 6. Time-course plots → saved to output directory
    plot_time_courses(analysis_df, stats_df, fdr_threshold=0.01, save_dir=save_plots_dir)


if __name__ == "__main__":
    main()

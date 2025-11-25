#!/usr/bin/env python
"""
Clinical biomarker analysis script for SageMaker Processing.

- Downloads clinical.csv from an S3 path
- Basic descriptive stats
- Missingness & QC
- Change From Baseline (CFB) for biomarkers
- Differential analysis at D2 (DRUG vs PLACEBO) with BH-FDR
- Time-course plots for significant markers → saved to /opt/ml/processing/output/
- ML train/validation dataset built from D2, using significant biomarkers
"""
from sklearn.model_selection import train_test_split  # <<< NEW
from scipy.stats import ttest_ind
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
import subprocess
import sys
import os

# <<< NEW: install scikit-learn as well
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
    ]
)


plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Split an S3 URI into bucket and prefix components."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI starting with 's3://', got: {s3_uri}")

    path = s3_uri[5:]
    if "/" not in path:
        return path, ""
    bucket, prefix = path.split("/", 1)
    return bucket, prefix.rstrip("/")


def download_s3_prefix(s3_uri: str, local_dir: str) -> None:
    """Download the full contents of an S3 prefix into ``local_dir``."""
    bucket, prefix = parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    logger.info("Downloading dataset from %s to %s", s3_uri, local_dir)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, prefix) if prefix else key
            dest_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            logger.debug("Downloading s3://%s/%s -> %s", bucket, key, dest_path)
            client.download_file(bucket, key, dest_path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s (%d rows)", path, len(df))


def perform_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    validation_size = max(1, int(0.1 * len(shuffled)))
    validation = shuffled.iloc[:validation_size]
    train = shuffled.iloc[validation_size:]
    return train, validation


def three_way_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation/test sets.

    Validation size is 10% of the non-test portion (at least one row when possible).
    """

    if df.empty:
        return df.copy(), df.copy(), df.copy()

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_size = max(1, int(test_fraction * len(shuffled)))
    test = shuffled.iloc[:test_size]
    remaining = shuffled.iloc[test_size:]

    if remaining.empty:
        return (
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
            test,
        )

    train, validation = perform_split(remaining)
    return train, validation, test


def prepare_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    base_dir: Path,
    cleaned_full: Optional[pd.DataFrame] = None,
) -> None:
    output_dirs = {
        "train": base_dir / "train",
        "validation": base_dir / "validation",
        "test": base_dir / "test",
        "clean": base_dir / "clean",
    }

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    train_path = output_dirs["train"] / "train.csv"
    validation_path = output_dirs["validation"] / "validation.csv"
    test_path = output_dirs["test"] / "test.csv"

    save_dataframe(train, train_path)
    logger.info("Cleaned training split saved to %s", train_path)

    save_dataframe(validation, validation_path)
    logger.info("Cleaned validation split saved to %s", validation_path)

    save_dataframe(test, test_path)
    logger.info("Cleaned test split saved to %s", test_path)

    if cleaned_full is not None:
        clean_full_path = output_dirs["clean"] / "clean.csv"
        save_dataframe(cleaned_full, clean_full_path)
        logger.info("Full cleaned dataset saved to %s", clean_full_path)


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
# Column validation and cleaning
# =========================================================
def validate_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate required columns exist and clean column names.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print("\n=== Column Validation ===")
    print(f"Available columns: {df.columns.tolist()}")

    # Required columns mapping (case-insensitive check)
    required_cols = {
        "VISIT": ["VISIT", "visit", "Visit", "AVISIT", "AVISITN"],
        "TREATMENT": ["TREATMENT", "treatment", "TRT", "ARM", "ARMCD"],
        "USUBJID": ["USUBJID", "usubjid", "SUBJID", "subject_id", "SUBJECT"],
        "GENDER": ["GENDER", "gender", "SEX", "sex"],
    }

    # Try to find and standardize each required column
    for std_name, variants in required_cols.items():
        found = False
        for variant in variants:
            if variant in df.columns:
                if variant != std_name:
                    print(f"Renaming column '{variant}' to '{std_name}'")
                    df = df.rename(columns={variant: std_name})
                found = True
                break

        if not found:
            # Check for partial matches
            partial_matches = [
                col
                for col in df.columns
                if any(v.lower() in col.lower() for v in variants)
            ]
            if partial_matches:
                raise ValueError(
                    f"Could not find required column '{std_name}'. "
                    f"Possible matches found: {partial_matches}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            else:
                raise ValueError(
                    f"Required column '{std_name}' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )

    # Validate VISIT values
    if "VISIT" in df.columns:
        unique_visits = df["VISIT"].unique()
        print(f"\nUnique VISIT values: {unique_visits}")

        # Check if visits need standardization
        visit_mapping = {}
        for visit in unique_visits:
            if pd.notna(visit):
                visit_str = str(visit).strip().upper()
                # Handle various formats: D0, Day 0, Day0, 0, etc.
                if "0" in visit_str or "BASELINE" in visit_str or "BL" in visit_str:
                    visit_mapping[visit] = "D0"
                elif "1" in visit_str:
                    visit_mapping[visit] = "D1"
                elif "2" in visit_str:
                    visit_mapping[visit] = "D2"

        if visit_mapping:
            print(f"Applying visit mapping: {visit_mapping}")
            df["VISIT"] = df["VISIT"].map(lambda x: visit_mapping.get(x, x))
            print(f"Standardized VISIT values: {df['VISIT'].unique()}")

    # Validate TREATMENT values
    if "TREATMENT" in df.columns:
        unique_treatments = df["TREATMENT"].unique()
        print(f"\nUnique TREATMENT values: {unique_treatments}")

        # Standardize treatment names
        treatment_mapping = {}
        for trt in unique_treatments:
            if pd.notna(trt):
                trt_str = str(trt).strip().upper()
                if "DRUG" in trt_str or "ACTIVE" in trt_str or "TRT" in trt_str:
                    treatment_mapping[trt] = "DRUG"
                elif (
                    "PLACEBO" in trt_str
                    or "PBO" in trt_str
                    or "CONTROL" in trt_str
                ):
                    treatment_mapping[trt] = "PLACEBO"

        if treatment_mapping:
            print(f"Applying treatment mapping: {treatment_mapping}")
            df["TREATMENT"] = df["TREATMENT"].map(
                lambda x: treatment_mapping.get(x, x)
            )
            print(f"Standardized TREATMENT values: {df['TREATMENT'].unique()}")

    return df


# =========================================================
# Core analysis steps
# =========================================================
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Validate and clean columns first
    df = validate_and_clean_columns(df)

    marker_cols = [c for c in df.columns if c.startswith("MARKER_")]

    print("\n=== Basic Info ===")
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
    """
    D2 differential analysis (DRUG vs PLACEBO) with BH-FDR.

    IMPORTANT: This function expects analysis_df in LONG format with VISIT column intact.
    """
    marker_cols = [c for c in analysis_df.columns if c.startswith("MARKER_")]

    # Verify VISIT column exists (should be in long format)
    if "VISIT" not in analysis_df.columns:
        raise ValueError(
            f"VISIT column not found in analysis dataframe. "
            f"This function requires long-format data with VISIT column. "
            f"Available columns: {analysis_df.columns.tolist()}"
        )

    # Filter to D2 only
    d2 = analysis_df[analysis_df["VISIT"] == "D2"].copy()

    print("\n=== D2 subset info (analysis dataset) ===")
    print("D2 rows:", len(d2))

    if len(d2) == 0:
        print("WARNING: No rows found for VISIT='D2'")
        print(f"Available VISIT values: {analysis_df['VISIT'].unique()}")
        return pd.DataFrame(columns=["marker", "p_value", "q_value"])

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


# =========================================================
# NEW: Build ML dataset like in your notebook
# =========================================================
def build_ml_dataset(
    analysis_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    fdr_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, list]:
    """
    Build ML-ready dataset at D2:
      - choose significant markers (FDR <= threshold, else top 3),
      - restrict to VISIT=='D2',
      - drop rows with missing values in those markers,
      - add binary LABEL (1=DRUG, 0=PLACEBO).
    Returns (ml_df, sig_markers).
    """
    if stats_df.empty:
        print("\nWARNING: stats_df is empty; cannot build ML dataset.")
        return pd.DataFrame(), []

    # choose markers
    sig_df = stats_df[stats_df["q_value"] <= fdr_threshold].copy()
    if len(sig_df) > 0:
        sig_markers = sig_df["marker"].tolist()
    else:
        sig_markers = stats_df.head(3)["marker"].tolist()
        print("\nNo markers passed FDR <= 1%. Using top 3 markers by q-value instead:")
        print(sig_markers)

    if "VISIT" not in analysis_df.columns:
        raise ValueError(
            "VISIT column not found in analysis_df; cannot build D2 ML dataset."
        )

    d2 = analysis_df[analysis_df["VISIT"] == "D2"].copy()
    if d2.empty:
        print("\nWARNING: No VISIT == 'D2' rows; cannot build ML dataset.")
        return pd.DataFrame(), sig_markers

    # drop rows with any missing in sig_markers
    d2_ml = d2.dropna(subset=sig_markers).copy()
    if d2_ml.empty:
        print("\nWARNING: After dropping NA for significant markers, no rows left.")
        return pd.DataFrame(), sig_markers

    # build label: 1=DRUG, 0=PLACEBO
    y = (d2_ml["TREATMENT"] == "DRUG").astype(int).values

    print("\n=== ML dataset (D2, selected markers) ===")
    print("Number of rows:", d2_ml.shape[0])
    print("Number of features:", len(sig_markers))
    unique, counts = np.unique(y, return_counts=True)
    print("Class balance (0=PLACEBO, 1=DRUG):")
    print(dict(zip(unique, counts)))

    # Same sanity check as in your notebook
    if len(np.unique(y)) < 2 or d2_ml.shape[0] < 6:
        print(
            "\nNot enough data to train a meaningful ML model; returning empty ML dataset."
        )
        return pd.DataFrame(), sig_markers

    d2_ml = d2_ml.copy()
    d2_ml["LABEL"] = y  # binary target

    # Order columns: meta, features, label
    base_cols = [c for c in ["USUBJID", "TREATMENT", "GENDER"] if c in d2_ml.columns]
    ordered_cols = base_cols + sig_markers + ["LABEL"]
    d2_ml = d2_ml[ordered_cols]

    return d2_ml, sig_markers


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
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "fc": "white",
                    "ec": "black",
                    "alpha": 0.8,
                },
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
    """Download the first CSV file from an S3 prefix or path."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3")

    # list all objects under the prefix
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in resp:
        raise FileNotFoundError(f"No objects found at {s3_uri}")

    # find the first CSV
    csv_keys = [obj["Key"] for obj in resp["Contents"] if obj["Key"].endswith(".csv")]
    if not csv_keys:
        raise FileNotFoundError(f"No CSV files found under {s3_uri}")

    key = csv_keys[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s to %s", bucket, key, destination)
    s3.download_file(bucket, key, str(destination))

    return destination


def stratified_train_val_test(
    df: pd.DataFrame,
    label_col: str = "LABEL",
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits when possible.

    * ``test_fraction`` is applied to the full dataset.
    * ``val_fraction`` is applied to the remaining portion (train+val).
    * Guarantees at least one row in test/validation when the input has
      enough rows; falls back to unstratified splitting when stratification
      is impossible (e.g., single-class labels or very small datasets).
    """

    if df.empty:
        return df.copy(), df.copy(), df.copy()

    n_rows = len(df)
    stratify_labels: Optional[pd.Series]

    # Determine whether stratification is feasible
    if df[label_col].nunique() > 1 and n_rows >= 4:
        stratify_labels = df[label_col]
    else:
        stratify_labels = None
        logger.warning(
            "Unable to stratify splits (unique labels: %d, rows: %d); "
            "falling back to unstratified splitting.",
            df[label_col].nunique(),
            n_rows,
        )

    # Compute test size with an integer floor of the fraction but ensure at least 1 and
    # leave room for train/validation
    test_size = max(1, int(test_fraction * n_rows))
    if test_size >= n_rows:
        test_size = n_rows - 1

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels,
    )

    if train_val.empty:
        return (
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
            test.reset_index(drop=True),
        )

    # Validation fraction is applied to the remaining train/val pool
    val_size = max(1, int(val_fraction * len(train_val)))
    if val_size >= len(train_val):
        val_size = len(train_val) - 1

    val_stratify: Optional[pd.Series]
    if (
        stratify_labels is not None
        and train_val[label_col].nunique() > 1
        and len(train_val) >= 3
    ):
        val_stratify = train_val[label_col]
    else:
        val_stratify = None

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=42,
        stratify=val_stratify,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# =========================================================
# Main
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical biomarker analysis (QC, CFB, D2 diff, plots, ML splits)."
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
        default="/opt/ml/processing/figures",
        help=(
            "Directory where plots will be stored "
            "(default: /opt/ml/processing/figures)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_arg = args.csv_path
    plots_dir = Path(args.plots_dir)
    output_base_dir = Path("/opt/ml/processing")

    if csv_arg.startswith("s3://"):
        local_csv = output_base_dir / "input" / "clinical.csv"
        csv_path = download_from_s3(csv_arg, local_csv)
    else:
        csv_path = Path(csv_arg)

    # Always use output directory for plots inside processing output
    save_plots_dir = plots_dir

    # 1. Load data (includes column validation)
    df = load_data(csv_path)

    # 2. Descriptive stats
    descriptive_stats(df)

    # 3. Missingness & QC
    analysis_df = qc_missingness(df)

    # 4. Compute CFB (creates wide-format, per-subject)
    cfb_df = compute_cfb(analysis_df)
    print("\n=== CFB dataframe shape ===")
    print(cfb_df.shape)

    # Optionally save full CFB-wide dataset for reference
    cfb_dir = output_base_dir / "cfb"
    cfb_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(cfb_df, cfb_dir / "cfb.csv")
    logger.info("CFB-wide dataset saved to %s", cfb_dir / "cfb.csv")

    # 5. Differential analysis at D2 (LONG format)
    stats_df = differential_analysis_d2(analysis_df)

    # 6. Build ML-ready dataset at D2 using significant markers
    ml_df, sig_markers = build_ml_dataset(analysis_df, stats_df)

    if ml_df.empty:
        # Fallback to previous behaviour: 3-way split on CFB data
        logger.warning(
            "ML dataset is empty or insufficient; falling back to CFB-based three-way split."
        )
        train_df, validation_df, test_df = three_way_split(cfb_df)
        cleaned_full = cfb_df
    else:
        train_df, validation_df, test_df = stratified_train_val_test(
            ml_df,
            label_col="LABEL",
            test_fraction=0.2,
            val_fraction=0.25,  # 20% test; 25% of remaining ≈ 20% validation
        )
        cleaned_full = ml_df

    logger.info(
        "Final ML splits: %d train rows, %d validation rows, %d test rows",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    # 7. Save train/validation/test and full cleaned ML dataset
    prepare_outputs(
        train=train_df,
        validation=validation_df,
        test=test_df,
        base_dir=output_base_dir,
        cleaned_full=cleaned_full,
    )

    # 8. Time-course plots → saved to output directory
    plot_time_courses(
        analysis_df,
        stats_df,
        fdr_threshold=0.01,
        save_dir=save_plots_dir,
    )


if __name__ == "__main__":
    main()

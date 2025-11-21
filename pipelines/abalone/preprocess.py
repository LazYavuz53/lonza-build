#!/usr/bin/env python
"""Custom preprocessing script for the SageMaker pipeline.

This module ports the data cleaning and visualisation workflow defined in
``data_analysis/preprocessing.py`` so it can be executed inside a SageMaker
Processing job.  The script expects peptide binding datasets stored in an S3
prefix and produces:

* cleaned train/validation/test CSV files
* distribution plots saved as PNG images
* a metadata JSON payload describing the processed corpus

The Processing job copies every output directory declared in ``pipeline.py`` to
Amazon S3, therefore the figures are automatically persisted when the step
completes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
from typing import Iterable, Tuple

import boto3
import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "pandas"]
)

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())


AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_VOCAB)


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

    LOGGER.info("Downloading dataset from %s to %s", s3_uri, local_dir)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, prefix) if prefix else key
            dest_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            LOGGER.debug("Downloading s3://%s/%s -> %s", bucket, key, dest_path)
            client.download_file(bucket, key, dest_path)


def fix_allele_format(allele: str) -> str:
    """Fix allele name to follow standard HLA format: HLA-A*02:101."""

    allele = str(allele).strip().upper()
    valid_pattern = re.compile(r"^HLA-[A-Z]\*\d{2}:\d{2,3}$")

    if valid_pattern.match(allele):
        return allele

    allele = allele.replace(" ", "").replace("_", "").replace("--", "-")

    if not allele.startswith("HLA"):
        allele = "HLA-" + allele
    elif not allele.startswith("HLA-"):
        allele = allele.replace("HLA", "HLA-", 1)

    if "*" not in allele:
        allele = re.sub(r"^(HLA-[A-Z])(\d+)", r"\1*\2", allele)

    allele = re.sub(r"(\*\d{2})(\d{2,3})$", r"\1:\2", allele)
    return allele


def clean_allele_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    LOGGER.info("[%s] Checking and fixing allele formats", name)
    valid_pattern = re.compile(r"^HLA-[A-Z]\*\d{2}:\d{2,3}$")
    corrections_made = False
    corrected_alleles = []

    for allele in df["allele"]:
        corrected = fix_allele_format(allele)
        if corrected != allele:
            corrections_made = True
        corrected_alleles.append(corrected)

    df["allele"] = corrected_alleles

    invalid_mask = ~df["allele"].str.match(valid_pattern)
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        LOGGER.warning(
            "[%s] %d allele entries remain invalid after correction.",
            name,
            invalid_count,
        )
    elif corrections_made:
        LOGGER.info("[%s] All allele names corrected successfully.", name)
    else:
        LOGGER.info("[%s] No allele mismatches found.", name)

    return df


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {"peptide", "allele", "hit"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def dataset_stats(df: pd.DataFrame, name: str) -> None:
    LOGGER.info(
        "\n=== Dataset Stats: %s ===\nRows: %s\nColumns: %s\nUnique alleles: %s",
        name,
        f"{len(df):,}",
        list(df.columns),
        df["allele"].nunique(),
    )
    if "hit" in df.columns:
        LOGGER.info("Label distribution (hit):\n%s", df["hit"].value_counts(dropna=False))
    lens = df["peptide"].astype(str).str.len()
    LOGGER.info("Peptide length stats:\n%s", lens.describe())


def remove_exact_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    LOGGER.info("[%s] Removed exact duplicate rows: %d", name, before - len(df2))
    return df2


def remove_conflicting_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    grp = df.groupby(["peptide", "allele"])["hit"].nunique()
    conflicts = grp[grp > 1]
    if conflicts.empty:
        LOGGER.info("[%s] Conflicting duplicates: none", name)
        return df
    before = len(df)
    bad_keys = set(conflicts.index)
    mask = df.set_index(["peptide", "allele"]).index.isin(bad_keys)
    df2 = df[~mask].copy()
    LOGGER.info(
        "[%s] Conflicting (peptide, allele) pairs: %d | Rows removed: %d",
        name,
        len(conflicts),
        before - len(df2),
    )
    return df2


def handle_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = df.isna().sum()
    if miss.sum() == 0:
        LOGGER.info("[%s] Missing values: none", name)
        return df
    LOGGER.info("[%s] Missing values per column:\n%s", name, miss)
    before = len(df)
    df2 = df.dropna(subset=["peptide", "allele", "hit"]).copy()
    LOGGER.info(
        "[%s] Rows dropped due to missing peptide/allele/hit: %d",
        name,
        before - len(df2),
    )
    return df2


def is_valid_peptide(seq: str) -> bool:
    s = str(seq).strip().upper()
    if len(s) == 0:
        return False
    return set(s).issubset(AA_SET)


def clean_invalid_peptides(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lens_before = df["peptide"].astype(str).str.len().describe()
    valid_mask = df["peptide"].astype(str).str.upper().apply(is_valid_peptide)
    invalid = (~valid_mask).sum()
    df2 = df[valid_mask].copy()
    LOGGER.info("[%s] Non-standard/invalid peptide rows removed: %d", name, invalid)
    LOGGER.debug(
        "[%s] Sequence length stats BEFORE removal:\n%s", name, lens_before
    )
    LOGGER.debug(
        "[%s] Sequence length stats AFTER removal:\n%s",
        name,
        df2["peptide"].astype(str).str.len().describe(),
    )
    return df2


def filter_test_alleles_in_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_alleles = set(train["allele"].unique())
    test_alleles = set(test["allele"].unique())
    unknown = sorted(list(test_alleles - train_alleles))
    if unknown:
        before = len(test)
        test2 = test[test["allele"].isin(train_alleles)].copy()
        LOGGER.warning(
            "[TEST] Alleles not present in TRAIN: %s", unknown,
        )
        LOGGER.info(
            "[TEST] Removed rows with unknown alleles: %d",
            before - len(test2),
        )
        return test2
    LOGGER.info("[TEST] All test alleles exist in training")
    return test


def compute_max_len(train_df: pd.DataFrame) -> int:
    max_len = int(train_df["peptide"].astype(str).str.len().max())
    LOGGER.info("[FE] Max sequence length (from CLEANED TRAIN): %d", max_len)
    return max_len


def plot_class_distribution(df: pd.DataFrame, out_prefix: str, out_dir: str) -> None:
    plt.figure(figsize=(4, 4))
    sns.countplot(x="hit", data=df)
    plt.title("General Class Distribution (hit)")
    plt.xlabel("Class (hit)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_class_distribution.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    allele_counts = (
        df.groupby(["allele", "hit"]).size().reset_index(name="count")
    )
    top_alleles = df["allele"].value_counts().head(15).index
    sns.barplot(
        x="allele",
        y="count",
        hue="hit",
        data=allele_counts[allele_counts["allele"].isin(top_alleles)],
    )
    plt.title("Class Distribution per Allele (top 15)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{out_prefix}_class_distribution_per_allele.png")
    )
    plt.close()


def plot_length_distribution(df: pd.DataFrame, out_prefix: str, out_dir: str) -> None:
    df = df.copy()
    df["length"] = df["peptide"].astype(str).str.len()

    plt.figure(figsize=(5, 4))
    sns.histplot(df["length"], bins=20, kde=True, color="steelblue")
    plt.title("General Peptide Length Distribution")
    plt.xlabel("Peptide length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_length_distribution.png"))
    plt.close()

    top_alleles = df["allele"].value_counts().head(15).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="allele",
        y="length",
        data=df[df["allele"].isin(top_alleles)],
        showfliers=False,
    )
    plt.title("Peptide Length Distribution per Allele (top 15)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{out_prefix}_length_distribution_per_allele.png")
    )
    plt.close()


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s (%d rows)", path, len(df))


def perform_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    validation_size = max(1, int(0.1 * len(shuffled)))
    validation = shuffled.iloc[:validation_size]
    train = shuffled.iloc[validation_size:]
    return train, validation


def prepare_outputs(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                    base_dir: str) -> None:
    save_dataframe(train, os.path.join(base_dir, "train", "train.csv"))
    save_dataframe(
        validation, os.path.join(base_dir, "validation", "validation.csv")
    )
    save_dataframe(test, os.path.join(base_dir, "test", "test.csv"))


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parsed = parser.parse_args(args=args)

    base_dir = "/opt/ml/processing"
    raw_dir = os.path.join(base_dir, "raw")
    dataset_dir = os.path.join(raw_dir, "datasets")
    figures_dir = os.path.join(base_dir, "figures")
    metadata_dir = os.path.join(base_dir, "metadata")

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(figures_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(metadata_dir).mkdir(parents=True, exist_ok=True)

    download_s3_prefix(parsed.input_data, dataset_dir)

    train_parts = []
    for fold in sorted(pathlib.Path(dataset_dir).glob("fold_*.csv")):
        df = load_csv_safe(str(fold))
        df["peptide"] = df["peptide"].astype(str).str.upper().str.strip()
        df["allele"] = df["allele"].astype(str).str.strip()
        df = clean_allele_column(df, f"TRAIN file {fold.name}")
        train_parts.append(df)

    if not train_parts:
        raise RuntimeError(
            "No training folds found in the provided dataset. Expected files matching 'fold_*.csv'."
        )

    train_raw = pd.concat(train_parts, ignore_index=True)

    test_path = pathlib.Path(dataset_dir) / "test.csv"
    test_raw = load_csv_safe(str(test_path))
    test_raw["peptide"] = test_raw["peptide"].astype(str).str.upper().str.strip()
    test_raw["allele"] = test_raw["allele"].astype(str).str.strip()
    test_raw = clean_allele_column(test_raw, "TEST")

    dataset_stats(train_raw, "TRAIN (raw)")
    dataset_stats(test_raw, "TEST (raw)")

    train_clean = remove_exact_duplicates(train_raw, "TRAIN")
    test_clean = remove_exact_duplicates(test_raw, "TEST")

    train_clean = remove_conflicting_duplicates(train_clean, "TRAIN")
    if "hit" in test_clean.columns:
        test_clean = remove_conflicting_duplicates(test_clean, "TEST")

    train_clean = handle_missing(train_clean, "TRAIN")
    test_clean = handle_missing(test_clean, "TEST")

    train_clean = clean_invalid_peptides(train_clean, "TRAIN")
    test_clean = clean_invalid_peptides(test_clean, "TEST")

    test_clean = filter_test_alleles_in_train(train_clean, test_clean)

    dataset_stats(train_clean, "TRAIN (cleaned)")
    dataset_stats(test_clean, "TEST (cleaned)")

    LOGGER.info("Generating distribution plots")
    plot_class_distribution(train_clean, out_prefix="train", out_dir=figures_dir)
    plot_length_distribution(train_clean, out_prefix="train", out_dir=figures_dir)

    max_seq_len = compute_max_len(train_clean)

    clean_dir = os.path.join(base_dir, "clean")
    pathlib.Path(clean_dir).mkdir(parents=True, exist_ok=True)
    save_dataframe(train_clean, os.path.join(clean_dir, "train_clean.csv"))
    save_dataframe(test_clean, os.path.join(clean_dir, "test_clean.csv"))

    metadata = {
        "aa_vocab": AA_VOCAB,
        "max_seq_len": max_seq_len,
        "train_alleles": sorted(train_clean["allele"].unique()),
    }
    metadata_path = os.path.join(metadata_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Saved metadata to %s", metadata_path)

    train_split, validation_split = perform_split(train_clean)
    prepare_outputs(train_split, validation_split, test_clean, base_dir)


if __name__ == "__main__":
    main()

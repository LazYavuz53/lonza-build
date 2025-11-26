"""Evaluation script that reports binary classification metrics."""
from __future__ import annotations

import json
import logging
import pathlib
import tarfile
from typing import Tuple
import subprocess
import sys


def ensure_dependencies():
    """Install required Python packages if missing, then import them globally."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "numpy==1.23.5",
            "matplotlib==3.7.5",
            "xgboost==1.7.6",
        ]
    )

    # Import dependencies dynamically after installation
    global np, pd, plt, xgb
    global calibration_curve, accuracy_score, classification_report, roc_auc_score, roc_curve
    global matplotlib

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
        roc_curve,
    )


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

NON_FEATURE_COLUMNS = ["USUBJID", "TREATMENT", "GENDER"]


def load_test_split(csv_path: pathlib.Path) -> Tuple["np.ndarray", "pd.DataFrame"]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected test split at {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Test split is empty; returning no labels/features.")
        return np.array([]), df

    if "LABEL" in df.columns:
        y_test = df.pop("LABEL").to_numpy()
    else:
        y_test = df.iloc[:, 0].to_numpy()
        df = df.iloc[:, 1:]

    feature_df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    return y_test, feature_df


def load_model(model_artifact: pathlib.Path) -> "xgb.Booster":
    booster = xgb.Booster()
    booster.load_model(model_artifact)
    return booster


if __name__ == "__main__":
    # Install dependencies BEFORE running evaluation and make them available globally
    ensure_dependencies()

    logger.info("Starting evaluation.")

    model_path = pathlib.Path("/opt/ml/processing/model/model.tar.gz")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    booster = load_model(pathlib.Path("xgboost-model"))

    test_path = pathlib.Path("/opt/ml/processing/test/test.csv")
    y_test, X_test = load_test_split(test_path)

    output_dir = pathlib.Path("/opt/ml/processing/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(y_test) == 0:
        logger.warning("No test rows available; emitting zeroed metrics.")
        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": 0.0, "standard_deviation": 0.0},
                "roc_auc": {"value": 0.0, "standard_deviation": 0.0},
            },
        }
    else:
        dtest = xgb.DMatrix(X_test.values)
        probabilities = booster.predict(dtest)

        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        try:
            roc_auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            roc_auc = 0.0

        y_np = y_test

        if len(np.unique(y_np)) > 1:
            fpr, tpr, _ = roc_curve(y_np, probabilities)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--")
            plt.legend()
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve – D2 Important Biomarkers")
            plt.grid(True)
            plt.savefig(output_dir / "roc_curve.png", bbox_inches="tight")

        try:
            prob_true, prob_pred = calibration_curve(y_np, probabilities, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, marker="o")
            plt.plot([0, 1], [0, 1], "--")
            plt.title("Calibration Curve")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.grid(True)
            plt.savefig(output_dir / "calibration_curve.png", bbox_inches="tight")
        except ValueError:
            pass

        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": float(accuracy), "standard_deviation": 0.0},
                "roc_auc": {"value": float(roc_auc), "standard_deviation": 0.0},
            },
        }

    with (output_dir / "evaluation.json").open("w") as f:
        f.write(json.dumps(report_dict, allow_nan=False))

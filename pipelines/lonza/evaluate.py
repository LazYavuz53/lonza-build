"""Evaluation script that reports binary classification metrics."""
import json
import logging
import pathlib
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def load_test_split(csv_path: pathlib.Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load the test split and return labels and feature frame."""

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

    return y_test, df


def load_model(model_artifact: pathlib.Path) -> xgboost.Booster:
    """Load a persisted XGBoost booster from the extracted model bundle."""

    booster = xgboost.Booster()
    booster.load_model(model_artifact)
    return booster


if __name__ == "__main__":
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
        logger.warning("No test rows available; emitting zeroed metrics instead of NaN values.")
        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": 0.0, "standard_deviation": 0.0},
                "roc_auc": {"value": 0.0, "standard_deviation": 0.0},
            },
        }
    else:
        logger.info("Performing predictions against test data (%d rows).", len(y_test))
        dtest = xgboost.DMatrix(X_test.values)
        probabilities = booster.predict(dtest)

        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        y_test_np = np.asarray(y_test)

        try:
            roc_auc = roc_auc_score(y_test_np, probabilities)
        except ValueError:
            logger.warning("ROC AUC could not be computed because only one class is present. Using 0.0.")
            roc_auc = 0.0

        print("\nAccuracy:", accuracy_score(y_test_np, predictions))
        print("ROC AUC:", roc_auc)
        print("\nClassification Report:\n", classification_report(y_test_np, predictions, zero_division=0))

        if len(np.unique(y_test_np)) > 1:
            fpr, tpr, _ = roc_curve(y_test_np, probabilities)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--")
            plt.legend()
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve – D2 Important Biomarkers")
            plt.grid(True)
            roc_path = output_dir / "roc_curve.png"
            plt.savefig(roc_path, bbox_inches="tight")
        else:
            logger.warning("Skipping ROC curve plot because only one class is present in y_test.")

        try:
            prob_true, prob_pred = calibration_curve(y_test_np, probabilities, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, marker="o")
            plt.plot([0, 1], [0, 1], "--")
            plt.title("Calibration Curve")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.grid(True)
            calibration_path = output_dir / "calibration_curve.png"
            plt.savefig(calibration_path, bbox_inches="tight")
        except ValueError:
            logger.warning("Skipping calibration curve because it cannot be computed for this dataset.")

        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": float(accuracy), "standard_deviation": 0.0},
                "roc_auc": {"value": float(roc_auc), "standard_deviation": 0.0},
            },
        }

    accuracy_value = report_dict["classification_metrics"]["accuracy"]["value"]
    logger.info("Writing out evaluation report with accuracy: %s", accuracy_value)
    evaluation_path = output_dir / "evaluation.json"
    with evaluation_path.open("w") as f:
        f.write(json.dumps(report_dict, allow_nan=False))

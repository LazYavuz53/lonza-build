"""Evaluation script that reports binary classification metrics."""
import json
import logging
import pathlib
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import accuracy_score, roc_auc_score

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

    if len(y_test) == 0:
        logger.warning("No test rows available; emitting empty metrics with NaN values.")
        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": float("nan"), "standard_deviation": 0.0},
                "roc_auc": {"value": float("nan"), "standard_deviation": 0.0},
            },
        }
    else:
        logger.info("Performing predictions against test data (%d rows).", len(y_test))
        dtest = xgboost.DMatrix(X_test.values)
        probabilities = booster.predict(dtest)

        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        try:
            roc_auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            roc_auc = float("nan")

        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": float(accuracy), "standard_deviation": 0.0},
                "roc_auc": {"value": float(roc_auc), "standard_deviation": 0.0},
            },
        }

    output_dir = pathlib.Path("/opt/ml/processing/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_value = report_dict["classification_metrics"]["accuracy"]["value"]
    logger.info("Writing out evaluation report with accuracy: %s", accuracy_value)
    evaluation_path = output_dir / "evaluation.json"
    with evaluation_path.open("w") as f:
        f.write(json.dumps(report_dict))

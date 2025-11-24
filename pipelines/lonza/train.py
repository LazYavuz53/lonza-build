"""Training script that consumes pre-split datasets from preprocess.py outputs."""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NON_FEATURE_COLUMNS = ["USUBJID", "TREATMENT", "GENDER"]
LABEL_COLUMN = "LABEL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=50, help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (eta)")
    parser.add_argument("--max-depth", type=int, default=3, help="Tree depth")
    parser.add_argument("--random-state", type=int, default=53, help="Random seed")
    return parser.parse_args()


def _load_split(path: Path) -> Tuple[pd.Series, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Expected split at {path} is missing")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Split at {path} is empty")

    if LABEL_COLUMN in df.columns:
        y = df.pop(LABEL_COLUMN)
    else:
        y = df.iloc[:, 0]
        df = df.iloc[:, 1:]

    feature_df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    return y, feature_df


def get_dmatrix(split_dir: Path) -> Tuple[xgb.DMatrix, pd.Series, pd.DataFrame]:
    csv_path = split_dir / f"{split_dir.name}.csv"
    y, X = _load_split(csv_path)
    dmatrix = xgb.DMatrix(X.values, label=y.values)
    return dmatrix, y, X


def train_model(train_dir: Path, validation_dir: Path, args: argparse.Namespace) -> xgb.Booster:
    dtrain, y_train, X_train = get_dmatrix(train_dir)
    evals = [(dtrain, "train")]

    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "seed": args.random_state,
    }

    if validation_dir.exists():
        try:
            dval, _, _ = get_dmatrix(validation_dir)
            evals.append((dval, "validation"))
        except ValueError:
            logger.warning("Validation split is empty; proceeding without it.")
    else:
        logger.warning("Validation directory %s does not exist; proceeding without it.", validation_dir)

    logger.info("Training dataset: %d rows, %d features", X_train.shape[0], X_train.shape[1])
    if len(evals) > 1:
        logger.info("Validation dataset present; using for evaluation during training.")

    booster = xgb.train(
        params=booster_params,
        dtrain=dtrain,
        num_boost_round=args.max_iter,
        evals=evals,
    )
    return booster


def save_model(model: xgb.Booster, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost-model"
    model.save_model(model_path)
    logger.info("Saved model to %s", model_path)


def save_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp)
    logger.info("Saved training metrics to %s", metrics_path)


if __name__ == "__main__":
    args = parse_args()

    input_dir = Path(os.environ.get("SM_INPUT_DATA_DIR", "/opt/ml/input/data"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    train_dir = input_dir / "train"
    validation_dir = input_dir / "validation"

    logger.info("Starting training using splits from preprocess step.")
    booster = train_model(train_dir, validation_dir, args)

    save_model(booster, model_dir)

    # Save simple training metrics
    metrics = {
        "num_boost_round": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
    }
    save_metrics(metrics, output_dir)

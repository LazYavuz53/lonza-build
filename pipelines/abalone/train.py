"""Train a logistic regression model to predict treatment arm at D2."""
import json
import os
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    q = np.empty(m, dtype=float)
    prev_q = 1.0
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


def find_first_csv(data_dir: Path) -> Path:
    """Return the first CSV file under ``data_dir``."""
    candidates = sorted(data_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    return candidates[0]


def differential_analysis_d2(df: pd.DataFrame) -> pd.DataFrame:
    """Perform t-tests for each marker at D2 (DRUG vs PLACEBO) and return stats."""
    marker_cols = [c for c in df.columns if c.startswith("MARKER_")]
    d2 = df[df["VISIT"] == "D2"].copy()

    drug_mask = d2["TREATMENT"] == "DRUG"
    plac_mask = d2["TREATMENT"] == "PLACEBO"

    pvals = []
    marker_list: List[str] = []

    for marker in marker_cols:
        vals_drug = d2.loc[drug_mask, marker].dropna()
        vals_plac = d2.loc[plac_mask, marker].dropna()
        if len(vals_drug) >= 3 and len(vals_plac) >= 3:
            _, p_val = ttest_ind(vals_drug, vals_plac, equal_var=False)
            pvals.append(p_val)
            marker_list.append(marker)

    if not pvals:
        return pd.DataFrame(columns=["marker", "p_value", "q_value"])

    pvals_arr = np.array(pvals)
    qvals = bh_fdr(pvals_arr)

    return (
        pd.DataFrame({"marker": marker_list, "p_value": pvals_arr, "q_value": qvals})
        .sort_values("q_value")
        .reset_index(drop=True)
    )


def select_markers(stats_df: pd.DataFrame, fdr_threshold: float = 0.01) -> List[str]:
    """Choose significant markers or fall back to the top three."""
    if stats_df.empty:
        return []

    sig_df = stats_df[stats_df["q_value"] <= fdr_threshold]
    if not sig_df.empty:
        return sig_df["marker"].tolist()

    return stats_df.head(3)["marker"].tolist()


def train_model(df: pd.DataFrame, markers: Sequence[str]):
    """Train a logistic regression classifier on D2 rows using selected markers."""
    d2_ml = df[df["VISIT"] == "D2"].dropna(subset=markers).copy()
    X = d2_ml[list(markers)].values
    y = (d2_ml["TREATMENT"] == "DRUG").astype(int).values

    print("\n=== ML dataset (D2, selected markers) ===")
    print("Number of rows:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Class balance (0=PLACEBO, 1=DRUG):")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    if len(np.unique(y)) < 2 or X.shape[0] < 6:
        print("\nNot enough data to train a meaningful ML model.")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n=== ML model performance (Logistic Regression) ===")
    print("Accuracy:", acc)
    print("ROC AUC:", roc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["PLACEBO", "DRUG"]))

    print("Confusion matrix (rows=true, cols=pred):")
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_test, y_pred))

    coefs = pipe.named_steps["logreg"].coef_[0]
    coef_df = pd.DataFrame({"marker": markers, "coef_standardized": coefs}).sort_values(
        "coef_standardized", ascending=False
    )
    print("\nFeature importances (standardized coefficients):")
    print(coef_df)

    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(roc),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["PLACEBO", "DRUG"], output_dict=True
        ),
    }

    return pipe, metrics


def main():
    data_dir = Path(os.environ.get("SM_CHANNEL_CLEAN", "/opt/ml/input/data/clean"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_first_csv(data_dir)
    df = pd.read_csv(csv_path)

    stats_df = differential_analysis_d2(df)
    markers = select_markers(stats_df)

    if not markers:
        print("No markers available for model training.")
        markers = []

    model, metrics = train_model(df, markers) if markers else (None, {})

    if model is not None:
        model_path = model_dir / "logreg_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    markers_path = model_dir / "markers.json"
    markers_path.write_text(json.dumps({"markers": markers}))
    print(f"Saved marker list to {markers_path}")

    metrics_path = model_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()

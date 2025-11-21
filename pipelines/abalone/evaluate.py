"""Evaluate the logistic regression treatment classifier on held-out data."""
import json
import tarfile
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


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


def find_first_csv(data_dirs: Sequence[Path]) -> Path:
    """Return the first CSV file found under any of the provided directories."""
    for data_dir in data_dirs:
        candidates = sorted(data_dir.rglob("*.csv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No CSV files found under {[str(d) for d in data_dirs]}")


def differential_analysis_d2(df: pd.DataFrame) -> pd.DataFrame:
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
    if stats_df.empty:
        return []
    sig_df = stats_df[stats_df["q_value"] <= fdr_threshold]
    if not sig_df.empty:
        return sig_df["marker"].tolist()
    return stats_df.head(3)["marker"].tolist()


def load_model(model_dir: Path):
    model_path = model_dir / "logreg_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def extract_model_artifacts(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
    return extract_dir


def evaluate_model(df: pd.DataFrame, markers: Sequence[str], model) -> dict:
    d2_ml = df[df["VISIT"] == "D2"].dropna(subset=markers).copy()
    if d2_ml.empty:
        return {
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "classification_report": {},
            "note": "No D2 rows available for evaluation",
        }

    X = d2_ml[list(markers)].values
    y_true = (d2_ml["TREATMENT"] == "DRUG").astype(int).values

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "accuracy": float(acc),
        "roc_auc": float(roc),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["PLACEBO", "DRUG"], output_dict=True
        ),
    }


def main():
    model_tar_path = Path("/opt/ml/processing/model/model.tar.gz")
    extracted_model_dir = extract_model_artifacts(model_tar_path, Path("/tmp/model"))

    markers_file = extracted_model_dir / "markers.json"
    markers: List[str] = []
    if markers_file.exists():
        markers = json.loads(markers_file.read_text()).get("markers", [])
    else:
        print("No markers.json found; will recompute markers from data.")

    data_dirs = [
        Path("/opt/ml/processing/test"),
        Path("/opt/ml/processing/clean"),
    ]
    csv_path = find_first_csv(data_dirs)
    df = pd.read_csv(csv_path)

    if not markers:
        stats_df = differential_analysis_d2(df)
        markers = select_markers(stats_df)

    if not markers:
        print("No markers available for evaluation.")
        metrics = {
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "classification_report": {},
            "note": "No markers could be derived for evaluation",
        }
    else:
        model = load_model(extracted_model_dir)
        metrics = evaluate_model(df, markers, model)

    evaluation_dir = Path("/opt/ml/processing/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report_dict = {"classification_metrics": metrics}

    evaluation_path = evaluation_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(report_dict, indent=2))
    print(f"Wrote evaluation report to {evaluation_path}")


if __name__ == "__main__":
    main()

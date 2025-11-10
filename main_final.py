
# -*- coding: utf-8 -*-
"""
main_final.py  (ICU mortality pipeline - dataset swapped to 'full_selected_副本.csv' by default)

Enhancements kept:
- Reproducible paths and saving strategy (./output, ./figures)
- Data loading with robust UTF-8 handling for non-ASCII filenames
- Target auto-detection (configurable via --target / --positive-label)
- Proper train/test split with stratification
- Pipelines with train-only fit for imputation/scaling
- Cross-validation (StratifiedKFold) for Logistic & XGBoost
- GridSearchCV hyperparameter tuning
- Metrics: AUROC, AUPRC, Precision/Recall/F1（少数类）
- ROC, PR curves and Confusion Matrix saved to ./figures
- Results table (with figure paths) saved to ./output/metrics_summary.csv
- Youden-J threshold + optional manual override (--threshold 0.121)
- XGBoost feature importance (Top 30) plot
- LSTM stubs for future sequence modeling

Run examples:
    python main_final.py
    python main_final.py --threshold 0.121
    python main_final.py --target mortality --positive-label 1

"""

import argparse
import warnings
from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# XGBoost optional
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    warnings.warn("xgboost is not installed. XGBoost model will be skipped. Install with `pip install xgboost`.")

RANDOM_STATE = 42
DEFAULT_DATA = "full_selected_副本.csv"  # <- swapped default

# =========================
# Paths & Saving Strategy
# =========================
def setup_paths(base_path: Path) -> tuple[Path, Path]:
    base_path = Path(base_path).resolve()
    os.makedirs(base_path, exist_ok=True)
    output_dir = base_path / "output"
    fig_dir = base_path / "figures"
    for d in (output_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    return output_dir, fig_dir

# =========================
# Target Column Detection
# =========================
CANDIDATE_TARGETS = ["mortality", "death", "deceased", "y", "label", "outcome"]

def detect_target_column(df: pd.DataFrame, candidates=CANDIDATE_TARGETS) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # if none found but a binary-like column exists, try the last column
    last = df.columns[-1]
    if df[last].nunique() == 2:
        warnings.warn(f"No standard target name found; using the last column '{last}' as target (nunique=2).")
        return last
    raise ValueError(f"Target column not found. Tried {candidates}. Please set --target.")

# =========================
# Threshold Utilities
# =========================
def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    j_best_idx = int(np.argmax(j_scores))
    if j_best_idx >= len(thresh):
        j_best_idx = len(thresh) - 1
    return float(thresh[j_best_idx])

def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(int)

# =========================
# Plotting Helpers
# =========================
def plot_roc(y_true, y_prob, title, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_pr(y_true, y_prob, title, save_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_confusion(cm: np.ndarray, title: str, save_path: Path, labels=("Negative","Positive")):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

# =========================
# Modeling Routines
# =========================
def evaluate_model(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = youden_optimal_threshold(y_true, y_prob)
    y_pred = apply_threshold(y_prob, threshold)
    metrics = {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return metrics, cm

def logistic_pipeline():
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    param_grid = {
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear", "lbfgs"],
    }
    return pipe, param_grid

def xgb_pipeline():
    if not HAS_XGB:
        return None, None
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist"
        ))
    ])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__scale_pos_weight": [1.0, 3.0, 5.0],
        "clf__reg_lambda": [1.0, 5.0, 10.0],
    }
    return pipe, param_grid

def fit_with_cv_grid(estimator, param_grid, X_train, y_train, scoring="roc_auc", n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    return grid, best

# =========================
# LSTM/Sequential Modeling Stubs
# =========================
def prepare_sequences(df: pd.DataFrame, id_col: str, time_col: str, feature_cols: list[str], target_col: str, timesteps: int):
    """
    Prepare 3D tensors for sequence models (to be implemented when long-format TS is ready).
    """
    raise NotImplementedError("Implement windowing/padding to create (n, T, d) tensors.")

def build_lstm_model(input_shape, units=64, dropout=0.2):
    """
    Example Keras LSTM architecture (placeholder).
    """
    raise NotImplementedError("Add Keras model once sequence tensors are available.")

# =========================
# Main
# =========================
def run(args):
    base_dir = Path(__file__).resolve().parent
    output_dir, fig_dir = setup_paths(base_dir)

    # Load data (robust to UTF-8 filename)
    data_path = Path(args.data) if args.data else (base_dir / DEFAULT_DATA)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    # read with explicit encoding fallback
    try:
        df = pd.read_csv(data_path)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="utf-8-sig")

    # Determine target column
    target_col = args.target if args.target else detect_target_column(df)

    # If target is non-numeric (e.g., 'Yes'/'No'), map to 0/1 with --positive-label control
    y_raw = df[target_col]
    if y_raw.dtype.kind not in "biufc":
        # categorical to numeric
        if args.positive_label is None:
            # try common positive tokens
            pos_tokens = {"yes","y","true","t","death","dead","deceased","1","positive","pos"}
            y = y_raw.astype(str).str.lower().isin(pos_tokens).astype(int).values
            warnings.warn("Non-numeric target mapped via common tokens. Use --positive-label to control mapping.")
        else:
            y = (y_raw.astype(str) == str(args.positive_label)).astype(int).values
    else:
        y = y_raw.astype(float)
        # If values are not 0/1, binarize treating the larger value as positive
        uniq = np.unique(y[~np.isnan(y)])
        if len(uniq) == 2 and set(uniq) != {0.0, 1.0}:
            # map min->0, max->1
            mapping = {float(min(uniq)):0, float(max(uniq)):1}
            y = np.vectorize(mapping.get)(y).astype(int)
        else:
            y = (y > 0.5).astype(int)

    # Drop non-feature columns
    drop_cols = set([target_col]) | set(args.drop_cols or [])
    for candidate in ["patient_id", "stay_id", "hadm_id", "subject_id", "icustay_id", "timestamp", "charttime", "time"]:
        if candidate in df.columns:
            drop_cols.add(candidate)

    feature_names = [c for c in df.columns if c not in drop_cols]
    X = df[feature_names].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    artifacts = []

    # Logistic Regression
    logi_pipe, logi_grid = logistic_pipeline()
    logi_gridcv, logi_best = fit_with_cv_grid(logi_pipe, logi_grid, X_train, y_train, scoring="roc_auc", n_splits=args.cv_splits)

    y_prob_logi = logi_best.predict_proba(X_test)[:, 1]
    threshold_logi = args.threshold if args.threshold is not None else youden_optimal_threshold(y_test, y_prob_logi)
    metrics_logi, cm_logi = evaluate_model(y_test, y_prob_logi, threshold=threshold_logi)

    roc_path_logi = fig_dir / "roc_logistic.png"
    pr_path_logi = fig_dir / "pr_logistic.png"
    cm_path_logi = fig_dir / "cm_logistic.png"
    plot_roc(y_test, y_prob_logi, "ROC - Logistic Regression", roc_path_logi)
    plot_pr(y_test, y_prob_logi, "PR Curve - Logistic Regression", pr_path_logi)
    plot_confusion(cm_logi, f"Confusion Matrix - Logistic (thr={metrics_logi['threshold']:.3f})", cm_path_logi)

    results.append({
        "model": "LogisticRegression",
        **metrics_logi,
        "best_params": json.dumps(logi_gridcv.best_params_),
        "roc_path": str(roc_path_logi),
        "pr_path": str(pr_path_logi),
        "cm_path": str(cm_logi),
    })
    artifacts += [roc_path_logi, pr_path_logi, cm_path_logi]

    # XGBoost
    if HAS_XGB:
        xgb_pipe, xgb_grid = xgb_pipeline()
        xgb_gridcv, xgb_best = fit_with_cv_grid(xgb_pipe, xgb_grid, X_train, y_train, scoring="roc_auc", n_splits=args.cv_splits)

        y_prob_xgb = xgb_best.predict_proba(X_test)[:, 1]
        threshold_xgb = args.threshold if args.threshold is not None else youden_optimal_threshold(y_test, y_prob_xgb)
        metrics_xgb, cm_xgb = evaluate_model(y_test, y_prob_xgb, threshold=threshold_xgb)

        roc_path_xgb = fig_dir / "roc_xgboost.png"
        pr_path_xgb = fig_dir / "pr_xgboost.png"
        cm_path_xgb = fig_dir / "cm_xgboost.png"
        plot_roc(y_test, y_prob_xgb, "ROC - XGBoost", roc_path_xgb)
        plot_pr(y_test, y_prob_xgb, "PR Curve - XGBoost", pr_path_xgb)
        plot_confusion(cm_xgb, f"Confusion Matrix - XGBoost (thr={metrics_xgb['threshold']:.3f})", cm_path_xgb)

        # Feature importance
        fi_path_str = ""
        try:
            fitted_xgb = xgb_best.named_steps["clf"]
            importances = fitted_xgb.feature_importances_
            if len(importances) == len(feature_names):
                imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
                imp_path = fig_dir / "feature_importance_xgb_top30.png"
                plt.figure(figsize=(8, 10))
                plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                plt.title("XGBoost Feature Importance (Top 30)")
                plt.tight_layout()
                plt.savefig(imp_path, dpi=160)
                plt.close()
                artifacts.append(imp_path)
                fi_path_str = str(imp_path)
        except Exception as e:
            warnings.warn(f"Failed to plot XGBoost feature importance: {e}")

        results.append({
            "model": "XGBoost",
            **metrics_xgb,
            "best_params": json.dumps(xgb_gridcv.best_params_),
            "roc_path": str(roc_path_xgb),
            "pr_path": str(pr_path_xgb),
            "cm_path": str(cm_path_xgb),
            "feature_importance_path": fi_path_str,
        })

    # Save results
    results_df = pd.DataFrame(results)
    metrics_csv = output_dir / "metrics_summary.csv"
    results_df.to_csv(metrics_csv, index=False)

    print("\n=== Metrics Summary ===")
    print(results_df[["model", "auroc", "auprc", "precision_pos", "recall_pos", "f1_pos", "threshold"]])
    print(f"\nSaved metrics table to: {metrics_csv}")
    print(f"Figures saved to: {fig_dir}")

    manifest = {
        "data_path": str(data_path),
        "target_col": target_col,
        "n_samples": int(df.shape[0]),
        "n_features": int(len(feature_names)),
        "class_balance": {
            "positive_rate": float(np.mean(y)),
            "negative_rate": float(1.0 - np.mean(y)),
        },
        "artifacts": [str(p) for p in artifacts],
        "metrics_csv": str(metrics_csv),
        "notes": "Thresholds based on Youden-J unless overridden by --threshold.",
        "dataset_default": DEFAULT_DATA
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest to: {manifest_path}")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ICU Mortality Modeling Pipeline (Enhanced) - default dataset swapped")
    parser.add_argument("--data", type=str, default=None, help=f"Path to CSV data (default: ./{DEFAULT_DATA})")
    parser.add_argument("--target", type=str, default=None, help="Target column name. If omitted, auto-detect from common names or last binary column.")
    parser.add_argument("--positive-label", type=str, default=None, help="When target is non-numeric, which value counts as positive (e.g., 'Yes' or 'Death').")
    parser.add_argument("--drop-cols", type=str, nargs="*", default=None, help="Extra columns to drop from features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size proportion (default 0.2).")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV folds (default 5).")
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed decision threshold (e.g., 0.121). If omitted, use Youden-J.")
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

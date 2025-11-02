# -*- coding: utf-8 -*-
"""
mortality_revised.py
Train/evaluate Logistic Regression & XGBoost on windowed ICU features.

Usage:
  python3 mortality_revised.py --data processed_icu_data_causal.csv \
    --target "In-hospital_death" --id_cols RecordID window_start window_end

Outputs:
  - Console metrics (Accuracy, AUROC, AUPRC, Precision, Recall, F1, Brier)
  - figures/confusion_matrix_logit.png
  - figures/confusion_matrix_xgb.png
  - test_predictions.csv
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, brier_score_loss
)

RANDOM_STATE = 42

def load_data(path, target, id_cols):
    df = pd.read_csv(path)
    print(f"✅ Loaded: {path} | shape={df.shape}")

    # remove obvious non-feature columns
    drop_cols = set(id_cols + [target])
    # keep only numeric features
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].copy()
    y = df[target].astype(int).values

    # Basic sanity
    if X.isnull().all(axis=1).any():
        # Replace rows that are all-NaN with zeros (rare)
        X = X.fillna(0)
    else:
        X = X.fillna(X.median(numeric_only=True))

    return df, X, y, feature_cols

def report_class_balance(split_name, y):
    n = len(y)
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    ratio = f"{neg}:{pos}" if pos > 0 else "inf"
    print(f"🔢 Class balance [{split_name}]: neg={neg}, pos={pos} (ratio {ratio}, pos_rate={pos/n:.3f})")

def pick_best_threshold(y_true, y_prob, beta=1.0):
    """Search threshold maximizing F-score on validation set."""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0, beta=beta)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)

def metrics_suite(model_name, y_true, y_prob, y_pred, out_prefix):
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan
    auprc = average_precision_score(y_true, y_prob)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    print(f"[{model_name}] Accuracy={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} "
          f"| Precision={p:.4f} | Recall={r:.4f} | F1={f1:.4f} | Brier={brier:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), va="center", ha="center", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    png_path = fig_dir / f"confusion_matrix_{out_prefix}.png"
    plt.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"🖼️  Saved confusion matrix → {png_path}")

    return {
        "accuracy": acc, "auroc": float(auroc), "auprc": float(auprc),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "brier": float(brier), "cm": cm.tolist()
    }

def main(args):
    # ---- load
    id_cols = args.id_cols or []
    df, X, y, feature_cols = load_data(args.data, args.target, id_cols)
    print(f"🧠 Using {len(feature_cols)} numeric features")

    # ---- split (train/val/test = 70/10/20)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(2/3), random_state=RANDOM_STATE, stratify=y_tmp
    )
    report_class_balance("train", y_train)
    report_class_balance("valid", y_val)
    report_class_balance("test",  y_test)

    results = {}

    # =========================
    # Logistic Regression
    # =========================
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),   # sparse-safe
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
            solver="lbfgs", n_jobs=None
        ))
    ])
    logit.fit(X_train, y_train)
    val_prob = logit.predict_proba(X_val)[:,1]
    t_logit = pick_best_threshold(y_val, val_prob, beta=1.0)
    print(f"🔎 [Logit] chosen threshold (F1@val) = {t_logit:.3f}")

    test_prob = logit.predict_proba(X_test)[:,1]
    test_pred = (test_prob >= t_logit).astype(int)
    results["logit"] = metrics_suite("Logit", y_test, test_prob, test_pred, "logit")

    # =========================
    # XGBoost
    # =========================
    try:
        import xgboost as xgb
    except Exception as e:
        print("❌ xgboost is not installed. Please `pip install xgboost`.")
        sys.exit(1)

    # scale_pos_weight = neg/pos on TRAIN
    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))
    spw = neg / pos
    # ⚙️ Compute class imbalance weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"🎄 XGBoost scale_pos_weight (neg/pos on train) = {scale_pos_weight:.2f}")

    # ✅ Initialize and train XGBoost
    xgb_clf = XGBClassifier(
        eval_metric='logloss',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )

    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_prob_xgb = xgb_clf.predict_proba(X_val)[:,1]
    t_xgb = pick_best_threshold(y_val, val_prob_xgb, beta=1.0)
    print(f"🔎 [XGBoost] chosen threshold (F1@val) = {t_xgb:.3f}")

    test_prob_xgb = xgb_clf.predict_proba(X_test)[:,1]
    test_pred_xgb = (test_prob_xgb >= t_xgb).astype(int)
    results["xgb"] = metrics_suite("XGBoost", y_test, test_prob_xgb, test_pred_xgb, "xgb")

    # ---- save predictions
    out_pred = pd.DataFrame({
        "y_true": y_test,
        "logit_prob": test_prob,
        "logit_pred": test_pred,
        "xgb_prob": test_prob_xgb,
        "xgb_pred": test_pred_xgb
    })
    out_pred.to_csv("test_predictions.csv", index=False)
    print("💾 Saved predictions → test_predictions.csv")

    # ---- save a small JSON summary
    with open("metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📑 Saved metrics summary → metrics_summary.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="processed_icu_data_causal.csv")
    parser.add_argument("--target", default="In-hospital_death")
    parser.add_argument("--id_cols", nargs="*", default=["RecordID","window_start","window_end"],
                        help="ID/time-ish columns to drop from features if present")
    args = parser.parse_args()
    main(args)

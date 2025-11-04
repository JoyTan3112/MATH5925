#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mortality_revised.py
- Logical de-duplication by patient/admission ID (keep last window)
- 80/10/10 stratified split (test ≈ 10%)
- 10-fold CV for Logistic Regression & XGBoost
- Full metrics on held-out test set, confusion matrix figures, predictions CSV, metrics JSON
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# xgboost (sklearn API)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- Utility: safe printing ----------
def hr(title: str):
    print("\n" + " " + title)
    print("-" * (len(title) + 1))

# ---------- Find columns ----------
ID_CANDIDATES = ["icustay_id", "subject_id", "hadm_id", "RecordID", "patient_id"]
TIME_CANDIDATES = ["event_time", "window_end_time", "end_time", "charttime", "timestamp"]
TARGET_CANDIDATES = ["In-hospital_death", "In_hospital_death", "hospital_death", "mortality", "label", "y"]

def find_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# ---------- Metrics ----------
def full_metrics(y_true, y_prob, threshold=0.5, prefix=""):
    y_hat = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_hat)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")

    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    print(f"[{prefix}] Accuracy={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
          f"Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | Brier={brier:.4f}")
    print(classification_report(y_true, y_hat, digits=4))

    return {
        "accuracy": acc, "auroc": auroc, "auprc": auprc,
        "precision": prec, "recall": rec, "f1": f1, "brier": brier,
        "threshold": threshold
    }

def youden_threshold(y_true, y_prob):
    """Choose threshold on validation set via Youden J (TPR - FPR)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thr[np.argmax(j)]

def save_confusion(y_true, y_prob, threshold, out_path):
    y_hat = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

# ---------- Main ----------
def main(args):
    data_path = Path(args.data)
    assert data_path.exists(), f"File not found: {data_path}"

    df = pd.read_csv(data_path)
    print(f"✅ Loaded: {data_path.name} | shape={df.shape}")

    cols = df.columns.tolist()
    ID_COL = find_first(cols, ID_CANDIDATES)
    TIME_COL = find_first(cols, TIME_CANDIDATES)
    TARGET_COL = find_first(cols, TARGET_CANDIDATES)
    assert TARGET_COL is not None, f"Target column not found; tried: {TARGET_CANDIDATES}"

    # ---------- logical de-dup ----------
    if ID_COL is not None:
        before = df.shape[0]
        if TIME_COL is not None:
            df = df.sort_values([ID_COL, TIME_COL]).groupby(ID_COL, as_index=False, sort=False).tail(1)
        else:
            df = df.drop_duplicates(subset=[ID_COL], keep="last")
        after = df.shape[0]
        print(f"🔎 逻辑去重：按 {ID_COL} 聚合  {before} → {after} 行；唯一ID数：{df[ID_COL].nunique()}")
    else:
        print("⚠️ 未找到可用的ID列（icustay_id/subject_id/hadm_id/RecordID），跳过逻辑去重。")

    # ---------- select numeric features ----------
    ignore_cols = set([TARGET_COL])
    if ID_COL: ignore_cols.add(ID_COL)
    if TIME_COL: ignore_cols.add(TIME_COL)

    num_cols = [c for c in df.columns
                if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    assert len(num_cols) > 0, "No numeric features found!"
    print(f"🧠 Using {len(num_cols)} numeric features")

    X = df[num_cols].values
    y = df[TARGET_COL].values.astype(int)

    # ---------- stratified 80/10/10 ----------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=1/9, random_state=RANDOM_STATE, stratify=y_trainval
    )

    def cnt(yv): return (np.sum(yv==0), np.sum(yv==1))
    n0,n1 = cnt(y_train); print(f"🔢 Class balance [train]: neg={n0}, pos={n1} (ratio {n0}:{n1}, pos_rate={n1/(n0+n1):.3f})")
    n0,n1 = cnt(y_val);   print(f"🔢 Class balance [valid]: neg={n0}, pos={n1} (ratio {n0}:{n1}, pos_rate={n1/(n0+n1):.3f})")
    n0,n1 = cnt(y_test);  print(f"🔢 Class balance [test ]: neg={n0}, pos={n1} (ratio {n0}:{n1}, pos_rate={n1/(n0+n1):.3f})")

    # ---------- pipelines ----------
    # Logistic Regression: impute + scale + L2
    logit = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=200, class_weight=None, random_state=RANDOM_STATE))
    ])

    # XGBoost (sklearn API). 不传 early_stopping 避免版本差异；先把 CV 做好
    neg, pos = np.sum(y_train==0), np.sum(y_train==1)
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
    print(f"🌲 XGBoost scale_pos_weight (neg/pos on train) = {scale_pos_weight:.2f}")
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # ---------- 10-fold CV ----------
    hr("10-fold Cross-Validation on TRAIN")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_acc_log = cross_val_score(logit, X_train, y_train, cv=skf, scoring="accuracy")
    cv_auc_log = cross_val_score(logit, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"[Logit] 10-fold CV Accuracy: mean={cv_acc_log.mean():.4f} ± {cv_acc_log.std():.4f}")
    print(f"[Logit] 10-fold CV AUROC:    mean={cv_auc_log.mean():.4f} ± {cv_auc_log.std():.4f}")

    cv_acc_xgb = cross_val_score(xgb, X_train, y_train, cv=skf, scoring="accuracy")
    cv_auc_xgb = cross_val_score(xgb, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"[XGB ] 10-fold CV Accuracy: mean={cv_acc_xgb.mean():.4f} ± {cv_acc_xgb.std():.4f}")
    print(f"[XGB ] 10-fold CV AUROC:    mean={cv_auc_xgb.mean():.4f} ± {cv_auc_xgb.std():.4f}")

    # ---------- Fit on TRAIN, choose threshold on VAL, evaluate on TEST ----------
    out_dir = Path("figures"); out_dir.mkdir(exist_ok=True, parents=True)

    # Logistic
    hr("Logistic Regression (Val threshold -> Test metrics)")
    logit.fit(X_train, y_train)
    val_prob_log = logit.predict_proba(X_val)[:, 1]
    th_log = youden_threshold(y_val, val_prob_log)
    print(f"[Logit] chosen threshold (F1/Youden on VAL) = {th_log:.3f}")

    test_prob_log = logit.predict_proba(X_test)[:, 1]
    m_log = full_metrics(y_test, test_prob_log, threshold=th_log, prefix="Logit")
    save_confusion(y_test, test_prob_log, th_log, out_dir / "confusion_matrix_logit.png")

    # XGBoost
    hr("XGBoost (Val threshold -> Test metrics)")
    xgb.fit(X_train, y_train, verbose=False)
    val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
    th_xgb = youden_threshold(y_val, val_prob_xgb)
    print(f"[XGBoost] chosen threshold (F1@VAL) = {th_xgb:.3f}")

    test_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    m_xgb = full_metrics(y_test, test_prob_xgb, threshold=th_xgb, prefix="XGBoost")
    save_confusion(y_test, test_prob_xgb, th_xgb, out_dir / "confusion_matrix_xgb.png")

    # ---------- Save predictions & metrics ----------
    pred_df = pd.DataFrame({
        "y_test": y_test,
        "logit_prob": test_prob_log,
        "xgb_prob": test_prob_xgb
    })
    pred_df.to_csv("test_predictions.csv", index=False)
    print("💾 Saved predictions → test_predictions.csv")

    metrics = {
        "cv": {
            "logit": {"acc_mean": float(cv_acc_log.mean()), "acc_std": float(cv_acc_log.std()),
                      "auc_mean": float(cv_auc_log.mean()), "auc_std": float(cv_auc_log.std())},
            "xgb":   {"acc_mean": float(cv_acc_xgb.mean()), "acc_std": float(cv_acc_xgb.std()),
                      "auc_mean": float(cv_auc_xgb.mean()), "auc_std": float(cv_auc_xgb.std())},
        },
        "test": {"logit": m_log, "xgb": m_xgb},
        "meta": {
            "n_total": int(df.shape[0]),
            "n_features": int(len(num_cols)),
            "target": TARGET_COL,
            "id_col": ID_COL,
            "time_col": TIME_COL
        }
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("💾 Saved metrics summary → metrics_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    args = parser.parse_args()
    main(args)

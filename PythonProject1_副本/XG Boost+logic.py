# -*- coding: utf-8 -*-
"""
ICU Mortality Prediction — Logistic Regression & (optional) XGBoost
pip install -U pandas numpy scikit-learn matplotlib seaborn xgboost
"""


from pathlib import Path
import argparse, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[Note] xgboost 未安装，可通过 `pip install xgboost` 安装。")

# -------------------- 路径与参数 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH        = SCRIPT_DIR / "processed_icu_data.csv"
UNIVARIATE_PATH  = SCRIPT_DIR / "univariate_analysis_results.csv"
CORRELATION_PATH = SCRIPT_DIR / "correlation_analysis_results.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default=None,
                    help="标签列名（如 In-hospital_death / hospital_expire_flag / outcome 等）")
args, _ = parser.parse_known_args()

print("当前运行目录:", Path.cwd())
print("脚本所在目录:", SCRIPT_DIR)
print("数据文件是否存在:", DATA_PATH.exists())

# -------------------- 读数据 --------------------
df = pd.read_csv(DATA_PATH)
print(f"✅ 成功读取数据！Shape: {df.shape}")
print("前20个列名：", list(df.columns)[:20])

# -------------------- 目标列选择与标准化 --------------------
def normalize_label_series(s: pd.Series) -> pd.Series:
    """将各种写法统一成0/1"""
    if s.dtype.kind in "biu":  # 已经是数值型
        return s.astype(int)
    s_str = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    pos = r"^(1|true|yes|y|dead|deceased|expired?|死亡)$"
    neg = r"^(0|false|no|n|alive|surviv(e|ed)|discharged?|存活)$"
    out[s_str.str.match(pos, na=False)] = 1
    out[s_str.str.match(neg, na=False)] = 0
    if out.isna().any():
        with np.errstate(all="ignore"):
            num = pd.to_numeric(s, errors="coerce")
        out = out.fillna(num)
    return out.astype(float)

def pick_target_column(df: pd.DataFrame, prefer: str | None = None) -> str | None:
    # 允许优先手动指定
    if prefer and prefer in df.columns:
        return prefer
    # 常见名字（注意含连字符的 In-hospital_death）
    candidates = [
        "in-hospital_death", "in_hospital_death", "inhospital_death",
        "hospital_expire_flag", "in_hospital_mortality", "icu_mortality",
        "mortality", "death", "deceased", "expire_flag", "outcome", "label", "y", "target"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    # 模糊匹配
    patt = re.compile(r"(mort(al(ity)?)?|expir(e|ed|y)|death|deceas)", re.I)
    for c in df.columns:
        if patt.search(c):
            return c
    # 二值列兜底
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 3:
            s = normalize_label_series(df[c])
            vals = set(pd.unique(s.dropna()))
            if vals <= {0.0, 1.0}:
                return c
    return None

target_col = pick_target_column(df, prefer=args.target)
if target_col is None:
    print("\n❌ 没找到标签列。可用 `--target 列名` 指定。")
    print("全部列名：", list(df.columns))
    raise SystemExit(1)

print(f"🎯 使用目标列: {target_col}")

y = normalize_label_series(df[target_col]).round().astype(int).values
ID_COL = "RecordID" if "RecordID" in df.columns else None
drop_cols = [c for c in [target_col, ID_COL] if c in df.columns]
X = df.drop(columns=drop_cols)
print(f"特征维度: {X.shape},  剔除列: {drop_cols}")

# -------------------- EDA（可选） --------------------
print("\n📊 缺失率（前10个特征）：")
print(X.isna().mean().sort_values(ascending=False).head(10))

plt.figure()
sns.countplot(x=y)
plt.title("Target 分布")
plt.show()

if CORRELATION_PATH.exists():
    corr_df = pd.read_csv(CORRELATION_PATH)
    if {"feature1","feature2","corr"} <= set(map(str.lower, corr_df.columns)):
        pivot = corr_df.pivot(index=corr_df.columns[0], columns=corr_df.columns[1], values=corr_df.columns[2])
        plt.figure(figsize=(10,8))
        sns.heatmap(pivot, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (provided)")
        plt.show()
else:
    num_X = X.select_dtypes(include=[np.number]).iloc[:, :50]
    if num_X.shape[1] >= 2:
        plt.figure(figsize=(10,8))
        sns.heatmap(num_X.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (auto subset)")
        plt.show()

# -------------------- 划分数据 --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain:", X_train.shape, " Test:", X_test.shape)

# -------------------- 预处理 --------------------
num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
cat_cols = [c for c in X_train.columns if c not in num_cols]

num_pipe_logit = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
pre_logit = ColumnTransformer([
    ("num", num_pipe_logit, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# -------------------- Logistic --------------------
logit = Pipeline([
    ("pre", pre_logit),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
])
logit.fit(X_train, y_train)
proba_logit = logit.predict_proba(X_test)[:, 1]
print(f"\n[Logit] AUROC={roc_auc_score(y_test, proba_logit):.4f} "
      f"| AUPRC={average_precision_score(y_test, proba_logit):.4f} "
      f"| Brier={brier_score_loss(y_test, proba_logit):.4f}")
print(classification_report(y_test, (proba_logit>=0.5).astype(int), digits=4))
sns.heatmap(confusion_matrix(y_test, (proba_logit>=0.5).astype(int)), annot=True, fmt="d", cbar=False)
plt.title("Logistic Confusion Matrix (0.5)")
plt.show()
CalibrationDisplay.from_predictions(y_test, proba_logit, n_bins=10)
plt.title("Logistic Calibration")
plt.show()

# -------------------- XGBoost（可选） --------------------
if HAS_XGB:
    pre_xgb = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    xgb = Pipeline([
        ("pre", pre_xgb),
        ("clf", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42,
            eval_metric="logloss", tree_method="hist"
        ))
    ])
    xgb.fit(X_train, y_train)
    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    print(f"\n[XGB]  AUROC={roc_auc_score(y_test, proba_xgb):.4f} "
          f"| AUPRC={average_precision_score(y_test, proba_xgb):.4f} "
          f"| Brier={brier_score_loss(y_test, proba_xgb):.4f}")
    sns.heatmap(confusion_matrix(y_test, (proba_xgb>=0.5).astype(int)), annot=True, fmt="d", cbar=False)
    plt.title("XGBoost Confusion Matrix (0.5)")
    plt.show()
    CalibrationDisplay.from_predictions(y_test, proba_xgb, n_bins=10)
    plt.title("XGBoost Calibration")
    plt.show()

# -------------------- 导出预测 --------------------
out = pd.DataFrame({"y_true": y_test, "proba_logit": proba_logit})
try:
    out["proba_xgb"] = proba_xgb  # 若未训练XGB会跳过
except Exception:
    pass
out.to_csv(SCRIPT_DIR / "test_predictions.csv", index=False)
print("\n✅ 已生成 test_predictions.csv")

from sklearn.metrics import accuracy_score

# Logistic Regression 的准确率
logit_preds = (proba_logit >= 0.5).astype(int)
acc_logit = accuracy_score(y_test, logit_preds)
print(f"🎯 Logistic Regression 准确率 (Accuracy): {acc_logit:.4f}")

# 若训练了 XGBoost，也计算它的准确率
if HAS_XGB:
    xgb_preds = (proba_xgb >= 0.5).astype(int)
    acc_xgb = accuracy_score(y_test, xgb_preds)
    print(f"🌲 XGBoost 准确率 (Accuracy): {acc_xgb:.4f}")

plt.figure()
models = ["Logistic", "XGBoost"]
accs = [acc_logit, acc_xgb if HAS_XGB else np.nan]
plt.bar(models, accs, color=["#1f77b4", "#2ca02c"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("模型准确率对比")
plt.show()

# -*- coding: utf-8 -*-
"""
window-trendfeatures.py
Convert raw ICU data into windowed trend features for modeling.
Usage:
  python3 window-trendfeatures.py \
    --data raw_icu_data.csv \
    --id_col RecordID \
    --time_col event_time \
    --target_col In-hospital_death \
    --resample 1H \
    --window_hours 6 \
    --stride_hours 1 \
    --max_hours 48 \
    --out processed_icu_data_causal.csv
"""
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_trend(x):
    """Compute linear trend slope"""
    if len(x.dropna()) < 2:
        return np.nan
    try:
        t = np.arange(len(x))
        coef = np.polyfit(t, x.fillna(method="ffill"), 1)
        return coef[0]
    except Exception:
        return np.nan

def make_window_features(df, id_col, time_col, target_col, resample="1H",
                         window_hours=6, stride_hours=1, max_hours=48):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col]).sort_values([id_col, time_col])

    # 先得到“病人级标签”：是否曾出现过1
    if target_col in df.columns:
        pid_y = (pd.to_numeric(df[target_col], errors="coerce")
                   .fillna(0).groupby(df[id_col]).max().astype(int))
    else:
        pid_y = pd.Series(dtype=int)

    feats = []

    for pid, g in df.groupby(id_col):
        y_pid = int(pid_y.get(pid, 0))  # 该病人的最终标签

        g = g.set_index(time_col).sort_index()
        # 仅用数值列做时间聚合
        data_cols = g.drop(columns=[target_col], errors="ignore")
        numeric_cols = data_cols.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            continue
        g_num = (data_cols[numeric_cols]
                 .resample(resample).mean()
                 .interpolate(limit_direction="forward"))

        if g_num.empty:
            continue

        start = g_num.index.min()
        cutoff = start + pd.Timedelta(hours=max_hours)
        g_num = g_num.loc[g_num.index <= cutoff]
        if g_num.empty:
            continue

        end_max = g_num.index.max() - pd.Timedelta(hours=window_hours)
        if end_max < g_num.index.min():
            continue

        for t0 in pd.date_range(start, end_max, freq=f"{stride_hours}H"):
            t1 = t0 + pd.Timedelta(hours=window_hours)
            win = g_num.loc[(g_num.index >= t0) & (g_num.index < t1)]
            if win.empty:
                continue

            w = {id_col: pid, "window_start": t0, "window_end": t1}
            for col in g_num.columns:
                s = win[col]
                w[f"{col}_mean"]   = s.mean()
                w[f"{col}_max"]    = s.max()
                w[f"{col}_min"]    = s.min()
                w[f"{col}_median"] = s.median()
                w[f"{col}_count"]  = s.count()
                if s.notna().sum() >= 2:
                    t = np.arange(len(s))
                    try:
                        w[f"{col}_trend"] = np.polyfit(t, s.ffill(), 1)[0]
                    except Exception:
                        w[f"{col}_trend"] = np.nan
                else:
                    w[f"{col}_trend"] = np.nan

            # 关键：用病人级标签赋给该病人的所有窗口
            w[target_col] = y_pid
            feats.append(w)

    return pd.DataFrame(feats)


def main(args):
    print(f"📂 Loading {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"✅ Loaded shape: {df.shape}")

    print("🧩 Building windowed features...")
    df_feat = make_window_features(df,
                                   id_col=args.id_col,
                                   time_col=args.time_col,
                                   target_col=args.target_col,
                                   resample=args.resample,
                                   window_hours=args.window_hours,
                                   stride_hours=args.stride_hours,
                                   max_hours=args.max_hours)
    print(f"✅ Feature shape: {df_feat.shape}")
    df_feat.to_csv(args.out, index=False)
    print(f"💾 Saved to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--time_col", required=True)
    ap.add_argument("--target_col", required=True)
    ap.add_argument("--resample", default="1H")
    ap.add_argument("--window_hours", type=int, default=6)
    ap.add_argument("--stride_hours", type=int, default=1)
    ap.add_argument("--max_hours", type=int, default=48)
    ap.add_argument("--out", default="processed_icu_data_causal.csv")
    args = ap.parse_args()
    main(args)

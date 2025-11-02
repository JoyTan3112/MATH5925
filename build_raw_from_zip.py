# -*- coding: utf-8 -*-
"""
build_raw_from_zip.py  (recursive, txt/csv, auto-sep)
Usage:
  python3 build_raw_from_zip.py --zip Features_group_10.zip --out raw_icu_data.csv
"""
import re, os, io, zipfile, argparse
import pandas as pd
import numpy as np

ID_CANDS = ["recordid","subject_id","hadm_id","icustay_id","patient_id","stay_id","id"]
TIME_CANDS = ["event_time","charttime","time","hours","hour","hours_in","ts","timestamp","datetime"]
TARGET_CANDS = ["in-hospital_death","in_hospital_death","inhospital_death",
                "hospital_expire_flag","icu_mortality","in_hospital_mortality",
                "mortality","death","deceased","expire_flag","outcome","label","y","target"]

def canon(s): return re.sub(r"[^\w]+","_", str(s).strip().lower())
def pick(cols, cands):
    low = {canon(c): c for c in cols}
    for k in cands:
        if k in low: return low[k]
    for c in cols:
        if any(k in canon(c) for k in cands): return c
    return None

def normalize_y(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biu": return s.fillna(0).astype(int)
    ss = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    pos = r"^(1|true|yes|y|dead|deceased|expired?)$"
    neg = r"^(0|false|no|n|alive|surviv(e|ed)|discharged?)$"
    out[ss.str.match(pos, na=False)] = 1
    out[ss.str.match(neg, na=False)] = 0
    num = pd.to_numeric(s, errors="coerce")
    out = out.fillna(num)
    return out.fillna(0).round().astype(int)

def read_table_guess(fobj):
    """Robust loader: try sep inference & encodings."""
    # 1) let pandas infer sep with python engine
    try:
        return pd.read_csv(fobj, sep=None, engine="python")
    except Exception:
        pass
    # retry with common seps & encodings
    seps = [",", "\t", ";", "|"]
    encs = [None, "utf-8", "latin-1"]
    for enc in encs:
        for sep in seps:
            try:
                fobj.seek(0)
                return pd.read_csv(fobj, sep=sep, encoding=enc)
            except Exception:
                continue
    # last resort: read as whitespace
    try:
        fobj.seek(0)
        return pd.read_csv(fobj, delim_whitespace=True)
    except Exception:
        fobj.seek(0)
        return pd.DataFrame()

def main(zip_path: str, out_csv: str):
    assert os.path.exists(zip_path), f"ZIP not found: {zip_path}"
    frames = []
    n_files = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if not name.lower().endswith((".csv",".txt")):
                continue
            n_files += 1
            with zf.open(info) as raw:
                # load into BytesIO so we can seek
                bio = io.BytesIO(raw.read())
                df = read_table_guess(bio)
            if df.empty:
                continue

            cols = list(df.columns)
            id_col = pick(cols, ID_CANDS) or "_file_id"
            time_col = pick(cols, TIME_CANDS) or "_row_idx_time"
            tgt_col = pick(cols, TARGET_CANDS) or "_target_tmp"

            if id_col == "_file_id":
                df["_file_id"] = os.path.splitext(os.path.basename(name))[0]
            if time_col == "_row_idx_time":
                df["_row_idx_time"] = np.arange(len(df), dtype=float)
            if tgt_col == "_target_tmp":
                df["_target_tmp"] = np.nan

            # parse time as datetime if possible, else numeric
            try:
                parsed = pd.to_datetime(df[time_col], errors="coerce")
                if parsed.notna().sum() >= max(5, int(0.1*len(parsed))):
                    df[time_col] = parsed
                else:
                    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            except Exception:
                df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

            df = df.rename(columns={
                id_col: "RecordID",
                time_col: "event_time",
                tgt_col: "In-hospital_death"
            })
            for req in ["RecordID","event_time","In-hospital_death"]:
                if req not in df.columns: df[req] = np.nan

            frames.append(df)

    assert n_files > 0, "No .csv/.txt found inside ZIP (including subfolders)."
    assert frames, "Readable tables not found—check file encodings/separators."

    out = pd.concat(frames, ignore_index=True)
    out["In-hospital_death"] = normalize_y(out["In-hospital_death"])

    # event_time normalization
    if not np.issubdtype(out["event_time"].dtype, np.datetime64):
        parsed = pd.to_datetime(out["event_time"], errors="coerce")
        if parsed.notna().sum() >= max(50, int(0.05*len(parsed))):
            out["event_time"] = parsed
        else:
            out["event_time"] = pd.to_numeric(out["event_time"], errors="coerce")

    out = out.sort_values(["RecordID","event_time"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv} | shape={out.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ZIP with CSV/TXT (nested OK)")
    ap.add_argument("--out", default="raw_icu_data.csv", help="Output CSV path")
    args = ap.parse_args()
    main(args.zip, args.out)


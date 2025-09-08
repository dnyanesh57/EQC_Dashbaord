# Digital EQC Dashboard — Streamlit (v2.19, SJCPL theme)
# - Uses only SJCPL brand colors (Blue/Grey/Black/White) & Roboto font
# - Replaces all non-brand palettes with SJCPL discrete/continuous scales
# - Visual-only; your data logic, timestamp rules & TAT computations untouched
# - Author:ad

import io
import re
import itertools  # <<< added
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative as qual  # kept just in case
import streamlit as st

# <<< added
_plot_counter = itertools.count()
def plot_key(prefix: str) -> str:
    return f"{prefix}_{next(_plot_counter)}"
# >>>

pd.options.mode.copy_on_write = True
st.set_page_config(page_title="DigiQC — EQC Dashboard (SJCPL)", page_icon="🛠️", layout="wide")

# ----------------------------- SJCPL THEME -----------------------------
# Brand palette (as tints/shades where needed)
SJCPL = {
    "BLUE": "#00AEDA",
    "BLUE_700": "#007FA3",  # darker blue
    "BLUE_300": "#33C9EA",
    "BLUE_200": "#66D6F0",
    "GREY": "#939598",
    "GREY_600": "#6F7174",
    "GREY_300": "#B4B6B8",
    "GREY_200": "#C9CACC",
    "BLACK": "#000000",
    "WHITE": "#FFFFFF",
}

# Plotly defaults (brand-only)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    SJCPL["BLUE"], SJCPL["BLUE_700"], SJCPL["BLUE_200"],
    SJCPL["GREY"], SJCPL["GREY_600"], SJCPL["GREY_200"], SJCPL["BLACK"]
]

# Brand continuous scale (Grey -> Blue)
SJCPL_CONT = [
    [0.00, SJCPL["WHITE"]],
    [0.20, SJCPL["GREY_200"]],
    [0.45, SJCPL["GREY"]],
    [0.70, SJCPL["BLUE_200"]],
    [1.00, SJCPL["BLUE"]],
]

# Inject Roboto + brand header
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Roboto', sans-serif; }
:root {
  --sj-blue: %s;
  --sj-blue-dark: %s;
  --sj-grey: %s;
  --sj-grey-200: %s;
  --sj-grey-600: %s;
  --sj-black: %s;
  --sj-white: %s;
}
.sj-header {
  background: linear-gradient(90deg, var(--sj-black) 0%%, var(--sj-blue) 100%%);
  padding: 14px 18px; border-radius: 14px; color: var(--sj-white);
  margin: 6px 0 18px 0;
}
.sj-header h1 { margin: 0; font-weight: 800; letter-spacing: .2px; }
.sj-header p { margin: 4px 0 0 0; opacity: .9; }
</style>
""" % (
    SJCPL["BLUE"], SJCPL["BLUE_700"], SJCPL["GREY"],
    SJCPL["GREY_200"], SJCPL["GREY_600"], SJCPL["BLACK"], SJCPL["WHITE"]
), unsafe_allow_html=True)

# ----------------------------- Constants -----------------------------
STATUSES_STAGE = ["in_process", "redo", "rejected", "approved"]
STATUS_KEYS = STATUSES_STAGE + ["pass"]

# Table row tints (very light, brand-only)
ROW_COLORS = {
    "approved": "#E8F7FC",     # light tint of brand blue
    "pass":     "#F2FBFE",     # extra light blue
    "in_process":"#F4F4F5",    # light grey
    "redo":     "#F0F0F0",     # light grey
    "rejected": "#ECECED",     # light grey
    "other":    "#FAFAFA",     # near white
}

# Discrete status colors (brand-only)
STATUS_COLOR_MAP = {
    "approved": SJCPL["BLUE_700"],  # darker blue
    "pass": SJCPL["BLUE"],          # primary blue
    "in_process": SJCPL["GREY"],    # mid grey
    "redo": SJCPL["GREY_600"],      # dark grey
    "rejected": SJCPL["BLACK"],     # black
    "other": SJCPL["GREY_200"],     # light grey
}
# Softer variant (still brand tints)
STATUS_COLOR_MAP_PASTEL = {
    "approved": SJCPL["BLUE_300"],
    "pass": SJCPL["BLUE_200"],
    "in_process": SJCPL["GREY_300"],
    "redo": SJCPL["GREY_200"],
    "rejected": SJCPL["GREY_600"],
    "other": SJCPL["GREY"],
}

LOCATION_TOKENS = [
    "location", "tower", "building", "block", "wing", "zone", "level", "floor",
    "flat", "unit", "apartment", "apt", "room", "bay", "grid", "section", "segment",
    "area", "lane", "line", "column", "row", "axis"
]
FLOOR_TOKENS = ["floor", "level", "lvl", "storey", "story"]
LOCAL_PLACE_TOKENS = ["flat", "unit", "apartment", "apt", "room", "bay", "grid", "axis", "line"]

# ----------------------------- Helpers ------------------------------
def normalize_colname(c: str) -> str:
    c = str(c)
    return re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")

def first_present(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        for col in df.columns:
            if col == k:
                return col
    for k in keys:
        for col in df.columns:
            if k in col:
                return col
    return None

def parse_dt(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(pd.NaT, index=getattr(s, "index", None))
    if pd.api.types.is_numeric_dtype(s):
        base = pd.to_datetime("1899-12-30")
        return base + pd.to_timedelta(s, unit="D")
    x = s.astype(str).str.strip()
    x = x.replace({"": np.nan, "-": np.nan, "—": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan})
    dt1 = pd.to_datetime(x, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if dt1.isna().mean() > 0.5:
        dt2 = pd.to_datetime(x, errors="coerce", dayfirst=False, infer_datetime_format=True)
        dt1 = dt1.fillna(dt2)
    mask = dt1.isna()
    if mask.any():
        x2 = x.where(~mask, x.str.replace(r"[^0-9/\-: ]", "", regex=True))
        dt3 = pd.to_datetime(x2, errors="coerce", dayfirst=True, infer_datetime_format=True)
        dt1 = dt1.fillna(dt3)
    return dt1

_ctrl_re = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_multi_space = re.compile(r"\s+")

def norm_text(val) -> str:
    if pd.isna(val): return ""
    s = str(val).replace("\u00A0", " ")
    s = _ctrl_re.sub(" ", s)
    s = _multi_space.sub(" ", s).strip()
    return s

def sanitize_cols_for_plot(df: pd.DataFrame, cols: List[str], max_len: int = 500) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            if pd.api.types.is_categorical_dtype(out[c]) or out[c].dtype == "object":
                out[c] = out[c].astype(str).map(norm_text).str.slice(0, max_len)
    return out

def fill_blanks_for_display(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = x[c].astype(str).map(norm_text)
            x[c] = x[c].replace("", "—").fillna("—")
    return x

def status_normalizer(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "other"
    s = str(x).strip().lower().replace("_", " ")
    mapping = {
        "approved": "approved", "accepted": "approved", "ok": "approved",
        "done": "approved", "complete": "approved", "completed": "approved",
        "closed": "approved",
        "pass": "pass", "final pass": "pass",
        "approval pending": "in_process", "in process": "in_process",
        "inprocess": "in_process", "pending": "in_process", "open": "in_process",
        "review": "in_process", "on hold": "in_process", "hold": "in_process",
        "wip": "in_process", "work in progress": "in_process",
        "rework": "redo", "redo": "redo", "returned": "redo",
        "correction": "redo", "resubmit": "redo",
        "rejected": "rejected", "fail": "rejected", "failed": "rejected",
    }
    if s in mapping: return mapping[s]
    if "pass" in s: return "pass"
    if any(k in s for k in ["approve", "accept", "close", "complete", "done", "ok"]): return "approved"
    if any(k in s for k in ["pend", "process", "open", "review", "hold", "wip"]): return "in_process"
    if any(k in s for k in ["redo", "rework", "return", "resub", "correct"]): return "redo"
    if any(k in s for k in ["reject", "fail"]): return "rejected"
    return "other"

# ----------------------------- Load data ----------------------------
def load_data(file: Optional[io.BytesIO]) -> pd.DataFrame:
    if file is None:
        default_path = "https://raw.githubusercontent.com/dnyanesh57/EQC_Dashbaord/main/data/CSV-REPORT-09-06-2025-06-46-26.csv"
        try:
            df = pd.read_csv(default_path)
        except Exception:
            st.error("No file uploaded and demo CSV not found in working directory.")
            st.stop()
    else:
        name = getattr(file, "name", "uploaded.csv").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file)
        else:
            for enc in [None, "utf-8", "utf-8-sig", "latin-1"]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc)
                    break
                except Exception:
                    continue
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})
    df = df.loc[:, ~pd.Series(df.columns).duplicated().values]
    return df

# -------------- Location detection & features -----------------------
def detect_location_columns(df: pd.DataFrame) -> List[str]:
    loc_cols: List[str] = []
    for col in df.columns:
        c = str(col).lower()
        if any(tok in c for tok in LOCATION_TOKENS):
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                loc_cols.append(col)
    seen, ordered = set(), []
    for c in loc_cols:
        if c not in seen:
            seen.add(c); ordered.append(c)
    priority = ["tower", "building", "block", "wing", "zone", "level", "floor",
                "location_l0", "location_l1", "location_l2", "location_l3", "location"]
    ordered.sort(key=lambda c: (0 if c in priority else 1, c))
    return ordered

def pick_first_col(df: pd.DataFrame, tokens: List[str], fallbacks: List[str] = None) -> Optional[str]:
    fallbacks = fallbacks or []
    for col in df.columns:
        c = str(col).lower()
        if any(tok in c for tok in tokens):
            return col
    for fb in fallbacks:
        if fb in df.columns:
            return fb
    return None

def classify_struct_type(text: str) -> str:
    s = (text or "").lower()
    if "column" in s: return "Column"
    if "beam"   in s: return "Beam"
    if "slab"   in s: return "Slab"
    return "Other"

def build_location_features(df: pd.DataFrame, picked_cols: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    alias_candidates = [
        picked_cols.get("location_l0"), picked_cols.get("location_l1"),
        picked_cols.get("location_l2"), picked_cols.get("location_l3"),
        picked_cols.get("location_variable")
    ]
    alias_candidates = [c for c in alias_candidates if c in df.columns]
    auto_detected = detect_location_columns(df)
    loc_cols: List[str] = []
    for c in [*alias_candidates, *auto_detected]:
        if c and c in df.columns and c not in loc_cols:
            loc_cols.append(c)
    if not loc_cols:
        for fb in ["tower", "project_code", "location"]:
            if fb in df.columns:
                loc_cols = [fb]; break

    tower_col = (picked_cols.get("location_l1") if picked_cols.get("location_l1") in df.columns else
                 pick_first_col(df, ["tower","building","block","wing","zone"]))
    df["tower"] = df[tower_col].astype(str).map(norm_text) if tower_col else ""

    floor_col = pick_first_col(df, FLOOR_TOKENS, fallbacks=["location_l2"])
    place_local_col = (picked_cols.get("location_variable")
                       if picked_cols.get("location_variable") in df.columns
                       else pick_first_col(df, ["flat","unit","apartment","apt","room","bay","grid","axis","line"]))

    df["floor"] = df[floor_col].astype(str).map(norm_text) if floor_col else ""
    df["place_local"] = df[place_local_col].astype(str).map(norm_text) if place_local_col else ""

    def _path(row) -> str:
        parts = []
        for c in loc_cols:
            v = row.get(c, None)
            if pd.isna(v): continue
            vs = norm_text(v)
            if vs and vs not in parts:
                parts.append(vs)
        for c in ["tower", "floor", "place_local"]:
            v = row.get(c, None)
            if pd.isna(v): continue
            vs = norm_text(v)
            if vs and vs not in parts:
                parts.append(vs)
        return " / ".join(parts) if parts else ""
    df["location_path"] = df.apply(_path, axis=1)

    act_col = "activity" if "activity" in df.columns else None
    stage_col = "stage_name" if "stage_name" in df.columns else None
    base_txt = ((df[act_col] if act_col else pd.Series([""]*len(df)))\
               .astype(str) + " " + (df[stage_col] if stage_col else pd.Series([""]*len(df))).astype(str)).str.strip()
    df["struct_type"] = base_txt.map(classify_struct_type)

    return df, loc_cols

# ------------------------ Feature engineering -----------------------
def compute_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    alias: Dict[str, List[str]] = {
        "date": ["date", "created_date", "created_on", "submitted_date", "start_date", "createdtime", "created_time"],
        "project": ["project", "project_name"],
        "inspector": ["inspector", "engineer", "created_by", "user"],
        "eqc_type": ["eqc_type", "type", "checklist_type"],
        "activity": ["activity", "checklist", "scope", "work_item", "item"],
        "stage_name": ["stage", "stage_name", "stage_no", "stage_number"],
        "eqc": ["eqc", "eqc_no", "eqc_number", "eqc_id", "reference", "ref_no"],
        "location_l0": ["location_l0", "project_code"],
        "location_l1": ["location_l1", "tower", "building", "block", "wing", "zone"],
        "location_l2": ["location_l2"],
        "location_l3": ["location_l3"],
        "location_variable": ["location_variable", "location"],
        "status": ["status", "eqc_status", "approval_status", "current_status", "state"],
        "approver": ["approver", "approved_by"],
        "approved_timestamp": ["approved_timestamp", "approval_date", "approved_date", "closed_date", "completed_date", "end_date", "decision_date"],
        "time_zone": ["time_zone", "timezone"],
        "team": ["team", "dept", "department"],
        "total_eqc_stages": ["total_eqc_stages", "total_stages", "stages_total", "total"],
        "fail_stages": ["fail_stages"],
        "pct_fail": ["percent_fail", "pct_fail", "fail_percent", "fail"],
        "url": ["url", "link"],
        "remarks": ["remarks", "comments", "observation", "reason", "note"],
    }
    picked: Dict[str, str] = {}
    for key, keys in alias.items():
        col = first_present(df, keys)
        if col: picked[key] = col

    work = df.copy()

    start_col = picked.get("date")
    end_col   = picked.get("approved_timestamp")
    if start_col is None:
        for c in df.columns:
            if (("date" in c) or ("time" in c) or ("created" in c) or ("submit" in c)) and c != end_col:
                start_col = c; break
    if end_col is None:
        for c in df.columns:
            if (("approved" in c) or ("closed" in c) or ("completed" in c) or ("end" in c) or ("decision" in c)) and (("date" in c) or ("time" in c) or ("stamp" in c)):
                end_col = c; break

    work["_start_dt"] = parse_dt(work[start_col]) if start_col in work.columns else pd.NaT
    work["_end_dt"]   = parse_dt(work[end_col])   if end_col   in work.columns else pd.NaT

    work["event_dt"] = work["_end_dt"].combine_first(work["_start_dt"])
    if work["event_dt"].isna().any():
        candidate_cols = [c for c in work.columns
                          if any(tok in c for tok in ["date","time","created","submitted","approved","closed","completed","decision"])
                          and c not in {"_start_dt","_end_dt","event_dt"}]
        for c in candidate_cols:
            alt = parse_dt(work[c])
            mask = work["event_dt"].isna() & alt.notna()
            if mask.any():
                work.loc[mask, "event_dt"] = alt.loc[mask]

    now = pd.Timestamp.now(tz=None)
    work["tat_days"] = (work["_end_dt"] - work["_start_dt"]).dt.total_seconds() / 86400.0
    mask_in_prog = work["_start_dt"].notna() & work["_end_dt"].isna()
    work.loc[mask_in_prog, "tat_days"] = (now - work.loc[mask_in_prog, "_start_dt"]).dt.total_seconds() / 86400.0
    work["tat_days"] = pd.to_numeric(work["tat_days"], errors="coerce")
    work["_tat_was_negative"] = work["tat_days"] < 0
    work["tat_days"] = work["tat_days"].clip(lower=0)

    stat_col = picked.get("status")
    if stat_col in work.columns:
        work["status_norm"] = work[stat_col].apply(status_normalizer)
    else:
        work["status_norm"] = "other"

    tower_col = picked.get("location_l1") or picked.get("location_variable") or picked.get("location_l0")
    work["tower"] = work[tower_col].astype(str).map(norm_text) if tower_col in work.columns else ""

    act_col = picked.get("activity") or picked.get("eqc_type")
    if act_col in work.columns:
        work["activity"] = work[act_col].astype(str).map(norm_text)
    elif picked.get("eqc_type") in work.columns:
        work["activity"] = work[picked["eqc_type"]].astype(str).map(norm_text)
    else:
        work["activity"] = ""

    stage_col = picked.get("stage_name") or picked.get("stage") or picked.get("activity")
    work["stage_name"] = work[stage_col].astype(str).map(norm_text) if stage_col in work.columns else work["activity"]

    work["inspector_norm"] = work[picked.get("inspector")].astype(str).map(norm_text) if picked.get("inspector") in work.columns else ""
    work["approver_norm"]  = work[picked.get("approver")].astype(str).map(norm_text) if picked.get("approver") in work.columns else ""
    work["project_norm"]   = work[picked.get("project")].astype(str).map(norm_text) if picked.get("project") in work.columns else ""

    if picked.get("pct_fail") in work.columns:
        tmp = work[picked["pct_fail"]].astype(str).str.replace("%", "", regex=False)
        work["pct_fail_num"] = pd.to_numeric(tmp, errors="coerce")
    else:
        work["pct_fail_num"] = np.nan

    work["eqc_id"] = (work[picked.get("eqc")].astype(str).map(norm_text) if picked.get("eqc") in work.columns else work.index.astype(str))
    if picked.get("url") in work.columns:
        work["url_key"] = work[picked["url"]].astype(str).map(norm_text)
    else:
        work["url_key"] = work["eqc_id"].astype(str).map(norm_text)

    keep_cols = [
        picked.get("date", "date"), picked.get("project", "project"),
        picked.get("inspector", "inspector"), picked.get("eqc_type", "eqc_type"),
        picked.get("activity", "activity"), picked.get("stage_name", "stage"),
        picked.get("eqc", "eqc"), picked.get("status", "status"),
        picked.get("approver", "approver"), picked.get("approved_timestamp", "approved_timestamp"),
        picked.get("total_eqc_stages", "total_eqc_stages")
    ]
    cols_to_add = [c for c in keep_cols if (c in df.columns) and (c not in work.columns)]
    if cols_to_add:
        work = pd.concat([work, df[cols_to_add]], axis=1)

    return work, picked

# -------- PASS & TAT rules (unchanged logic) ----------
def patch_pass_event_times(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    appr = x[(x["status_norm"] == "approved") & x["event_dt"].notna()].copy()
    key_cols_full = [c for c in ["project_norm","eqc_id","inspector_norm","stage_name"] if c in appr.columns]
    appr["k_full"] = list(zip(*[appr[c].astype(str) for c in key_cols_full])) if key_cols_full else None
    map_full = appr.groupby("k_full")["event_dt"].max().to_dict() if key_cols_full else {}
    map_url_appr = appr.groupby("url_key")["event_dt"].max().to_dict() if "url_key" in appr.columns else {}
    key_cols_pe = [c for c in ["project_norm","eqc_id"] if c in appr.columns]
    appr["k_pe"] = list(zip(*[appr[c].astype(str) for c in key_cols_pe])) if key_cols_pe else None
    map_pe = appr.groupby("k_pe")["event_dt"].max().to_dict() if key_cols_pe else {}

    pass_mask = (x["status_norm"] == "pass") & (x["event_dt"].isna())
    if pass_mask.any() and key_cols_full:
        kf_vals = list(zip(*[x.loc[pass_mask, c].astype(str) for c in key_cols_full]))
        x.loc[pass_mask, "event_dt"] = [map_full.get(k, pd.NaT) for k in kf_vals]
    if pass_mask.any() and "url_key" in x.columns:
        idx = x.index[pass_mask & x["event_dt"].isna()]
        x.loc[idx, "event_dt"] = x.loc[idx, "url_key"].map(map_url_appr)
    if pass_mask.any() and key_cols_pe:
        idx2 = x.index[pass_mask & x["event_dt"].isna()]
        if len(idx2):
            kpe_vals = list(zip(*[x.loc[idx2, c].astype(str) for c in key_cols_pe]))
            x.loc[idx2, "event_dt"] = [map_pe.get(k, pd.NaT) for k in kpe_vals]
    if pass_mask.any() and "url_key" in x.columns:
        any_dt = (x.groupby("url_key")[["event_dt","_end_dt","_start_dt"]].max().max(axis=1))
        idx3 = x.index[pass_mask & x["event_dt"].isna()]
        x.loc[idx3, "event_dt"] = x.loc[idx3, "url_key"].map(any_dt)
    idx4 = x.index[(x["status_norm"] == "pass") & x["_end_dt"].isna() & x["event_dt"].notna()]
    x.loc[idx4, "_end_dt"] = x.loc[idx4, "event_dt"]
    return x

def backfill_end_dt_from_event(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    m = x["_end_dt"].isna() & x["event_dt"].notna() & ~x["status_norm"].eq("pass")
    x.loc[m, "_end_dt"] = x.loc[m, "event_dt"]
    return x

def recompute_tat(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    now = pd.Timestamp.now(tz=None)
    x["tat_days"] = (x["_end_dt"] - x["_start_dt"]).dt.total_seconds() / 86400.0
    m_in_prog = x["_start_dt"].notna() & x["_end_dt"].isna()
    x.loc[m_in_prog, "tat_days"] = (now - x.loc[m_in_prog, "_start_dt"]).dt.total_seconds() / 86400.0
    x["tat_days"] = pd.to_numeric(x["tat_days"], errors="coerce")
    x["_tat_was_negative"] = x["tat_days"] < 0
    x["tat_days"] = x["tat_days"].clip(lower=0)
    return x

def add_final_pass_flag(df: pd.DataFrame, id_col: str = "identity_key") -> pd.DataFrame:
    if df.empty or id_col not in df.columns: return df
    x = df.copy()
    pass_rows = x[(x["status_norm"] == "pass") & x["event_dt"].notna()]
    if pass_rows.empty:
        x["is_final_pass"] = False
        return x
    last_pass_dt = pass_rows.groupby(id_col)["event_dt"].max()
    x = x.merge(last_pass_dt.rename("last_pass_dt"), left_on=id_col, right_index=True, how="left")
    x["is_final_pass"] = (x["status_norm"].eq("pass")) & (x["event_dt"].notna()) & (x["event_dt"].eq(x["last_pass_dt"]))
    x.drop(columns=["last_pass_dt"], inplace=True)
    return x

def build_lifecycles_final_pass(df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    use_cols = [id_col, "eqc_id", "status_norm", "event_dt",
                "project_norm", "location_path", "tower", "activity",
                "inspector_norm", "_start_dt", "_end_dt", "stage_name", "is_final_pass"]
    use_cols = [c for c in use_cols if c in df.columns]
    x = df[use_cols].copy()
    x = x.dropna(subset=["event_dt"])
    if x.empty: return pd.DataFrame(), pd.DataFrame()
    for c in [id_col, "status_norm", "project_norm", "location_path",
              "tower", "activity", "inspector_norm", "eqc_id", "stage_name"]:
        if c in x.columns and x[c].dtype == "object":
            x[c] = x[c].astype("category")
    x.sort_values([id_col, "event_dt"], ascending=[True, True], inplace=True, kind="stable")
    term = x.get("is_final_pass", pd.Series([False]*len(x))).astype("int8")
    x["cycle"] = term.groupby(x[id_col], sort=False).shift(fill_value=0).cumsum() + 1
    gb = x.groupby([id_col, "cycle"], sort=False, observed=True)
    base = gb.agg(
        base_eqc_id=("eqc_id", "first"),
        project_norm=("project_norm", "first"),
        location_path=("location_path", "first"),
        tower=("tower", "first"),
        activity=("activity", "first"),
        inspector_norm=("inspector_norm", "first"),
        start_dt=("event_dt", "first"),
        stage_first=("stage_name", "first"),
    ).reset_index()
    last = gb[["event_dt", "status_norm"]].last().reset_index()
    last = last.rename(columns={"event_dt": "end_dt", "status_norm": "final_status"}, copy=False)
    stage_statuses = ["redo", "in_process", "approved", "rejected"]
    for s in stage_statuses:
        x[f"_c_{s}"] = (x["status_norm"] == s).astype("int8")
    counts = gb[[f"_c_{s}" for s in stage_statuses]].sum().reset_index()
    counts.rename(columns={f"_c_{s}": s for s in stage_statuses}, inplace=True)
    cyc = base.merge(last, on=[id_col, "cycle"], how="left").merge(counts, on=[id_col, "cycle"], how="left")
    now = pd.Timestamp.now(tz=None)
    cyc.loc[~cyc["final_status"].eq("pass"), "end_dt"] = now
    cyc["tat_days_cycle"] = (cyc["end_dt"] - cyc["start_dt"]).dt.total_seconds() / 86400.0
    cyc["tat_days_cycle"] = pd.to_numeric(cyc["tat_days_cycle"], errors="coerce").clip(lower=0)
    cyc["key"] = cyc[id_col].astype(str) + " · c" + cyc["cycle"].astype(str)
    eqc_sum = cyc.groupby(id_col, sort=False, observed=True).agg(
        cycles=("cycle", "max"),
        reopened=("cycle", lambda s: bool(s.max() > 1)),
        total_redo=("redo", "sum"),
        first_start=("start_dt", "min"),
        last_end=("end_dt", "max"),
        last_status=("final_status", "last"),
        any_pass=("final_status", lambda s: bool((s == "pass").any())),
    ).reset_index()
    return cyc, eqc_sum

def cycles_to_view_df(cyc: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if cyc.empty: return cyc
    shaped = cyc.rename(
        columns={"start_dt": "_start_dt", "end_dt": "_end_dt", "final_status": "status_norm"},
        copy=False
    )
    shaped["tat_days"] = shaped["tat_days_cycle"]
    shaped["eqc_id"] = shaped["key"]
    shaped[id_col] = shaped["key"]
    return shaped

def build_status_segments(df: pd.DataFrame, id_col: str = "identity_key") -> pd.DataFrame:
    if df.empty or id_col not in df.columns: return pd.DataFrame()
    use_cols = [id_col, "status_norm", "stage_name", "inspector_norm",
                "event_dt", "_end_dt", "project_norm", "tower", "location_path",
                "eqc_id", "floor", "struct_type", "place_local"]
    x = df[[c for c in use_cols if c in df.columns]].copy()
    if "event_dt" in x.columns and "_end_dt" in x.columns:
        x["event_dt"] = x["event_dt"].fillna(x["_end_dt"])
    x = x.dropna(subset=["event_dt"])
    if x.empty: return pd.DataFrame()
    x.sort_values([id_col, "event_dt"], inplace=True, kind="stable")
    grp_ai = x.groupby([id_col, "stage_name", "inspector_norm"], sort=False, dropna=False)
    x["_approved_seq"] = np.where(
        x["status_norm"].eq("approved"),
        grp_ai["status_norm"].transform(lambda s: s.eq("approved").cumsum()),
        0
    )
    base_key = (x["status_norm"].astype(str) + "||" +
                x.get("stage_name", pd.Series([""]*len(x))).astype(str) + "||" +
                x.get("inspector_norm", pd.Series([""]*len(x))).astype(str))
    x["seg_key"] = np.where(
        x["status_norm"].eq("approved"),
        base_key + "||" + x["_approved_seq"].astype(str),
        base_key
    )
    x["prev_seg"] = x.groupby(id_col)["seg_key"].shift(1)
    x = x[x["seg_key"] != x["prev_seg"]].copy()
    x["start"] = x["event_dt"]
    x["end"] = x.groupby(id_col)["start"].shift(-1)
    now = pd.Timestamp.now(tz=None)
    last_mask = x.groupby(id_col)["start"].transform("max").eq(x["start"])
    x.loc[last_mask, "end"] = now
    if "_end_dt" in df.columns:
        pass_map = (df[df["status_norm"] == "pass"][["identity_key","event_dt","_end_dt"]]
                    if "identity_key" in df.columns else
                    df[df["status_norm"] == "pass"][["identity_loc","event_dt","_end_dt"]].rename(columns={"identity_loc":"identity_key"}))
        if not pass_map.empty and "identity_key" in x.columns:
            x = x.merge(pass_map.rename(columns={"_end_dt":"_end_dt_pass"}), how="left",
                        left_on=["identity_key","start"], right_on=["identity_key","event_dt"])
            pass_seg = x["status_norm"].eq("pass") & x["_end_dt_pass"].notna()
            x.loc[pass_seg, "end"] = x.loc[pass_seg, "_end_dt_pass"]
            x.drop(columns=["event_dt","_end_dt_pass"], inplace=True, errors="ignore")
    bad = x["end"].isna() | (x["end"] < x["start"])
    x.loc[bad, "end"] = x.loc[bad, "start"]
    x["duration_days"] = (x["end"] - x["start"]).dt.total_seconds() / 86400.0
    x["duration_days"] = pd.to_numeric(x["duration_days"], errors="coerce").clip(lower=0)
    x["cycle_label"] = "c1"
    for c in [id_col, "project_norm", "tower", "location_path", "eqc_id",
              "cycle_label", "status_norm", "floor", "struct_type", "place_local",
              "stage_name", "inspector_norm"]:
        if c in x.columns:
            x[c] = x[c].astype(str).map(norm_text).str.slice(0, 160)
    return x[["project_norm", "tower", "floor", "struct_type", "place_local", "location_path", "eqc_id",
              id_col, "cycle_label", "status_norm", "stage_name", "inspector_norm",
              "start", "end", "duration_days"]]

def compute_current_status_per_identity(df: pd.DataFrame, id_col: str = "identity_key") -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return pd.DataFrame(columns=[id_col, "current_status"])
    x = df.copy()
    if "event_dt" in x.columns and x["event_dt"].notna().any():
        last_rows = x.sort_values("event_dt").groupby(id_col, as_index=False).tail(1)
    else:
        last_rows = x.groupby(id_col, as_index=False).tail(1)
    return last_rows[[id_col, "status_norm"]].rename(columns={"status_norm": "current_status"})

def compute_eqc_tat_to_final_pass(df: pd.DataFrame, id_col: str = "identity_key") -> pd.DataFrame:
    if df.empty or id_col not in df.columns: return pd.DataFrame()
    cols = [id_col, "event_dt", "status_norm", "_end_dt",
            "project_norm", "tower", "floor", "struct_type", "place_local",
            "eqc_id", "location_path"]
    cols = [c for c in cols if c in df.columns]
    x = df[cols].copy()
    if "event_dt" in x.columns and x["event_dt"].notna().any():
        x_sorted = x.sort_values("event_dt", kind="stable")
    else:
        x_sorted = x.copy()
    base = (x_sorted.groupby(id_col, as_index=False)
            .agg({
                "project_norm": "first", "tower": "first", "floor": "first",
                "struct_type": "first", "place_local": "first",
                "eqc_id": "first", "location_path": "first"
            }))
    start_first = x["event_dt"].groupby(x[id_col]).min().rename("start_first") if "event_dt" in x.columns else pd.Series(dtype="datetime64[ns]", name="start_first")
    pass_rows = x[x["status_norm"] == "pass"].copy()
    if not pass_rows.empty:
        pass_rows["pass_dt"] = pass_rows["_end_dt"].combine_first(pass_rows["event_dt"]) if "_end_dt" in pass_rows.columns else pass_rows["event_dt"]
        final_pass = pass_rows.groupby(id_col)["pass_dt"].max()
    else:
        final_pass = pd.Series(dtype="datetime64[ns]", name="pass_dt")
    cur = compute_current_status_per_identity(df, id_col=id_col)
    tat = (base.set_index(id_col)
                 .join(start_first)
                 .join(final_pass.rename("pass_time"))
                 .reset_index())
    tat["has_pass"] = tat["pass_time"].notna()
    if "start_first" in tat.columns:
        tat["tat_to_pass_days"] = (tat["pass_time"] - tat["start_first"]).dt.total_seconds() / 86400.0
        tat["tat_to_pass_days"] = pd.to_numeric(tat["tat_to_pass_days"], errors="coerce")
    else:
        tat["tat_to_pass_days"] = np.nan
    tat = tat.merge(cur, on=id_col, how="left")
    tat["current_status"] = tat["current_status"].fillna("other")
    for c in ["project_norm","tower","floor","struct_type","place_local","location_path","eqc_id"]:
        if c in tat.columns:
            tat[c] = tat[c].map(norm_text)
    return tat

def build_tile_text(row) -> str:
    def fmt(v):
        try: return f"{float(v):.1f}"
        except Exception: return "—"
    med_txt = fmt(row.get("median_tat", np.nan))
    parts = [
        f"EQCs {int(row.get('eqcs', 0))} | Closed {int(row.get('closed', 0))}",
        f"PASS {int(row.get('cur_pass', 0))} | APPR {int(row.get('cur_approved', 0))}",
        f"IP {int(row.get('cur_in_process', 0))} | REDO {int(row.get('cur_redo', 0))} | REJ {int(row.get('cur_rejected', 0))}",
        f"Med {med_txt}d"
    ]
    return "<br>".join(parts)

def node_key_from_row(row: pd.Series, keys: List[str]) -> str:
    vals = []
    for k in keys:
        v = row.get(k, "—")
        v = "—" if (pd.isna(v) or str(v).strip()=="" ) else str(v)
        vals.append(v)
    return " / ".join(vals)

def filter_by_node(df: pd.DataFrame, row: pd.Series, keys: List[str]) -> pd.DataFrame:
    x = df.copy()
    for k in keys:
        val = row.get(k, None)
        if val is not None and k in x.columns and str(val) not in {"(all)", "(none)"}:
            x = x[x[k] == val]
    return x

# ----------------------------- UI / Sidebar --------------------------
st.sidebar.header("📁 Data")
upl = st.sidebar.file_uploader("Upload DigiQC CSV/Excel", type=["csv", "xlsx", "xls"])
df_raw = load_data(upl)

# Engineer features
df_raw, picked_cols = compute_features(df_raw)

# Status guard
_valid = set(STATUS_KEYS + ["other"])
df_raw.loc[~df_raw["status_norm"].isin(_valid), "status_norm"] = "other"
df_raw.loc[df_raw["status_norm"].astype(str).str.strip() == "", "status_norm"] = "other"

# *** Apply timestamp rules ***
df_raw = patch_pass_event_times(df_raw)
df_raw = backfill_end_dt_from_event(df_raw)
df_raw = recompute_tat(df_raw)

df_raw, detected_loc_cols = build_location_features(df_raw, picked_cols)

# Identity
st.sidebar.header("🧬 Identity (URL-based)")
identity_mode = st.sidebar.radio("Treat records as…", ["URL only", "URL + Location"], index=1, horizontal=True)
if identity_mode == "URL + Location":
    fallback_path = df_raw["location_path"].replace("", np.nan)
    df_raw["identity_key"] = np.where(
        fallback_path.notna(),
        df_raw["url_key"].astype(str) + " @ " + df_raw["location_path"].astype(str),
        df_raw["url_key"].astype(str)
    )
else:
    df_raw["identity_key"] = df_raw["url_key"].astype(str)

# Friendly label
df_raw["identity_label"] = df_raw["eqc_id"].astype(str) + np.where(
    df_raw["location_path"].astype(str).str.len() > 0,
    " — " + df_raw["location_path"].astype(str),
    ""
)

df_raw = add_final_pass_flag(df_raw, id_col="identity_key")

st.sidebar.header("🎨 Theme")
st.sidebar.caption("SJCPL theme is locked to brand palette.")

st.sidebar.header("🔎 Filters")
include_no_date = st.sidebar.checkbox("Include records with no event date", value=True)

df = df_raw.copy()
if ("event_dt" in df.columns) and (df["event_dt"].notna().any()):
    min_d = pd.to_datetime(df["event_dt"]).min()
    max_d = pd.to_datetime(df["event_dt"]).max()
    dr = st.sidebar.date_input("Date range (by event date)", value=(min_d.date(), max_d.date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        in_range = df["event_dt"].dt.date.between(dr[0], dr[1])
        df = df[in_range | df["event_dt"].isna()] if include_no_date else df[in_range]

projects_all = sorted(df["project_norm"].dropna().unique().tolist())
chosen_proj = st.sidebar.multiselect("Project(s)", options=projects_all,
                                     default=projects_all[:5] if len(projects_all)>5 else projects_all)
if chosen_proj:
    df = df[df["project_norm"].isin(chosen_proj)]

# Location filters
if detected_loc_cols:
    st.sidebar.markdown("**Location filters**")
    df["location_path"] = df["location_path"].map(norm_text)
    uniq_paths = sorted([p for p in df["location_path"].dropna().unique().tolist() if p])
    selected_paths = st.sidebar.multiselect("Location (path)", options=uniq_paths, default=[])
    if selected_paths:
        df = df[df["location_path"].isin(selected_paths)]
    loc_search = st.sidebar.text_input("Location contains (text search)", "")
    if loc_search.strip():
        pat = norm_text(loc_search)
        df = df[df["location_path"].str.lower().str.contains(pat.lower(), na=False)]

# Inspector filter
if "inspector_norm" in df.columns:
    df["inspector_norm"] = df["inspector_norm"].map(norm_text)
    inspectors = sorted(df["inspector_norm"].dropna().unique().tolist())
    chosen_ins = st.sidebar.multiselect("Inspector(s)", options=inspectors, default=[])
    if chosen_ins:
        df = df[df["inspector_norm"].isin(chosen_ins)]

# Status filter
statuses = STATUS_KEYS + ["other"]
chosen_status = st.sidebar.multiselect("Status", options=statuses, default=STATUS_KEYS)
if chosen_status:
    df = df[df["status_norm"].isin(chosen_status)]

# Tower filter
df["tower"] = df["tower"].map(norm_text)
towers_all = sorted([t for t in df["tower"].dropna().unique().tolist() if t])
chosen_towers = st.sidebar.multiselect("Tower(s)", options=towers_all, default=[])
if chosen_towers:
    df = df[df["tower"].isin(chosen_towers)]

sla_days = st.sidebar.number_input("SLA threshold (days)", min_value=0.0, value=3.0, step=0.5)
st.session_state["sla_days_global"] = sla_days
show_outliers_only = st.sidebar.checkbox("Show only items breaching SLA (stage-level)", value=False)
if show_outliers_only:
    df = df[(df["tat_days"] > sla_days) & df["tat_days"].notna()]

st.sidebar.header("📦 Aggregate")
agg_mode = st.sidebar.radio("Aggregate mode", ["Records (raw)", "Lifecycles (FINAL PASS closes)"], index=0, horizontal=True)
enable_lifecycles = st.sidebar.checkbox("Enable lifecycles (uses more memory)", value=True)

# Lifecycles
cycles_all = pd.DataFrame(); eqc_summary = pd.DataFrame()
if enable_lifecycles:
    if len(df) > 400_000:
        st.warning("View is large; lifecycles skipped. Narrow filters or uncheck the guard to compute anyway.")
    else:
        cycles_all, eqc_summary = build_lifecycles_final_pass(df.rename(columns={"identity_key":"identity_key"}), id_col="identity_key")

df_view = cycles_to_view_df(cycles_all, id_col="identity_key") if (not cycles_all.empty and agg_mode.startswith("Lifecycle")) else df.copy()

# ----------------------------- Branded Header -------------------------------
st.markdown(f"""
<div class="sj-header">
  <h1>🛠️ DigiQC — EQC Insights Dashboard (v2.19)</h1>
  <p>SJCPL visual theme · Roboto · Brand colors only</p>
</div>
""", unsafe_allow_html=True)
st.caption("Treemap tiles include in-tile badges; details panel shows donuts for node status/inspectors/approvers. URL never shown.")

# ----------------------------- KPIs -------------------------------
file_rows = len(df_raw); view_rows = len(df_view)
file_entities = df_raw["identity_key"].nunique(); view_entities = df["identity_key"].nunique()
a,b,c,d = st.columns(4)
a.metric("Event rows — file", f"{file_rows:,}")
b.metric("Event rows — view", f"{view_rows:,}")
c.metric("Identities (URL or URL+Location) — file", f"{file_entities:,}")
d.metric("Identities (URL or URL+Location) — view", f"{view_entities:,}")

# EQC closure stats
if ("identity_key" in df.columns) and ("event_dt" in df.columns) and df["event_dt"].notna().any():
    last_event = (
        df.sort_values("event_dt")
          .groupby("identity_key", as_index=False, sort=False)
          .tail(1)
          .reset_index(drop=True)
    )
    final_pass_dt = (
        df.loc[df["status_norm"].eq("pass") & df["event_dt"].notna(), ["identity_key", "event_dt"]]
          .groupby("identity_key", as_index=True, sort=False)["event_dt"]
          .max()
          .rename("fp")
    )
    tmp = last_event.merge(final_pass_dt, left_on="identity_key", right_index=True, how="left")
    tmp["closed"] = tmp["status_norm"].eq("pass") & tmp["event_dt"].eq(tmp["fp"])
    eqc_closed = int(tmp["closed"].sum())
    eqc_open   = int(len(tmp) - eqc_closed)
else:
    eqc_closed = eqc_open = 0

e,f = st.columns(2)
e.metric("CLOSED (final PASS)", f"{eqc_closed:,}")
f.metric("OPEN (no final PASS yet)", f"{eqc_open:,}")

# Stage TAT KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)
approved = int((df_view["status_norm"] == "approved").sum())
in_process = int((df_view["status_norm"] == "in_process").sum())
redo = int((df_view["status_norm"] == "redo").sum())
rejected = int((df_view["status_norm"] == "rejected").sum())
passed = int((df_view["status_norm"] == "pass").sum())
k1.metric("Approved (stage)", f"{approved:,}")
k2.metric("In Process (stage)", f"{in_process:,}")
k3.metric("Redo (stage)", f"{redo:,}")
k4.metric("Rejected (stage)", f"{rejected:,}")
k5.metric("PASS rows", f"{passed:,}")
k6.metric("View rows", f"{view_rows:,}")

tat_series = df_view["tat_days"].copy()
cA, cB, cC = st.columns(3)
cA.metric("Median TAT (stage)", f"{tat_series.median():.1f} d" if not tat_series.empty else "—")
cB.metric("P75 TAT (stage)", f"{tat_series.quantile(0.75):.1f} d" if not tat_series.empty else "—")
cC.metric("Mean TAT (stage)", f"{tat_series.mean():.1f} d" if not tat_series.empty else "—")

if "_tat_was_negative" in df_view.columns and int(df_view["_tat_was_negative"].sum()) > 0:
    st.info(f"Note: {int(df_view['_tat_was_negative'].sum())} records had end < start; TAT clipped at 0 days.")

st.markdown("---")

# ----------------------------- Tabs -------------------------------
tabs = st.tabs([
    "Overview", "Project Status", "Project Explorer (URL-aware)",
    "Location-wise", "Tower-wise", "User-wise", "Activity-wise",
    "Timelines", "Lifecycles", "EQC Table", "🧪 Date/Time Audit"
])
(tab_overview, tab_status, tab_explorer,
 tab_location, tab_tower, tab_user, tab_activity,
 tab_timelines, tab_life, tab_table, tab_audit) = tabs

with tab_overview:
    c1, c2 = st.columns(2)
    status_counts = df_view["status_norm"].value_counts().reindex(STATUS_KEYS, fill_value=0).reset_index()
    status_counts.columns = ["status", "count"]
    fig1 = px.bar(
        status_counts, x="status", y="count", title="Status Distribution (current view)", text_auto=True,
        color="status", color_discrete_map=STATUS_COLOR_MAP
    )
    c1.plotly_chart(fig1, use_container_width=True, key=plot_key("overview_status"))

    tat_plot = df_view["tat_days"].dropna()
    if not tat_plot.empty:
        fig2 = px.histogram(tat_plot, nbins=30, title="Turnaround Time (days) — Distribution")
        c2.plotly_chart(fig2, use_container_width=True, key=plot_key("overview_tat_hist"))
    else:
        c2.info("No TAT data to plot.")

    # NaT-safe week derivation
    if df_view["_start_dt"].notna().any():
        tmp = df_view[df_view["_start_dt"].notna()].copy()
        periods = tmp["_start_dt"].dt.to_period("W")
        tmp["week"] = periods.map(lambda r: getattr(r, "start_time", pd.NaT))
        agg_dict = {"total": ("identity_key", "count"), "median_tat": ("tat_days", "median")}
        for s in [s for s in STATUS_KEYS if s != "other"]:
            agg_dict[s] = ("status_norm", lambda col, ss=s: (col == ss).sum())
        trend = tmp.groupby("week").agg(**agg_dict).reset_index()

        cfd = trend.melt(id_vars=["week", "total", "median_tat"],
                         value_vars=[s for s in STATUS_KEYS if s != "other"],
                         var_name="status", value_name="count")
        flow_type = st.radio("Flow chart type", ["Line", "Area"], horizontal=True, index=0, key="ov_flow_type")
        if flow_type == "Line":
            fig_cfd = px.line(cfd, x="week", y="count", color="status",
                              title="Cumulative Flow by Week", markers=False,
                              color_discrete_map=STATUS_COLOR_MAP)
        else:
            fig_cfd = px.area(cfd, x="week", y="count", color="status",
                              title="Cumulative Flow by Week",
                              color_discrete_map=STATUS_COLOR_MAP)
        st.plotly_chart(fig_cfd, use_container_width=True, key=plot_key("overview_cfd"))

        fig4 = px.line(trend, x="week", y="median_tat", title="Weekly Median TAT (days)")
        st.plotly_chart(fig4, use_container_width=True, key=plot_key("overview_weekly_median"))

        if df_view["tat_days"].notna().any():
            tmp["breach"] = (tmp["tat_days"] > st.session_state.get("sla_days_global", 3.0))
            slat = tmp.groupby("week")["breach"].sum().reset_index(name="breaches")
            fig_b = px.bar(slat, x="week", y="breaches", title="SLA Breaches per Week")
            st.plotly_chart(fig_b, use_container_width=True, key=plot_key("overview_sla_breaches"))

with tab_status:
    proj = df_view.groupby(["project_norm", "status_norm"]).size().unstack(fill_value=0).reindex(columns=STATUS_KEYS, fill_value=0)
    proj["total"] = proj.sum(axis=1)
    st.subheader("Project-wise Status Summary (stage-level + PASS rows)")
    st.dataframe(proj.sort_values("total", ascending=False), use_container_width=True)

    top_n = st.slider("Top N projects", min_value=3, max_value=max(3, len(proj)), value=min(10, len(proj)))
    proj_top = proj.sort_values("total", ascending=False).head(top_n).reset_index()
    proj_melt = proj_top.melt(id_vars=["project_norm", "total"], value_vars=STATUS_KEYS, var_name="status", value_name="count")
    fig_ps = px.bar(proj_melt, x="project_norm", y="count", color="status", barmode="stack",
                    title="Project-wise EQC Status", color_discrete_map=STATUS_COLOR_MAP)
    st.plotly_chart(fig_ps, use_container_width=True, key=plot_key("status_proj_stack"))

with tab_explorer:
    st.subheader("Project → Location/Tower → Inspector → Status → EQC Explorer (URL-aware)")
    df_scope = df_view.copy()
    proj_sel = st.selectbox("Project", options=sorted(df_scope["project_norm"].unique().tolist()))
    df_scope = df_scope[df_scope["project_norm"] == proj_sel]

    loc_opts = sorted([p for p in df_scope["location_path"].dropna().unique().tolist() if p]) if "location_path" in df_scope.columns else []
    use_location = st.checkbox("Filter by exact Location (path)", value=bool(loc_opts))
    if use_location and loc_opts:
        loc_sel = st.selectbox("Location (path)", options=["All"] + loc_opts)
        if loc_sel != "All":
            df_scope = df_scope[df_scope["location_path"] == loc_sel]
    else:
        towers = sorted(df_scope["tower"].unique().tolist()) if "tower" in df_scope.columns else []
        tower_sel = st.selectbox("Tower", options=["All"] + towers) if towers else "All"
        if tower_sel != "All":
            df_scope = df_scope[df_scope["tower"] == norm_text(tower_sel)]

    if "inspector_norm" in df_scope.columns:
        ins_opts = sorted(df_scope["inspector_norm"].dropna().unique().tolist())
    else:
        ins_opts = []
    ins_sel = st.selectbox("Inspector", options=["All"] + ins_opts) if ins_opts else "All"
    if ins_sel != "All":
        df_scope = df_scope[df_scope["inspector_norm"] == ins_sel]

    status_sel = st.multiselect("Status filter", options=STATUS_KEYS + ["other"], default=STATUS_KEYS)
    if status_sel:
        df_scope = df_scope[df_scope["status_norm"].isin(status_sel)]

    if "identity_key" in df_scope.columns:
        id_to_label = (df_scope[["identity_key", "identity_label"]].drop_duplicates())
        eqc_labels = ["(none)"] + sorted(id_to_label["identity_label"].tolist())
        choice = st.selectbox("EQC (identity)", options=eqc_labels)
        if choice != "(none)":
            sel_identity = id_to_label.loc[id_to_label["identity_label"] == choice, "identity_key"].iloc[0]
            df_e = df.copy()
            df_e = df_e[df_e["identity_key"] == sel_identity].sort_values("event_dt")
        else:
            df_e = pd.DataFrame()
    else:
        st.warning("Identity key missing in current view.")
        df_e = pd.DataFrame()

    colV1, colV2 = st.columns(2)
    st_counts = df_scope["status_norm"].value_counts().reindex(STATUS_KEYS, fill_value=0).rename_axis("status").reset_index(name="count")
    fig_scope_status = px.pie(st_counts, names="status", values="count", hole=0.55, title="Status mix (scope)",
                              color="status", color_discrete_map=STATUS_COLOR_MAP)
    colV1.plotly_chart(fig_scope_status, use_container_width=True, key=plot_key("explorer_scope_status_pie"))

    tat_scope = df_scope["tat_days"].dropna()
    if not tat_scope.empty:
        fig_scope_tat = px.histogram(tat_scope, nbins=30, title="TAT distribution (stage-level scope)")
        colV2.plotly_chart(fig_scope_tat, use_container_width=True, key=plot_key("explorer_scope_tat_hist"))
    else:
        colV2.info("No TAT values in current scope.")

    if not df_e.empty:
        st.markdown("---")
        st.markdown("### Selected Identity — Snapshot, Status progression (segments) & TAT to FINAL PASS")
        tat_one = compute_eqc_tat_to_final_pass(df_e, id_col="identity_key")
        tat_val = tat_one["tat_to_pass_days"].iloc[0] if not tat_one.empty else np.nan
        overall_status = tat_one["current_status"].iloc[0] if not tat_one.empty else (
            "pass" if (df_e["status_norm"] == "pass").any() else (str(df_e["status_norm"].iloc[-1]) if not df_e.empty else "other")
        )
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Overall status", str(overall_status).upper())
        cB.metric("# PASS rows", int((df_e["status_norm"] == "pass").sum()))
        cC.metric("TAT to FINAL PASS (d)", f"{tat_val:.1f}" if pd.notna(tat_val) else "—")
        cD.metric("# Unique stages", df_e["stage_name"].nunique() if "stage_name" in df_e.columns else 0)

        rows = df_e.dropna(subset=["event_dt"]).sort_values("event_dt").copy()
        if not rows.empty:
            segs = build_status_segments(rows.assign(identity_key=rows["identity_key"]), id_col="identity_key")
            if not segs.empty:
                segs = segs.merge(rows[["identity_key","identity_label"]].drop_duplicates(), on="identity_key", how="left")
                segs["y"] = segs["identity_label"].iloc[0]
                segs = sanitize_cols_for_plot(segs, ["y","status_norm","stage_name","inspector_norm"])
                fig_prog = px.timeline(
                    segs, x_start="start", x_end="end", y="y",
                    color="status_norm", color_discrete_map=STATUS_COLOR_MAP,
                    hover_data=["stage_name","inspector_norm"]
                )
                fig_prog.update_yaxes(title=None, autorange="reversed")
                fig_prog.update_layout(title="Status Progression (every APPROVED shown separately)", height=280)
                st.plotly_chart(fig_prog, use_container_width=True, key=plot_key("explorer_status_progression"))
    else:
        st.info("Select an EQC to see URL-aware status & timelines.")

with tab_location:
    st.subheader("Location-wise — Project → Tower → Floor → Type → Local place (current status + TAT to FINAL PASS)")
    df_locview = df.copy()
    df_locview["location_path"] = df_locview["location_path"].map(norm_text)
    df_locview["tower"] = df_locview["tower"].map(norm_text)
    df_locview["identity_loc"] = np.where(
        df_locview["location_path"].replace("", np.nan).notna(),
        df_locview["url_key"].astype(str) + " @ " + df_locview["location_path"].astype(str),
        df_locview["url_key"].astype(str)
    )
    tat_eqc = compute_eqc_tat_to_final_pass(df_locview, id_col="identity_loc")
    id_loc = "identity_loc"

    if tat_eqc.empty:
        st.info("No rows to compute Location-wise overview.")
    else:
        for col in ["project_norm","tower","floor","struct_type","place_local"]:
            if col in tat_eqc.columns:
                tat_eqc[col] = tat_eqc[col].replace("", "—").fillna("—")

        for s in STATUS_KEYS + ["other"]:
            tat_eqc[f"is_{s}"] = (tat_eqc["current_status"] == s).astype(int)

        f1, f2, f3, f4 = st.columns(4)
        proj_opts = ["(all)"] + sorted(tat_eqc["project_norm"].unique().tolist())
        tower_opts = ["(none)"] + sorted(tat_eqc["tower"].unique().tolist())
        sel_proj  = f1.selectbox("Project filter", options=proj_opts, index=0)
        sel_tower = f2.selectbox("Tower focus (always include)", options=tower_opts, index=0)
        min_leaf  = f3.number_input("Min #EQCs per leaf", min_value=1, max_value=1000, value=1, step=1)
        max_depth = f4.slider("Max depth", 1, 5, 5)

        filt = tat_eqc.copy()
        if sel_proj != "(all)":
            filt = filt[filt["project_norm"] == sel_proj]

        keys_all = [c for c in ["project_norm","tower","floor","struct_type","place_local"] if c in filt.columns]
        keys = keys_all[:max_depth]

        grouped = (filt.groupby(keys, dropna=False)
                   .agg(
                        eqcs=(id_loc, "nunique"),
                        closed=("has_pass", "sum"),
                        median_tat=("tat_to_pass_days", lambda s: float(np.nanmedian(s)) if len(s)>0 else np.nan),
                        p75_tat=("tat_to_pass_days", lambda s: float(np.nanpercentile(s, 75)) if np.isfinite(np.nanmean(s)) else np.nan),
                        cur_pass=("is_pass", "sum"),
                        cur_approved=("is_approved", "sum"),
                        cur_in_process=("is_in_process", "sum"),
                        cur_redo=("is_redo", "sum"),
                        cur_rejected=("is_rejected", "sum"),
                        cur_other=("is_other", "sum"),
                   )
                   .reset_index())

        show = grouped[grouped["eqcs"] >= min_leaf].copy()
        if sel_tower != "(none)" and "tower" in grouped.columns:
            add = grouped[grouped["tower"] == sel_tower]
            if not add.empty:
                show = pd.concat([show, add], ignore_index=True).drop_duplicates(subset=keys).reset_index(drop=True)

        show["tile_text"] = show.apply(build_tile_text, axis=1)
        show["node_key"] = show.apply(lambda r: node_key_from_row(r, keys), axis=1)

        kA, kB, kC, kD = st.columns(4)
        kA.metric("Identities (URL+Location)", f"{tat_eqc[id_loc].nunique():,}")
        kB.metric("Closed (final PASS)", f"{int(show['closed'].sum()):,}")
        med_overall = float(np.nanmedian(filt["tat_to_pass_days"])) if filt["tat_to_pass_days"].notna().any() else np.nan
        kC.metric("Median TAT to FINAL PASS (overall)", f"{med_overall:.1f} d" if pd.notna(med_overall) else "—")
        kD.metric("Groups shown", f"{len(show):,}")

        chart_mode = st.radio("Hierarchy chart", ["Treemap", "Sunburst"], horizontal=True, index=0)
        path = [px.Constant("All")] + keys
        hover_data = {"eqcs": True, "closed": True, "median_tat": ":.2f", "p75_tat": ":.2f", "node_key": True}
        if chart_mode == "Treemap":
            fig_h = px.treemap(
                show, path=path, values="eqcs",
                color="median_tat", color_continuous_scale=SJCPL_CONT,
                hover_data=hover_data,
                title="EQCs by Project → Tower → Floor → Type → Local place"
            )
            fig_h.update_traces(textinfo="label+text", text=show["tile_text"])
        else:
            fig_h = px.sunburst(
                show, path=path, values="eqcs",
                color="median_tat", color_continuous_scale=SJCPL_CONT,
                hover_data=hover_data,
                title="EQCs by Project → Tower → Floor → Type → Local place"
            )
        st.plotly_chart(fig_h, use_container_width=True, key=plot_key("loc_hierarchy"))

        # ---------- Details panel ----------
        st.markdown("### Details for a node (donut + top people)")
        node_opts = ["(all)"] + show["node_key"].tolist()
        node_pick = st.selectbox("Node", options=node_opts, index=0, help="Select a treemap node path")
        if node_pick == "(all)":
            node_row = None
            node_df = filt.copy()
        else:
            node_row = show.loc[show["node_key"] == node_pick].iloc[0]
            node_df = filter_by_node(filt, node_row, keys)

        cL, cM, cR = st.columns(3)
        if not node_df.empty:
            stat_df = (node_df["current_status"]
                       .value_counts()
                       .reindex(STATUS_KEYS + ["other"], fill_value=0)
                       .rename_axis("current_status")
                       .reset_index(name="count"))
            fig_donut = px.pie(
                stat_df, names="current_status", values="count",
                hole=0.6, title="Current status mix (identities)",
                color="current_status", color_discrete_map=STATUS_COLOR_MAP
            )
            cL.plotly_chart(fig_donut, use_container_width=True, key=plot_key("loc_node_status_donut"))
        else:
            cL.info("No identities for this node.")

        if node_pick == "(all)":
            ids_in_node = tat_eqc[id_loc].unique().tolist()
        else:
            ids_in_node = node_df[id_loc].unique().tolist()

        events_node = df_locview[df_locview[id_loc].isin(ids_in_node)].copy()
        if not events_node.empty and "inspector_norm" in events_node.columns:
            top_insp = (events_node["inspector_norm"].replace("", np.nan).dropna()
                        .value_counts().head(10).reset_index())
            top_insp.columns = ["inspector", "events"]
            fig_insp = px.bar(top_insp, x="inspector", y="events", title="Top Inspectors (events in node)")
            cM.plotly_chart(fig_insp, use_container_width=True, key=plot_key("loc_top_insp"))
        else:
            cM.info("No inspector data in node.")

        if not events_node.empty and "approver_norm" in events_node.columns:
            mask_ap = events_node["status_norm"].isin(["approved", "pass"])
            top_app = (events_node.loc[mask_ap, "approver_norm"].replace(["", "-"], np.nan).dropna()
                       .value_counts().head(10).reset_index())
            if not top_app.empty:
                top_app.columns = ["approver", "events"]
                fig_app = px.bar(top_app, x="approver", y="events", title="Top Approvers (approved+pass in node)")
                cR.plotly_chart(fig_app, use_container_width=True, key=plot_key("loc_top_approvers"))
            else:
                cR.info("No approver rows (approved/pass) in this node.")
        else:
            cR.info("No approver data in node.")

        # ---- Lifecycle Treemap (brand colors) ----
        st.markdown("---")
        st.markdown("### Lifecycle Treemap for URL+Location (grouped by location; size = time in status) + Donut panel")
        segs_loc = build_status_segments(df_locview.rename(columns={"identity_loc": "identity_loc"}), id_col="identity_loc")
        if segs_loc.empty:
            st.info("Not enough timestamps to build lifecycle segments for URL+Location.")
        else:
            segs_plot = segs_loc.copy()
            if sel_proj != "(all)":
                segs_plot = segs_plot[segs_plot["project_norm"] == sel_proj]
            if sel_tower != "(none)":
                segs_plot = segs_plot[segs_plot["tower"] == sel_tower]

            cur_stat = compute_current_status_per_identity(df_locview.rename(columns={"identity_loc": "identity_loc"}), id_col="identity_loc")
            segs_plot = segs_plot.merge(cur_stat, on="identity_loc", how="left")

            max_ids_loc = st.slider("Max URL+Location identities to plot (lifecycle)", 50, 3000, 600, step=50, key="loc_life_max_ids")
            keep_ids = (segs_plot.groupby("identity_loc")["duration_days"].sum()
                                   .sort_values(ascending=False)
                                   .head(max_ids_loc).index)
            segs_plot = segs_plot[segs_plot["identity_loc"].isin(keep_ids)]

            min_days_loc = st.number_input("Min segment duration (days) — lifecycle", min_value=0.0, value=0.0, step=0.5, key="loc_life_min_days")
            segs_plot = segs_plot[segs_plot["duration_days"] >= float(min_days_loc)]

            segs_plot = segs_plot.merge(
                df_locview[["identity_loc","identity_label"]].drop_duplicates(),
                on="identity_loc", how="left"
            )
            segs_plot["identity_label"] = segs_plot["identity_label"].fillna("")

            base_path = ["project_norm", "tower", "floor", "struct_type", "place_local", "identity_label", "status_norm"]
            max_depth_loc = st.slider("Max hierarchy depth (lifecycle)", 3, 8, 7)
            path_keys = base_path[:max_depth_loc]

            size_metric_loc = st.radio("Treemap size by (lifecycle)", ["Time in status (days)", "Segment count"], horizontal=True, index=0, key="loc_life_size")
            segs_plot["metric"] = segs_plot["duration_days"].astype(float) if size_metric_loc == "Time in status (days)" else 1.0

            segs_plot = sanitize_cols_for_plot(segs_plot, ["project_norm","tower","floor","struct_type","place_local","identity_label","status_norm","stage_name","inspector_norm"])

            # Display-only NaT safety + blanks -> em dash
            text_cols = ["project_norm","tower","floor","struct_type","place_local","identity_label","status_norm","stage_name","inspector_norm"]
            segs_plot = fill_blanks_for_display(segs_plot, text_cols)
            for c in ["start", "end"]:
                if c in segs_plot.columns:
                    segs_plot[c] = pd.to_datetime(segs_plot[c], errors="coerce")
            if "start" in segs_plot.columns and "end" in segs_plot.columns:
                segs_plot["start"] = segs_plot["start"].fillna(segs_plot["end"])
                segs_plot["end"]   = segs_plot["end"].fillna(segs_plot["start"])
            if "current_status" in segs_plot.columns:
                segs_plot["current_status"] = segs_plot["current_status"].fillna("other")

            fig_life_loc = px.treemap(
                segs_plot,
                path=[px.Constant("All")] + path_keys,
                values="metric",
                color="status_norm",
                color_discrete_map=STATUS_COLOR_MAP,
                hover_data={
                    "duration_days": ":.2f",
                    "start": True,
                    "end": True,
                    "current_status": True,
                    "stage_name": True,
                    "inspector_norm": True
                },
                title="Lifecycle Treemap — Project → Tower → Floor → Type → Place → Identity → Status"
            )
            st.plotly_chart(fig_life_loc, use_container_width=True, key=plot_key("loc_lifecycle_treemap"))

            st.markdown("### Lifecycle node details (donut of TIME SHARE + top people)")
            life_keys = [k for k in ["project_norm","tower","floor","struct_type","place_local"] if k in segs_plot.columns]
            if not life_keys:
                st.info("No hierarchy columns to select a node.")
            else:
                life_nodes_df = (segs_plot.groupby(life_keys, dropna=False)["metric"].sum().reset_index())
                life_nodes_df["node_key"] = life_nodes_df.apply(lambda r: node_key_from_row(r, life_keys), axis=1)

                life_pick = st.selectbox("Lifecycle node", options=["(all)"] + life_nodes_df["node_key"].tolist(), index=0)

                if life_pick == "(all)":
                    life_node_df = segs_plot.copy()
                else:
                    life_row = life_nodes_df.loc[life_nodes_df["node_key"] == life_pick].iloc[0]
                    life_node_df = filter_by_node(segs_plot, life_row, life_keys)

                dL, dM, dR = st.columns(3)
                if not life_node_df.empty:
                    share = (life_node_df.groupby("status_norm")["duration_days"].sum()
                             .reindex(STATUS_KEYS + ["other"], fill_value=0).reset_index())
                    fig_donut_ts = px.pie(
                        share, names="status_norm", values="duration_days",
                        hole=0.6, title="Time share by status (duration)",
                        color="status_norm", color_discrete_map=STATUS_COLOR_MAP
                    )
                    dL.plotly_chart(fig_donut_ts, use_container_width=True, key=plot_key("loc_life_timeshare"))

                    top_insp_time = (life_node_df.groupby("inspector_norm")["duration_days"]
                                     .sum().sort_values(ascending=False).head(10).reset_index())
                    top_insp_time.columns = ["inspector", "days"]
                    fig_insp_time = px.bar(top_insp_time, x="inspector", y="days", title="Top Inspectors (by time in segments)")
                    dM.plotly_chart(fig_insp_time, use_container_width=True, key=plot_key("loc_life_top_insp_time"))

                    life_ids = life_node_df["identity_loc"].unique().tolist()
                    events_life_node = df_locview[df_locview["identity_loc"].isin(life_ids)]
                    if not events_life_node.empty and "approver_norm" in events_life_node.columns:
                        mask_ap2 = events_life_node["status_norm"].isin(["approved","pass"])
                        top_app2 = (events_life_node.loc[mask_ap2, "approver_norm"].replace(["", "-"], np.nan).dropna()
                                    .value_counts().head(10).reset_index())
                        if not top_app2.empty:
                            top_app2.columns = ["approver", "events"]
                            fig_app2 = px.bar(top_app2, x="approver", y="events", title="Top Approvers (approved+pass in node)")
                            dR.plotly_chart(fig_app2, use_container_width=True, key=plot_key("loc_life_top_approvers"))
                        else:
                            dR.info("No approver rows (approved/pass) in this lifecycle node.")
                    else:
                        dR.info("No approver data in this lifecycle node.")
                else:
                    dL.info("No segments in node.")
                    dM.empty(); dR.empty()

with tab_tower:
    if "tower" in df_view.columns:
        df_view["tower"] = df_view["tower"].map(norm_text)
        tower_grp = df_view.groupby(["tower", "status_norm"]).size().unstack(fill_value=0).reindex(columns=STATUS_KEYS, fill_value=0)
        tower_grp["total"] = tower_grp.sum(axis=1)
        st.dataframe(tower_grp.sort_values("total", ascending=False), use_container_width=True)

        top_n = st.slider("Top N towers", min_value=3, max_value=max(3, len(tower_grp)), value=min(10, len(tower_grp)))
        top_towers = tower_grp.sort_values("total", ascending=False).head(top_n).reset_index()
        top_melt = top_towers.melt(id_vars=["tower","total"], value_vars=STATUS_KEYS, var_name="status", value_name="count")
        fig = px.bar(top_melt, x="tower", y="count", color="status", barmode="stack",
                     title="Tower-wise EQC (stacked)", color_discrete_map=STATUS_COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True, key=plot_key("tower_stacked"))
    else:
        st.info("Tower column not available.")

with tab_user:
    if "inspector_norm" in df_view.columns:
        df_view["inspector_norm"] = df_view["inspector_norm"].map(norm_text)
        usr = (df_view.groupby(["inspector_norm", "status_norm"]).size()
               .unstack(fill_value=0).reindex(columns=STATUS_KEYS, fill_value=0))
        tat_user = df_view.groupby("inspector_norm")["tat_days"].median().rename("median_tat").to_frame()
        usr = usr.join(tat_user)
        usr["total"] = usr[STATUS_KEYS].sum(axis=1)
        usr = usr.sort_values("total", ascending=False)

        st.subheader("User-wise status (stage-level + PASS rows)")
        st.dataframe(usr, use_container_width=True)

        max_n = max(5, len(usr))
        top_n_users = st.slider("Top N inspectors for charts", min_value=5, max_value=max_n, value=min(25, max_n))
        usr_top = usr.head(top_n_users).reset_index()

        primary_y = "approved" if "approved" in usr_top.columns else STATUS_KEYS[0]
        fig_u1 = px.bar(usr_top, x="inspector_norm", y=primary_y, title=f"Top Inspectors by {primary_y.title()}", text_auto=True)
        st.plotly_chart(fig_u1, use_container_width=True, key=plot_key("user_top_by_primary"))

        fig_u2 = px.bar(usr_top, x="inspector_norm", y="median_tat", title="Median TAT (stage) by Inspector", text_auto=True)
        st.plotly_chart(fig_u2, use_container_width=True, key=plot_key("user_median_tat"))

        usr_melt = usr_top.melt(id_vars=["inspector_norm", "total", "median_tat"],
                                value_vars=STATUS_KEYS, var_name="status", value_name="count")
        fig_u3 = px.bar(usr_melt, x="inspector_norm", y="count", color="status",
                        barmode="stack", title="User-wise Status Mix",
                        color_discrete_map=STATUS_COLOR_MAP_PASTEL, category_orders={"status": STATUS_KEYS})
        st.plotly_chart(fig_u3, use_container_width=True, key=plot_key("user_status_mix"))

        # --- New: Redo/Rejected by Stage (per Inspector) ---
        st.markdown("### Redo/Rejected by Stage (per Inspector)")
        fail = df_view[df_view["status_norm"].isin(["redo", "rejected"])].copy()
        stage_col = "stage_name" if "stage_name" in fail.columns else ("activity" if "activity" in fail.columns else None)

        if stage_col is None or fail.empty:
            st.info("No stage/activity column or no redo/rejected rows in the current view.")
        else:
            inspectors_all = sorted(df_view["inspector_norm"].dropna().unique().tolist())
            default_users = usr_top["inspector_norm"].tolist() if "usr_top" in locals() else usr.index.tolist()
            mode = st.radio("View", ["Single inspector", "Multiple (facet)"], horizontal=True, index=0, key="redo_view_mode")

            if mode == "Single inspector":
                ins_pick = st.selectbox("Inspector", options=default_users if default_users else ["(none)"])
                if not ins_pick or ins_pick == "(none)":
                    st.info("Select an inspector to see their redo/rejected distribution by stage.")
                else:
                    data = (fail[fail["inspector_norm"] == ins_pick]
                            .groupby([stage_col, "status_norm"]).size()
                            .reset_index(name="count"))
                    if data.empty:
                        st.info("No redo/rejected rows for this inspector.")
                    else:
                        top_k = st.slider("Top stages", 5, 50, 15, key="redo_stage_topk_single")
                        top_stages = (data.groupby(stage_col)["count"].sum()
                                          .sort_values(ascending=False).head(top_k).index)
                        data = data[data[stage_col].isin(top_stages)]
                        data[stage_col] = data[stage_col].astype(str).str.slice(0, 80)
                        fig_urr = px.bar(
                            data, x=stage_col, y="count", color="status_norm",
                            barmode="stack",
                            title=f"Redo/Rejected by Stage — {ins_pick}",
                            color_discrete_map=STATUS_COLOR_MAP_PASTEL,
                            category_orders={"status_norm": ["redo", "rejected"]}
                        )
                        fig_urr.update_layout(xaxis_title="Stage", yaxis_title="Events")
                        st.plotly_chart(fig_urr, use_container_width=True, key=plot_key("user_redo_rej_single"))
            else:
                chosen = st.multiselect("Inspectors", options=inspectors_all, default=default_users[:6] if default_users else [])
                if not chosen:
                    st.info("Pick 1–6 inspectors to compare.")
                else:
                    data = (fail[fail["inspector_norm"].isin(chosen)]
                            .groupby(["inspector_norm", stage_col, "status_norm"]).size()
                            .reset_index(name="count"))
                    if data.empty:
                        st.info("No redo/rejected rows for the selected inspectors.")
                    else:
                        top_k = st.slider("Top stages (global)", 5, 40, 15, key="redo_stage_topk_multi")
                        top_stages = (data.groupby(stage_col)["count"].sum()
                                          .sort_values(ascending=False).head(top_k).index)
                        data = data[data[stage_col].isin(top_stages)]
                        data[stage_col] = data[stage_col].astype(str).str.slice(0, 60)
                        fig_urr_multi = px.bar(
                            data, x=stage_col, y="count", color="status_norm",
                            barmode="stack", facet_col="inspector_norm", facet_col_wrap=3,
                            title="Redo/Rejected by Stage — selected inspectors",
                            color_discrete_map=STATUS_COLOR_MAP_PASTEL,
                            category_orders={"status_norm": ["redo", "rejected"]}
                        )
                        fig_urr_multi.update_layout(xaxis_title="Stage", yaxis_title="Events", bargap=0.15)
                        st.plotly_chart(fig_urr_multi, use_container_width=True, key=plot_key("user_redo_rej_multi"))

        st.markdown("### 🔎 Stage Matrix — Redo/Rejected by Stage (per Inspector)")
        rr = df_view[df_view["status_norm"].isin(["redo", "rejected"])].copy()
        if not rr.empty and {"inspector_norm","stage_name"}.issubset(rr.columns):
            pivot = (rr.groupby(["inspector_norm","stage_name"]).size()
                       .rename("count").reset_index())
            pivot = pivot[pivot["inspector_norm"].isin(usr_top["inspector_norm"])]
            mt = pivot.pivot(index="inspector_norm", columns="stage_name", values="count").fillna(0).astype(int)
            top_k = st.slider("Max stages to display", 5, min(50, mt.shape[1] or 5), min(15, mt.shape[1] or 5))
            top_stage_order = mt.sum(axis=0).sort_values(ascending=False).head(top_k).index.tolist()
            mt = mt[top_stage_order] if len(top_stage_order) else mt
            fig_heat = px.imshow(
                mt, aspect="auto", text_auto=True,
                title="Redo + Rejected — counts by Inspector × Stage",
                color_continuous_scale=[[0, SJCPL["WHITE"]], [1, SJCPL["BLUE"]]],
                height=600, width=800
            )
            st.plotly_chart(fig_heat, use_container_width=True, key=plot_key("user_stage_heatmap"))
        else:
            st.info("No redo/rejected data to render the stage matrix.")
    else:
        st.info("Inspector column not available in current view.")

with tab_activity:
    if "activity" in df_view.columns:
        act = df_view.groupby(["activity", "status_norm"]).size().unstack(fill_value=0).reindex(columns=STATUS_KEYS, fill_value=0)
        act["total"] = act.sum(axis=1)
        st.dataframe(act.sort_values("total", ascending=False), use_container_width=True)

        top_n = st.slider("Top N activities", min_value=3, max_value=max(3, len(act)), value=min(10, len(act)), key="act-topn")
        act_top = act.sort_values("total", ascending=False).head(top_n).reset_index()

        fig_a0 = px.bar(
            act_top.melt(id_vars=["activity", "total"], value_vars=STATUS_KEYS, var_name="status", value_name="count"),
            x="activity", y="count", color="status", barmode="stack", title="Activity-wise EQC (stacked)",
            color_discrete_map=STATUS_COLOR_MAP
        )
        st.plotly_chart(fig_a0, use_container_width=True, key=plot_key("activity_stacked"))

        fail_df = df_view[df_view["status_norm"].isin(["rejected", "redo"])]
        pareto = fail_df.groupby("activity").size().sort_values(ascending=False).rename("count").reset_index()
        if not pareto.empty:
            pareto["cum_pct"] = pareto["count"].cumsum() / pareto["count"].sum() * 100
            fig_p = go.Figure()
            fig_p.add_bar(x=pareto["activity"], y=pareto["count"], name="Count", marker_color=SJCPL["BLUE"])
            fig_p.add_scatter(x=pareto["activity"], y=pareto["cum_pct"], yaxis="y2", mode="lines+markers", name="Cumulative %",
                              line=dict(color=SJCPL["GREY_600"]))
            fig_p.update_layout(
                title="Pareto of Fail/Redo (by activity)",
                yaxis=dict(title="Count"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
            )
            st.plotly_chart(fig_p, use_container_width=True, key=plot_key("activity_pareto"))
        else:
            st.info("No rejected/redo records to plot Pareto.")
    else:
        st.info("Activity column not available.")

with tab_timelines:
    st.subheader("Global Timelines & WIP Aging (URL-based identities)")
    now = pd.Timestamp.now(tz=None)
    tl = df_view[df_view["_start_dt"].notna()].copy()
    tl["finish"] = tl["_end_dt"].fillna(now)

    group_by = st.selectbox("Group rows by", ["identity_key", "identity_label", "eqc_id", "location_path", "tower", "project_norm", "inspector_norm", "activity"], index=0)
    if group_by in tl.columns:
        if group_by == "identity_key":
            tl_rows = tl.copy()
            tl_rows["y_disp"] = tl_rows["identity_label"]
        else:
            agg = tl.groupby(group_by).agg(_start_dt=("_start_dt", "min"), finish=("finish", "max"), count=("identity_key", "count")).reset_index()
            agg["status_norm"] = "other"
            tl_rows = agg
            tl_rows["y_disp"] = tl_rows[group_by]

        top_n = st.slider("Rows to show", 5, min(200, max(5, len(tl_rows))), value=min(40, len(tl_rows)))
        tl_rows = tl_rows.sort_values("_start_dt", ascending=False).head(top_n)

        if not tl_rows.empty:
            hover_cols = [c for c in ["project_norm", "tower", "location_path", "activity", "inspector_norm"] if c in tl_rows.columns] if "identity_key" in tl_rows.columns else ["count"]
            tl_rows_plot = sanitize_cols_for_plot(tl_rows, ["y_disp"] + hover_cols)
            fig_gl = px.timeline(
                tl_rows_plot, x_start="_start_dt", x_end="finish", y="y_disp",
                color="status_norm" if "status_norm" in tl_rows_plot.columns else None,
                color_discrete_map=STATUS_COLOR_MAP,
                hover_data=hover_cols
            )
            fig_gl.update_yaxes(autorange="reversed", title=None)
            fig_gl.add_vline(x=now, line_dash="dot", line_color=SJCPL["GREY"])
            fig_gl.update_layout(title=f"Timeline — grouped by {group_by}", height=500)
            st.plotly_chart(fig_gl, use_container_width=True, key=plot_key("timelines_grouped"))
        else:
            st.info("No rows to plot in timeline.")
    else:
        st.info(f"Column '{group_by}' not available for timeline grouping.")

    wip = df_view[(df_view["status_norm"] == "in_process") & df_view["_start_dt"].notna()].copy()
    if not wip.empty:
        wip = sanitize_cols_for_plot(wip, ["tower","identity_label","eqc_id","location_path","inspector_norm","activity","project_norm"])
        fig_age = px.scatter(
            wip, x="_start_dt", y="tat_days", color="tower" if "tower" in wip.columns else None,
            hover_data=[c for c in ["identity_label", "eqc_id", "location_path", "inspector_norm", "activity", "project_norm"] if c in wip.columns],
            title="WIP Aging — In-process stage entries"
        )
        st.plotly_chart(fig_age, use_container_width=True, key=plot_key("timelines_wip_age"))
    else:
        st.info("No in-process items for WIP Aging scatter.")

with tab_life:
    st.subheader("Lifecycles — treemap (size = time in status) — *every APPROVED is a separate segment*")

    id_basis = st.radio("Lifecycle identity", ["URL only", "URL + Location"], horizontal=True, index=0)
    if id_basis == "URL + Location":
        df_life = df.copy()
        df_life["identity_loc"] = np.where(
            df_life["location_path"].replace("", np.nan).notna(),
            df_life["url_key"].astype(str) + " @ " + df_life["location_path"].astype(str),
            df_life["url_key"].astype(str)
        )
        id_col_life = "identity_loc"
        label_map = df_life[["identity_loc","identity_label"]].drop_duplicates()
    else:
        df_life = df.copy()
        id_col_life = "identity_key"
        label_map = df_life[["identity_key","identity_label"]].drop_duplicates().rename(columns={"identity_key":"identity_loc"})

    segs_all = build_status_segments(df_life.rename(columns={id_col_life:"identity_loc"}), id_col="identity_loc")

    if segs_all.empty:
        st.info("Not enough timestamps to build lifecycle segments. Once records have event dates, the treemap will populate.")
    else:
        f1, f2, f3, f4 = st.columns([1,1,1,1])
        proj_opts = ["(all)"] + sorted(segs_all["project_norm"].unique().tolist())
        sel_proj  = f1.selectbox("Project", proj_opts, index=0)

        towers = sorted([t for t in segs_all["tower"].unique().tolist() if t])
        tower_opts = ["(all)"] + towers
        sel_twr  = f2.selectbox("Tower", tower_opts, index=0)

        max_ids  = f3.slider("Max identities to plot", 50, 3000, 500, step=50,
                             help="Keeps the figure responsive; identities ranked by total lifecycle duration.")
        min_days = f4.number_input("Min segment duration (days)", min_value=0.0, value=0.0, step=0.5)

        segs = segs_all.copy()
        if sel_proj != "(all)":
            segs = segs[segs["project_norm"] == sel_proj]
        if sel_twr != "(all)":
            segs = segs[segs["tower"] == sel_twr]

        keep_ids = (segs.groupby("identity_loc")["duration_days"].sum()
                          .sort_values(ascending=False)
                          .head(max_ids).index)
        segs = segs[segs["identity_loc"].isin(keep_ids)]
        segs = segs[segs["duration_days"] >= float(min_days)]

        if segs.empty:
            st.warning("No segments match the current filters.")
        else:
            segs = segs.merge(label_map, how="left", on="identity_loc")
            segs["identity_label"] = segs["identity_label"].fillna("")
            base_path = ["project_norm", "tower", "identity_label", "status_norm"]
            default_depth = 4
            max_depth = st.slider("Max hierarchy depth", min_value=3, max_value=len(base_path), value=default_depth)
            path_keys = base_path[:max_depth]

            size_metric = st.radio("Treemap size by", ["Time in status (days)", "Segment count"], horizontal=True, index=0)
            segs["metric"] = segs["duration_days"].astype(float) if size_metric == "Time in status (days)" else 1.0

            segs_plot = sanitize_cols_for_plot(segs, list({*base_path, "status_norm","stage_name","inspector_norm"}))
            text_cols = ["project_norm","tower","identity_label","status_norm","stage_name","inspector_norm"]
            segs_plot = fill_blanks_for_display(segs_plot, text_cols)
            for c in ["start", "end"]:
                if c in segs_plot.columns:
                    segs_plot[c] = pd.to_datetime(segs_plot[c], errors="coerce")
            if "start" in segs_plot.columns and "end" in segs_plot.columns:
                segs_plot["start"] = segs_plot["start"].fillna(segs_plot["end"])
                segs_plot["end"]   = segs_plot["end"].fillna(segs_plot["start"])

            fig_life = px.treemap(
                segs_plot,
                path=[px.Constant("All")] + path_keys,
                values="metric",
                color="status_norm",
                color_discrete_map=STATUS_COLOR_MAP,
                hover_data={
                    "duration_days": ":.2f",
                    "start": True,
                    "end": True,
                    "stage_name": True,
                    "inspector_norm": True
                },
                title="Lifecycle Treemap (URL-based identity; each APPROVED separate)"
            )
            st.plotly_chart(fig_life, use_container_width=True, key=plot_key("life_treemap"))

            st.markdown("#### Time share by status (filtered scope)")
            share = (segs_plot.groupby("status_norm")["duration_days"].sum()
                               .reindex(STATUS_KEYS, fill_value=0).reset_index())
            total_d = share["duration_days"].sum()
            share["pct"] = np.where(total_d>0, share["duration_days"]/total_d*100, 0.0)
            fig_share = px.bar(share, x="status_norm", y="pct", text_auto=".1f",
                               color="status_norm", color_discrete_map=STATUS_COLOR_MAP,
                               title="Share of time spent in each status (%)")
            st.plotly_chart(fig_share, use_container_width=True, key=plot_key("life_share"))

            st.markdown("### Lifecycle details (donut of TIME SHARE + top people)")
            life_keys = [k for k in ["project_norm","tower"] if k in segs_plot.columns]
            life_nodes_df = (segs_plot.groupby(life_keys, dropna=False)["metric"].sum().reset_index()) if life_keys else None
            node_opt = ["(all)"] if life_nodes_df is None else ["(all)"] + life_nodes_df.apply(lambda r: node_key_from_row(r, life_keys), axis=1).tolist()
            life_pick2 = st.selectbox("Lifecycle node (project/tower)", options=node_opt, index=0)

            if life_pick2 == "(all)" or life_nodes_df is None:
                life_node_df2 = segs_plot.copy()
            else:
                life_row2 = life_nodes_df.loc[
                    life_nodes_df.apply(lambda r: node_key_from_row(r, life_keys), axis=1) == life_pick2
                ].iloc[0]
                life_node_df2 = filter_by_node(segs_plot, life_row2, life_keys)

            eL, eM, eR = st.columns(3)
            if not life_node_df2.empty:
                share2 = (life_node_df2.groupby("status_norm")["duration_days"].sum()
                          .reindex(STATUS_KEYS + ["other"], fill_value=0).reset_index())
                fig_donut2 = px.pie(
                    share2, names="status_norm", values="duration_days",
                    hole=0.6, title="Time share by status (selected node)",
                    color="status_norm", color_discrete_map=STATUS_COLOR_MAP
                )
                eL.plotly_chart(fig_donut2, use_container_width=True, key=plot_key("life_node_timeshare"))

                top_insp_ts2 = (life_node_df2.groupby("inspector_norm")["duration_days"]
                                .sum().sort_values(ascending=False).head(10).reset_index())
                top_insp_ts2.columns = ["inspector", "days"]
                fig_insp2 = px.bar(top_insp_ts2, x="inspector", y="days", title="Top Inspectors (by time in node)")
                eM.plotly_chart(fig_insp2, use_container_width=True, key=plot_key("life_node_top_insp"))

                life_ids2 = life_node_df2["identity_loc"].unique().tolist()
                events_life_node2 = df[df["identity_key"].isin(life_ids2)] if "identity_key" in df.columns else df_raw[df_raw["identity_loc"].isin(life_ids2)]
                if "identity_key" in df.columns:
                    events_life_node2 = df[df["identity_key"].isin(life_ids2)]
                else:
                    events_life_node2 = df_locview[df_locview["identity_loc"].isin(life_ids2)]

                if not events_life_node2.empty and "approver_norm" in events_life_node2.columns:
                    mask_ap3 = events_life_node2["status_norm"].isin(["approved","pass"])
                    top_app3 = (events_life_node2.loc[mask_ap3, "approver_norm"].replace(["", "-"], np.nan).dropna()
                                .value_counts().head(10).reset_index())
                    if not top_app3.empty:
                        top_app3.columns = ["approver", "events"]
                        fig_app3 = px.bar(top_app3, x="approver", y="events", title="Top Approvers (approved+pass in node)")
                        eR.plotly_chart(fig_app3, use_container_width=True, key=plot_key("life_node_top_approvers"))
                    else:
                        eR.info("No approver rows (approved/pass) in this node.")
                else:
                    eR.info("No approver data in this node.")
            else:
                eL.info("No segments in node.")
                eM.empty(); eR.empty()

with tab_table:
    flags = pd.DataFrame(index=df_view.index)
    flags["breach_SLA"] = (df_view["tat_days"] > sla_days) & df_view["tat_days"].notna()
    flags["is_rejected"] = df_view["status_norm"].eq("rejected")
    flags["is_redo"]     = df_view["status_norm"].eq("redo")
    flags["is_pass"]     = df_view["status_norm"].eq("pass")
    out = pd.concat([df_view, flags], axis=1)

    show_cols_pref = [
        "date", "project_norm", "identity_label", "eqc_id", "location_path", "tower", "floor",
        "activity", "struct_type", "place_local", "stage_name", "status_norm", "inspector_norm", "approver_norm",
        "approved_timestamp", "tat_days", "pct_fail_num"
    ]
    show_cols = [c for c in dict.fromkeys(show_cols_pref) if c in out.columns]
    st.dataframe(out[show_cols].sort_values(["tat_days"], ascending=False), use_container_width=True)

    csv_data = out[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered table (CSV)", data=csv_data, file_name="digiqc_filtered.csv", mime="text/csv")

with tab_audit:
    st.write("**Detected columns (after normalization):**")
    st.json(picked_cols, expanded=False)

    def _count_na(series): 
        return int(series.isna().sum()) if series is not None else 0

    st.write("**Missing timestamp counts**")
    m1 = _count_na(df_raw.get("_start_dt", pd.Series(dtype="datetime64[ns]")))
    m2 = _count_na(df_raw.get("_end_dt", pd.Series(dtype="datetime64[ns]")))
    m3 = _count_na(df_raw.get("event_dt", pd.Series(dtype="datetime64[ns]")))
    st.table(pd.DataFrame({
        "metric": ["_start_dt missing", "_end_dt missing", "event_dt missing"],
        "rows":   [m1, m2, m3]
    }))

    if "status_norm" in df_raw.columns:
        tmp = df_raw.copy()
        tmp["_end_missing"] = tmp["_end_dt"].isna()
        by_stat = (tmp.groupby("status_norm")["_end_missing"]
                      .agg(rows="count", end_dt_missing="sum"))
        by_stat["end_dt_present"] = by_stat["rows"] - by_stat["end_dt_missing"]
        st.write("**End timestamp presence by current status**")
        st.dataframe(by_stat.reset_index(), use_container_width=True)

    start_col = picked_cols.get("date")
    end_col   = picked_cols.get("approved_timestamp")
    if start_col in df_raw.columns:
        st.write(f"**Top raw values where _start_dt is NaT (from '{start_col}')**")
        st.write(df_raw.loc[df_raw["_start_dt"].isna(), start_col]
                  .astype(str).value_counts().head(15))
    if end_col in df_raw.columns:
        st.write(f"**Top raw values where _end_dt is NaT (from '{end_col}')**")
        st.write(df_raw.loc[df_raw["_end_dt"].isna(), end_col]
                  .astype(str).value_counts().head(15))

    if "status_norm" in df_raw.columns:
        no_time_pass = df_raw[(df_raw["status_norm"]=="pass") & df_raw["event_dt"].isna()]
        st.write(f"**PASS rows with no usable time (after inference): {len(no_time_pass):,}**")
        if len(no_time_pass):
            st.dataframe(no_time_pass.head(20)[[
                "project_norm","eqc_id","location_path","inspector_norm","stage_name",
                "status_norm","event_dt","_start_dt","_end_dt","url_key"
            ]], use_container_width=True)

st.caption("Built with ❤️ for Digital QAQC — Streamlit app (v2.19 · SJCPL theme)")

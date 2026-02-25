import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Baseline Calculator", layout="wide")

# ============================================================
# Helpers
# ============================================================
def norm_key(s: str) -> str:
    s = str(s).replace("\u00A0", " ")
    s = s.strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_datetime_dmy(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.normalize()

def parse_support_pct_0_100(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def trimmed_mean(arr: np.ndarray, trim_ratio: float = 0.20) -> float:
    arr = arr.astype(float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return np.nan
    k = int(np.floor((trim_ratio / 2.0) * n))
    if n - 2 * k <= 0:
        return np.nan
    arr_sorted = np.sort(arr)
    trimmed = arr_sorted[k : n - k]
    return float(np.mean(trimmed)) if trimmed.size else np.nan

def format_date_mmddyyyy(df: pd.DataFrame, col: str = "transaction_date") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%m/%d/%Y")
    return df

def to_excel_bytes(cleaned_daily: pd.DataFrame, baseline: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cleaned_daily.to_excel(writer, sheet_name="daily_cleaned", index=False)
        baseline.to_excel(writer, sheet_name="baseline", index=False)
    return output.getvalue()

def safe_error(msg: str, err: Exception):
    st.error(msg)
    with st.expander("Detail error (safe)"):
        st.write(f"Type: {type(err).__name__}")
        st.write(f"Message: {str(err)}")

def add_quarter_index(df: pd.DataFrame, quarter_col: str = "quarter") -> pd.DataFrame:
    out = df.copy()
    year = out[quarter_col].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    q = out[quarter_col].astype(str).str.extract(r"Q([1-4])")[0].astype(float)
    out["quarter_index"] = (year * 10 + q).astype("Int64")
    return out

def normalize_channel_value(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00A0", " ").strip().upper()
    s = re.sub(r"\s+", " ", s).strip()
    if s in {"OFFLINE", "OFF LINE", "OFF-LINE", "STORE", "INSTORE", "IN-STORE"}:
        return "OFFLINE"
    if s in {"ONLINE", "ON LINE", "ON-LINE", "ECOM", "E-COM", "E-COMMERCE"}:
        return "ONLINE"
    return s

# ============================================================
# Robust Excel/CSV Reader (auto-detect header row)
# ============================================================
HEADER_HINTS = [
    "EAN", "EAN CODE", "SUPPORT", "SUPPORT%", "QTY", "SUM OF QTY",
    "TRANSACTION", "DATE", "QUARTER", "MPL", "TYPE STORE", "ONLINE", "OFFLINE"
]

def looks_like_real_header(row_values) -> bool:
    keys = [norm_key(v) for v in row_values if str(v).strip() not in {"", "nan", "None"}]
    if not keys:
        return False
    hit = 0
    for h in HEADER_HINTS:
        hk = norm_key(h)
        if any(hk in k for k in keys):
            hit += 1
    # if we hit enough keywords, treat it as header
    return hit >= 3

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    # try normal comma, then semicolon
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";")

def read_excel_robust(uploaded_file, sheet_name=None) -> pd.DataFrame:
    # Step 1: read first 30 rows with header=None to find header row
    uploaded_file.seek(0)
    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=30)

    header_row_idx = None
    for i in range(min(30, len(preview))):
        if looks_like_real_header(preview.iloc[i].tolist()):
            header_row_idx = i
            break

    # Step 2: read full file using detected header row
    uploaded_file.seek(0)
    if header_row_idx is None:
        # fallback: assume row 0
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row_idx)

    return df

def read_any_table(uploaded_file, sheet_name=None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return read_csv_robust(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return read_excel_robust(uploaded_file, sheet_name=sheet_name)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Supported file types: .csv, .xlsx/.xls, .parquet")

# ============================================================
# Column aliases
# ============================================================
UNIT_ALIASES = {
    "EAN CODE": ["EAN CODE", "EAN", "EAN_CODE", "EANCODE", "BARCODE"],
    "SUPPORT%": ["SUPPORT%", "SUPPORT %", "SUPPORT_PCT", "SUPPORT PCT", "DISCOUNT%", "DISCOUNT %", "DISC%"],
    "SUM OF QTY SUM": ["SUM OF QTY SUM", "SUM OF QTY", "QTY", "QTY SUM", "UNITS", "UNIT SOLD", "UNIT_SOLD"],
    "TRANSACTION DATE": ["TRANSACTION DATE", "TRANSACTION_DATE", "TRANSACTIONDATE", "DATE", "TRANS DATE"],
    "QUARTER": ["QUARTER", "QTR", "QUARTER NO", "QUARTER_NUMBER"],
    "MPL 2026": ["MPL 2026", "MPL2026", "MPL", "MPL NAME", "MPL_NAME"],
    "ONLINE/OFFLINE": [
        "TYPE STORE", "TYPE_STORE", "TYPE-STORE", "TYPESTORE",
        "ONLINE/OFFLINE", "ONLINE OFFLINE", "ONLINE_OFFLINE",
        "CHANNEL", "SALES CHANNEL"
    ],
    "PROMO TYPE": ["PROMOTION TYPE", "PROMO TYPE", "PROMO_TYPE", "PROMOTION_TYPE"],
}

SEAS_ALIASES = {
    "DATE": ["DATE", "TRANSACTION DATE", "TRANSACTION_DATE", "CALENDAR DATE", "CALDATE"],
}

def find_col_by_alias(df: pd.DataFrame, candidates: list[str]) -> str | None:
    key_to_actual = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        ck = norm_key(cand)
        if ck in key_to_actual:
            return key_to_actual[ck]
    return None

def standardize_unit_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    resolved = {}
    for canonical, alias_list in UNIT_ALIASES.items():
        resolved[canonical] = find_col_by_alias(df, alias_list)

    required = ["EAN CODE", "SUPPORT%", "SUM OF QTY SUM", "TRANSACTION DATE", "QUARTER", "MPL 2026", "ONLINE/OFFLINE"]
    missing_required = [k for k in required if resolved.get(k) is None]
    if missing_required:
        detected = [str(c) for c in df.columns]
        raise ValueError(f"Missing required columns in unit file: {missing_required}. Detected columns: {detected}")

    rename_map = {
        resolved["EAN CODE"]: "ean",
        resolved["SUPPORT%"]: "discount_pct",
        resolved["SUM OF QTY SUM"]: "unit_sold",
        resolved["TRANSACTION DATE"]: "transaction_date",
        resolved["QUARTER"]: "quarter_raw",
        resolved["MPL 2026"]: "mpl",
        resolved["ONLINE/OFFLINE"]: "channel",
    }

    promo_col = resolved.get("PROMO TYPE")
    if promo_col is not None:
        rename_map[promo_col] = "promo_type"

    df = df.rename(columns=rename_map)

    # OFFLINE filter
    df["channel_norm"] = df["channel"].apply(normalize_channel_value)
    df_off = df[df["channel_norm"] == "OFFLINE"].copy()
    if df_off.empty:
        unique_vals = sorted(set(df["channel_norm"].dropna().unique().tolist()))
        raise ValueError(
            "No rows found for OFFLINE after filtering. "
            f"Detected channel values: {unique_vals}. "
            "Please ensure Type Store column contains 'OFFLINE'."
        )
    df = df_off

    # types
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Transaction Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    df["ean"] = df["ean"].astype(str).str.strip()
    df["mpl"] = df["mpl"].astype(str).str.strip()
    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = parse_support_pct_0_100(df["discount_pct"])

    if "promo_type" not in df.columns:
        df["promo_type"] = "UNKNOWN"
    else:
        df["promo_type"] = df["promo_type"].astype(str).str.strip()
        df.loc[df["promo_type"].isin(["", "nan", "None", "NAN"]), "promo_type"] = "UNKNOWN"

    q = pd.to_numeric(df["quarter_raw"], errors="coerce").astype("Int64")
    year = df["transaction_date"].dt.year.astype("Int64")
    df["quarter"] = year.astype(str) + "Q" + q.astype(str)
    df = df.rename(columns={"channel_norm": "channel"})

    return df[["transaction_date", "quarter", "ean", "mpl", "unit_sold", "discount_pct", "promo_type", "channel"]]

def standardize_seasonality_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    date_col = find_col_by_alias(df, SEAS_ALIASES["DATE"])
    if date_col is None:
        detected = [str(c) for c in df.columns]
        raise ValueError(f"Seasonality file: kolom tanggal tidak ditemukan. Detected columns: {detected}")

    df = df.rename(columns={date_col: "transaction_date"})
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Seasonality Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")
    return df[["transaction_date"]].dropna().drop_duplicates()

# ============================================================
# Core processing
# ============================================================
def build_cleaned_and_baseline(unit_df, seasonality_df, upper_q, lower_q, trim_ratio):
    t1 = (
        unit_df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )
    seas_set = set(seasonality_df["transaction_date"].unique())
    t1["seasonality_flag"] = np.where(t1["transaction_date"].isin(seas_set), "Y", "N")

    base_for_fence = t1.loc[
        (t1["seasonality_flag"] == "N") & t1["unit_sold"].notna(),
        ["mpl", "quarter", "unit_sold"]
    ]

    fence = (
        base_for_fence.groupby(["mpl", "quarter"])["unit_sold"]
        .quantile([lower_q, upper_q])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={lower_q: "lower_fence", upper_q: "upper_fence"})
    )

    t3 = t1.merge(fence, on=["mpl", "quarter"], how="left")
    in_fence = t3["unit_sold"].ge(t3["lower_fence"]) & t3["unit_sold"].le(t3["upper_fence"])
    t3["outlier_flag"] = np.where(
        t3["lower_fence"].notna() & t3["upper_fence"].notna(),
        np.where(in_fence, "N", "Y"),
        "N"
    )

    weekday_ok = t3["transaction_date"].dt.weekday <= 4
    baseline_input = t3.loc[
        (t3["seasonality_flag"] == "N") &
        (t3["outlier_flag"] == "N") &
        weekday_ok &
        t3["unit_sold"].notna(),
        ["mpl", "quarter", "unit_sold"]
    ]

    def agg_trimmean(s: pd.Series) -> float:
        return trimmed_mean(s.to_numpy(dtype=float), trim_ratio=trim_ratio)

    t4 = (
        baseline_input.groupby(["mpl", "quarter"], as_index=False)
                      .agg(baseline=("unit_sold", agg_trimmean))
    )
    t4["baseline"] = t4["baseline"].round(2)
    return t3, t4

# ============================================================
# UI
# ============================================================
st.title("Baseline Calculator")

with st.sidebar:
    st.header("Parameters")
    upper_q = st.number_input("Upper fence percentile", 0.0, 1.0, 0.80, 0.01)
    lower_q = st.number_input("Lower fence percentile", 0.0, 1.0, 0.10, 0.01)
    trim_ratio = st.number_input("Trim ratio", 0.0, 0.8, 0.20, 0.05)
    st.caption("Note: Tool will automatically use OFFLINE rows only (Type Store = OFFLINE).")

c1, c2 = st.columns(2)
with c1:
    f_unit = st.file_uploader("Upload Unit Sold File", type=["csv", "xlsx", "xls", "parquet"])
with c2:
    f_seas = st.file_uploader("Upload Seasonality Calendar", type=["csv", "xlsx", "xls", "parquet"])

# If unit file is excel, allow sheet selection
sheet_choice = None
if f_unit and f_unit.name.lower().endswith((".xlsx", ".xls")):
    try:
        f_unit.seek(0)
        xl = pd.ExcelFile(f_unit)
        sheet_choice = st.selectbox("Select sheet (Unit Sold File)", xl.sheet_names, index=0)
    except Exception:
        sheet_choice = None

clicked = st.button("Calculate Baseline", type="primary", disabled=not (f_unit and f_seas))

if clicked:
    progress = st.progress(0)
    status = st.empty()

    try:
        if lower_q >= upper_q:
            st.error("Lower percentile must be < Upper percentile.")
            st.stop()

        with st.spinner("Calculating baseline..."):
            status.write("Step 1/5: Reading files...")
            progress.progress(10)

            unit_raw = read_any_table(f_unit, sheet_name=sheet_choice)
            seas_raw = read_any_table(f_seas)

            status.write("Step 2/5: Validating, filtering OFFLINE, & standardizing columns...")
            progress.progress(30)

            unit_df = standardize_unit_file(unit_raw)
            seas_df = standardize_seasonality_file(seas_raw)

            status.write("Step 3/5: Calculating outliers & cleaned daily (OFFLINE only)...")
            progress.progress(60)

            cleaned_daily, baseline = build_cleaned_and_baseline(
                unit_df=unit_df,
                seasonality_df=seas_df,
                upper_q=float(upper_q),
                lower_q=float(lower_q),
                trim_ratio=float(trim_ratio),
            )

            status.write("Step 4/5: Formatting outputs...")
            progress.progress(80)

            cleaned_daily_fmt = format_date_mmddyyyy(cleaned_daily, "transaction_date")

            status.write("Step 5/5: Preparing download...")
            progress.progress(95)

            xbytes = to_excel_bytes(cleaned_daily_fmt, baseline)
            progress.progress(100)
            status.empty()

        st.success("Completed")

        st.subheader("Baseline trend (quarter-to-quarter)")
        baseline_for_rank = baseline.copy()
        baseline_for_rank["mpl"] = baseline_for_rank["mpl"].astype(str).str.strip()

        mpl_rank = baseline_for_rank.groupby("mpl")["quarter"].nunique().sort_values(ascending=False)
        top20_mpl = mpl_rank.head(20).index.tolist()

        if not top20_mpl:
            st.warning("No baseline data available to plot.")
        else:
            default_mpl = "MYB_SSMI" if "MYB_SSMI" in top20_mpl else top20_mpl[0]
            selected_mpl = st.selectbox("Select MPL (Top 20 by coverage)", top20_mpl, index=top20_mpl.index(default_mpl))

            chart_df = baseline_for_rank[baseline_for_rank["mpl"] == selected_mpl].copy()
            chart_df = add_quarter_index(chart_df, "quarter").sort_values("quarter_index")
            st.line_chart(chart_df.set_index("quarter")[["baseline"]])

        tab1, tab2 = st.tabs([
            "Daily unit sold by MPL - cleaned up",
            "Baseline by MPL x quarter"
        ])
        with tab1:
            st.dataframe(cleaned_daily_fmt, use_container_width=True)
        with tab2:
            st.dataframe(baseline, use_container_width=True)

        st.download_button(
            "Download output (Excel)",
            data=xbytes,
            file_name="_baseline_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        progress.empty()
        status.empty()
        safe_error("Processing failed. Please check input file format.", e)

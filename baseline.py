import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Guardian Baseline Tool", layout="wide")

# ============================================================
# Helpers
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_datetime_dmy(s: pd.Series) -> pd.Series:
    # Input format: dd/mm/yy (e.g., 01/01/25)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.normalize()

def parse_support_pct_0_100(s: pd.Series) -> pd.Series:
    # "23%" -> 23
    x = s.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def trimmed_mean(arr: np.ndarray, trim_ratio: float = 0.20) -> float:
    arr = arr.astype(float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return np.nan
    k = int(np.floor((trim_ratio / 2.0) * n))  # each side
    if n - 2 * k <= 0:
        return np.nan
    arr_sorted = np.sort(arr)
    trimmed = arr_sorted[k : n - k]
    return float(np.mean(trimmed)) if trimmed.size else np.nan

def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Supported file types: .csv, .xlsx/.xls, .parquet")

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

# ============================================================
# Column aliases (robust)
# ============================================================
UNIT_ALIASES = {
    "EAN CODE": ["EAN CODE", "EAN", "EAN_CODE", "EANCODE", "BARCODE"],
    "SUPPORT%": ["SUPPORT%", "SUPPORT %", "SUPPORT_PCT", "SUPPORT PCT", "DISCOUNT%", "DISCOUNT %", "DISC%"],
    "SUM OF QTY SUM": ["SUM OF QTY SUM", "SUM OF QTY", "QTY", "QTY SUM", "UNITS", "UNIT SOLD", "UNIT_SOLD"],
    "TRANSACTION DATE": ["TRANSACTION DATE", "TRANSACTION_DATE", "TRANSACTIONDATE", "DATE", "TRANS DATE"],
    "QUARTER": ["QUARTER", "QTR", "QUARTER NO", "QUARTER_NUMBER"],
    "MPL 2026": ["MPL 2026", "MPL2026", "MPL", "MPL NAME", "MPL_NAME"],
}

SEAS_ALIASES = {
    "DATE": ["DATE", "TRANSACTION DATE", "TRANSACTION_DATE", "CALENDAR DATE", "CALDATE"],
}

def find_col_by_alias(df: pd.DataFrame, candidates_upper: list[str]) -> str | None:
    upper_map = {str(c).strip().upper(): c for c in df.columns}
    for cu in candidates_upper:
        if cu in upper_map:
            return upper_map[cu]
    return None

def standardize_unit_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    resolved = {}
    for canonical, alias_list in UNIT_ALIASES.items():
        resolved[canonical] = find_col_by_alias(df, [a.upper() for a in alias_list])

    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in unit file: {missing}")

    df = df.rename(columns={
        resolved["EAN CODE"]: "ean",
        resolved["SUPPORT%"]: "discount_pct",
        resolved["SUM OF QTY SUM"]: "unit_sold",
        resolved["TRANSACTION DATE"]: "transaction_date",
        resolved["QUARTER"]: "quarter_raw",
        resolved["MPL 2026"]: "mpl",
    })

    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Transaction Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    df["ean"] = df["ean"].astype(str).str.strip()
    df["mpl"] = df["mpl"].astype(str).str.strip()

    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = parse_support_pct_0_100(df["discount_pct"])  # 23% -> 23

    # Build quarter label: YYYYQ#
    q = pd.to_numeric(df["quarter_raw"], errors="coerce").astype("Int64")
    year = df["transaction_date"].dt.year.astype("Int64")
    df["quarter"] = year.astype(str) + "Q" + q.astype(str)

    # Not provided in your file -> default
    df["promo_type"] = "UNKNOWN"

    return df[["transaction_date", "quarter", "ean", "mpl", "unit_sold", "discount_pct", "promo_type"]]

def standardize_seasonality_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    date_col = find_col_by_alias(df, [a.upper() for a in SEAS_ALIASES["DATE"]])
    if date_col is None:
        raise ValueError("Seasonality file: kolom tanggal tidak ditemukan (contoh: Date).")

    df = df.rename(columns={date_col: "transaction_date"})
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Seasonality Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    return df[["transaction_date"]].dropna().drop_duplicates()

# ============================================================
# Core processing
# ============================================================
def build_cleaned_and_baseline(unit_df, seasonality_df, upper_q, lower_q, trim_ratio):
    # Internal Table 1: daily by MPL
    t1 = (
        unit_df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )

    # seasonality flag
    seas_set = set(seasonality_df["transaction_date"].unique())
    t1["seasonality_flag"] = np.where(t1["transaction_date"].isin(seas_set), "Y", "N")

    # Internal Table 2: outlier fences (non-seasonality only)
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

    # Shown Table 3: cleaned daily (includes flags + fences)
    t3 = t1.merge(fence, on=["mpl", "quarter"], how="left")

    in_fence = t3["unit_sold"].ge(t3["lower_fence"]) & t3["unit_sold"].le(t3["upper_fence"])
    t3["outlier_flag"] = np.where(
        t3["lower_fence"].notna() & t3["upper_fence"].notna(),
        np.where(in_fence, "N", "Y"),
        "N"
    )

    # Shown Table 4: baseline by MPL x quarter
    weekday_ok = t3["transaction_date"].dt.weekday <= 4  # Mon-Fri only
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

    # Round baseline to 2 decimals
    t4["baseline"] = t4["baseline"].round(2)

    return t3, t4

# ============================================================
# UI (Sales-friendly)
# ============================================================
st.title("Guardian Baseline Tool")

with st.sidebar:
    st.header("Parameters")
    upper_q = st.number_input("Upper fence percentile", 0.0, 1.0, 0.80, 0.01)
    lower_q = st.number_input("Lower fence percentile", 0.0, 1.0, 0.10, 0.01)
    trim_ratio = st.number_input("Trim ratio", 0.0, 0.8, 0.20, 0.05)

c1, c2 = st.columns(2)
with c1:
    f_unit = st.file_uploader("Upload Unit Sold File", type=["csv", "xlsx", "xls", "parquet"])
with c2:
    f_seas = st.file_uploader("Upload Seasonality Calendar", type=["csv", "xlsx", "xls", "parquet"])

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

            unit_raw = read_any_table(f_unit)
            seas_raw = read_any_table(f_seas)

            status.write("Step 2/5: Validating & standardizing columns...")
            progress.progress(30)

            unit_df = standardize_unit_file(unit_raw)
            seas_df = standardize_seasonality_file(seas_raw)

            status.write("Step 3/5: Calculating outliers & cleaned daily...")
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
            file_name="guardian_baseline_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        progress.empty()
        status.empty()
        safe_error("Processing failed. Please check input file format.", e)

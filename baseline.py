import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Guardian Baseline Tool", layout="wide")

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
        st.write(f"Type: {type(err)._name_}")
        st.write(f"Message: {str(err)}")

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
    return hit >= 3

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";")

def _ensure_df(obj, sheet_name=None) -> pd.DataFrame:
    if isinstance(obj, dict):
        if sheet_name and sheet_name in obj:
            return obj[sheet_name]
        first_key = list(obj.keys())[0]
        return obj[first_key]
    return obj

def read_excel_robust(uploaded_file, sheet_name=None) -> pd.DataFrame:
    uploaded_file.seek(0)
    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=30)
    preview = _ensure_df(preview, sheet_name=sheet_name)

    header_row_idx = None
    for i in range(min(30, len(preview))):
        if looks_like_real_header(preview.iloc[i].tolist()):
            header_row_idx = i
            break

    uploaded_file.seek(0)
    if header_row_idx is None:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row_idx)

    df = _ensure_df(df, sheet_name=sheet_name)
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
    "MPL 2026": ["MPL 2026", "MPL2026", "MPL", "MPL NAME", "MPL_NAME", "MPL 2025", "MPL2025"],
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

# ============================================================
# Standardization
# ============================================================
def standardize_unit_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    resolved = {k: find_col_by_alias(df, v) for k, v in UNIT_ALIASES.items()}

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
        resolved["QUARTER"]: "quarter",
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

    # Parse date
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Transaction Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    # Types
    df["ean"] = df["ean"].astype(str).str.strip()
    df["mpl"] = df["mpl"].astype(str).str.strip()

    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = parse_support_pct_0_100(df["discount_pct"])  # 23% -> 23

    # Quarter: keep as 1..4 (match your Excel)
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    if df["quarter"].isna().all():
        raise ValueError("Quarter gagal diparse (expected 1-4).")

    # Promo type (optional)
    if "promo_type" not in df.columns:
        df["promo_type"] = "UNKNOWN"
    else:
        df["promo_type"] = df["promo_type"].astype(str).str.strip()
        df.loc[df["promo_type"].isin(["", "nan", "None", "NAN"]), "promo_type"] = "UNKNOWN"

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
# Core processing (match Excel logic)
# ============================================================
def build_cleaned_and_baseline(unit_df, seasonality_df, upper_q, lower_q, trim_ratio):
    # (A) Detail for display
    t1_detail = (
        unit_df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )

    # (B) Base daily total per MPL (used for fence + baseline)
    t1_base = (
        unit_df.groupby(["transaction_date", "quarter", "mpl"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )

    seas_set = set(seasonality_df["transaction_date"].unique())
    t1_base["seasonality_flag"] = np.where(t1_base["transaction_date"].isin(seas_set), "Y", "N")

    # Fence calc input (non-seasonality + positive only)
    base_for_calc = t1_base[
        (t1_base["seasonality_flag"] == "N") &
        (t1_base["unit_sold"].notna()) &
        (t1_base["unit_sold"] > 0)
    ].copy()

    fence = (
        base_for_calc.groupby(["mpl", "quarter"])["unit_sold"]
        .quantile([lower_q, upper_q])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={lower_q: "lower_fence", upper_q: "upper_fence"})
    )

    t_base = t1_base.merge(fence, on=["mpl", "quarter"], how="left")

    in_fence = t_base["unit_sold"].ge(t_base["lower_fence"]) & t_base["unit_sold"].le(t_base["upper_fence"])
    t_base["outlier_flag"] = np.where(
        t_base["lower_fence"].notna() & t_base["upper_fence"].notna(),
        np.where(in_fence, "N", "Y"),
        "N"
    )

    # (C) Display cleaned daily = detail + flags from base
    t3 = t1_detail.merge(
        t_base[["transaction_date", "quarter", "mpl", "seasonality_flag", "outlier_flag", "lower_fence", "upper_fence"]],
        on=["transaction_date", "quarter", "mpl"],
        how="left"
    )

    # (D) Baseline from base daily totals
    weekday_ok = t_base["transaction_date"].dt.weekday <= 4  # Mon-Fri
    baseline_input = t_base[
        (t_base["seasonality_flag"] == "N") &
        (t_base["outlier_flag"] == "N") &
        weekday_ok &
        (t_base["unit_sold"].notna()) &
        (t_base["unit_sold"] > 0)
    ][["mpl", "quarter", "unit_sold"]]

    def agg_trimmean(s: pd.Series) -> float:
        return trimmed_mean(s.to_numpy(dtype=float), trim_ratio=trim_ratio)

    t4 = (
        baseline_input.groupby(["mpl", "quarter"], as_index=False)
                      .agg(baseline=("unit_sold", agg_trimmean))
    )
    t4["baseline"] = t4["baseline"].round(2)

    return t3, t4

# ============================================================
# Chart MPL list (fixed list you provided)
# ============================================================
CHART_MPL_LIST = [
    "MYB_SSMI",
    "GSN_MICELLAR_BASIC_125",
    "MYB_SSVI",
    "GSN_MICELLAR_BASIC_400",
    "MYB_FIT ME COMPACT PWD 12HR",
    "MYB_HYPERCURL",
    "MYB_SKY HIGH",
    "GCN_BIG KIT",
    "MYB_MAGNUM",
    "GSN_CLEANSER_BC_100",
    "LMU_INF LE MATTE RESISTANCE",
    "ELS_HA PURE SHP COND",
    "MYB_HYPERSHARP",
    "ELS_XO GOLD_100",
    "ELS_SHP_280",
    "DEX_GLYCO_MOIST_50",
    "DEX_GLYCO_SERUM_30",
    "ELS_GLYCOLIC GLOSS_280",
]

# ============================================================
# UI
# ============================================================
st.title("Guardian Baseline Tool")

with st.sidebar:
    st.header("Parameters")
    upper_q = st.number_input("Upper fence percentile", 0.0, 1.0, 0.80, 0.01)
    lower_q = st.number_input("Lower fence percentile", 0.0, 1.0, 0.10, 0.01)
    trim_ratio = st.number_input("Trim ratio (TRIMMEAN proportion)", 0.0, 0.8, 0.20, 0.05)
    st.caption("Note: Tool uses OFFLINE rows only (Type Store = OFFLINE).")

c1, c2 = st.columns(2)
with c1:
    f_unit = st.file_uploader("Upload Unit Sold File", type=["csv", "xlsx", "xls", "parquet"])
with c2:
    f_seas = st.file_uploader("Upload Seasonality Calendar", type=["csv", "xlsx", "xls", "parquet"])

# Sheet selectors (safe)
unit_sheet = None
if f_unit and f_unit.name.lower().endswith((".xlsx", ".xls")):
    try:
        f_unit.seek(0)
        xl = pd.ExcelFile(f_unit)
        unit_sheet = st.selectbox("Select sheet (Unit Sold File)", xl.sheet_names, index=0)
    except Exception:
        unit_sheet = None

seas_sheet = None
if f_seas and f_seas.name.lower().endswith((".xlsx", ".xls")):
    try:
        f_seas.seek(0)
        xl2 = pd.ExcelFile(f_seas)
        seas_sheet = st.selectbox("Select sheet (Seasonality Calendar)", xl2.sheet_names, index=0)
    except Exception:
        seas_sheet = None

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

            unit_raw = read_any_table(f_unit, sheet_name=unit_sheet)
            seas_raw = read_any_table(f_seas, sheet_name=seas_sheet)

            status.write("Step 2/5: Validating, filtering OFFLINE, & standardizing columns...")
            progress.progress(30)

            unit_df = standardize_unit_file(unit_raw)
            seas_df = standardize_seasonality_file(seas_raw)

            status.write("Step 3/5: Calculating outliers & baseline (OFFLINE only)...")
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

        # ============================================================
        # Baseline trend chart (your fixed MPL list) + DATA LABELS
        # ============================================================
        st.subheader("Baseline trend (quarter-to-quarter)")

        baseline_chart = baseline.copy()
        baseline_chart["mpl"] = baseline_chart["mpl"].astype(str).str.strip()

        # Keep only MPLs from your list (and only those present in results)
        present = sorted(set(baseline_chart["mpl"].unique().tolist()))
        dropdown_list = [m for m in CHART_MPL_LIST if m in present]

        if not dropdown_list:
            st.warning("None of the predefined MPL list exists in this baseline output.")
        else:
            default_mpl = "MYB_SSMI" if "MYB_SSMI" in dropdown_list else dropdown_list[0]
            selected_mpl = st.selectbox("Select MPL (predefined list)", dropdown_list, index=dropdown_list.index(default_mpl))

            chart_df = baseline_chart[baseline_chart["mpl"] == selected_mpl].copy()
            chart_df["quarter"] = pd.to_numeric(chart_df["quarter"], errors="coerce")
            chart_df = chart_df.dropna(subset=["quarter"]).sort_values("quarter")
            chart_df["quarter_label"] = "Q" + chart_df["quarter"].astype(int).astype(str)

            if chart_df.empty:
                st.warning(f'No baseline found for "{selected_mpl}".')
            else:
                # Altair line + points + text labels
                base = alt.Chart(chart_df).encode(
                    x=alt.X("quarter:O", title="Quarter"),
                    y=alt.Y("baseline:Q", title="Baseline"),
                    tooltip=["mpl:N", "quarter:O", alt.Tooltip("baseline:Q", format=".2f")]
                )

                line = base.mark_line()
                points = base.mark_point()
                labels = base.mark_text(dy=-10).encode(text=alt.Text("baseline:Q", format=".2f"))

                st.altair_chart((line + points + labels).properties(height=320), use_container_width=True)

        # ============================================================
        # Tables
        # ============================================================
        tab1, tab2 = st.tabs([
            "Daily unit sold by MPL - cleaned up",
            "Baseline by MPL x quarter"
        ])

        with tab1:
            st.dataframe(cleaned_daily_fmt, use_container_width=True)

        with tab2:
            st.dataframe(baseline.sort_values(["mpl", "quarter"]), use_container_width=True)

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

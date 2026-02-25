import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Guardian Baseline Builder (EAN)", layout="wide")

# ============================================================
# Helpers
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Trim spaces, unify internal spaces, keep original but add a normalized view
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def cols_upper(df: pd.DataFrame) -> list[str]:
    return [str(c).strip().upper() for c in df.columns]

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

def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Supported file types: .csv, .xlsx/.xls, .parquet")

def to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for k, df in tables.items():
            df.to_excel(writer, sheet_name=k[:31], index=False)
    return output.getvalue()

def safe_error(msg: str, details: dict | None = None):
    st.error(msg)
    if details:
        with st.expander("Debug info (safe)"):
            for k, v in details.items():
                st.write(f"**{k}:** {v}")

# ============================================================
# Standardize input files with aliases
# ============================================================
UNIT_ALIASES = {
    # target -> possible headers (upper-normalized)
    "EAN CODE": ["EAN CODE", "EAN", "EAN_CODE", "EANCODE", "BARCODE", "EAN BARCODE"],
    "SUPPORT%": ["SUPPORT%", "SUPPORT %", "SUPPORT_PCT", "SUPPORT PCT", "DISCOUNT%", "DISCOUNT %", "DISC%"],
    "SUM OF QTY SUM": ["SUM OF QTY SUM", "SUM OF QTY", "QTY", "QTY SUM", "SUM OF QTY", "UNITS", "UNIT SOLD", "UNIT_SOLD"],
    "TRANSACTION DATE": ["TRANSACTION DATE", "TRANSACTION_DATE", "TRANSACTIONDATE", "DATE", "TRANS DATE"],
    "QUARTER": ["QUARTER", "QTR", "QUARTER NO", "QUARTER_NUMBER"],
    "MPL 2026": ["MPL 2026", "MPL2026", "MPL", "MPL NAME", "MPL_NAME"],
}

SEAS_ALIASES = {
    "DATE": ["DATE", "TRANSACTION DATE", "TRANSACTION_DATE", "CALENDAR DATE", "CALDATE"],
}

def find_col_by_alias(df: pd.DataFrame, candidates_upper: list[str]) -> str | None:
    # Return actual column name that matches any candidate (case-insensitive)
    upper_map = {str(c).strip().upper(): c for c in df.columns}
    for cu in candidates_upper:
        if cu in upper_map:
            return upper_map[cu]
    return None

def standardize_unit_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    # resolve each required column via aliases
    resolved = {}
    for canonical, alias_list in UNIT_ALIASES.items():
        actual = find_col_by_alias(df, [a.upper() for a in alias_list])
        if actual is None:
            resolved[canonical] = None
        else:
            resolved[canonical] = actual

    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns (after alias matching): {missing}")

    # rename into internal schema
    df = df.rename(columns={
        resolved["EAN CODE"]: "ean",
        resolved["SUPPORT%"]: "discount_pct",
        resolved["SUM OF QTY SUM"]: "unit_sold",
        resolved["TRANSACTION DATE"]: "transaction_date",
        resolved["QUARTER"]: "quarter_raw",
        resolved["MPL 2026"]: "mpl",
    })

    # types
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Transaction Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    df["ean"] = df["ean"].astype(str).str.strip()
    df["mpl"] = df["mpl"].astype(str).str.strip()

    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = parse_support_pct_0_100(df["discount_pct"])  # 23% -> 23

    q = pd.to_numeric(df["quarter_raw"], errors="coerce").astype("Int64")
    year = df["transaction_date"].dt.year.astype("Int64")
    df["quarter"] = year.astype(str) + "Q" + q.astype(str)

    df["promo_type"] = "UNKNOWN"

    return df[["transaction_date", "quarter", "ean", "mpl", "unit_sold", "discount_pct", "promo_type"]]

def standardize_seasonality_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    date_col = find_col_by_alias(df, [a.upper() for a in SEAS_ALIASES["DATE"]])
    if date_col is None:
        raise ValueError("Seasonality file: kolom tanggal tidak ditemukan. Kolom yang dicari: Date / Transaction Date / Calendar Date.")

    df = df.rename(columns={date_col: "transaction_date"})
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Seasonality Date gagal diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    return df[["transaction_date"]].dropna().drop_duplicates()

# ============================================================
# Core processing
# ============================================================
def build_outputs(unit_df, seasonality_df, upper_q, lower_q, trim_ratio):
    table1 = (
        unit_df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )
    seas_set = set(seasonality_df["transaction_date"].unique())
    table1["seasonality_flag"] = np.where(table1["transaction_date"].isin(seas_set), "Y", "N")

    base_for_fence = table1.loc[
        (table1["seasonality_flag"] == "N") & table1["unit_sold"].notna(),
        ["mpl", "quarter", "unit_sold"]
    ]

    fence = (
        base_for_fence.groupby(["mpl", "quarter"])["unit_sold"]
        .quantile([lower_q, upper_q])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={lower_q: "lower_fence", upper_q: "upper_fence"})
    )
    fence["upper_q"] = float(upper_q)
    fence["lower_q"] = float(lower_q)
    table2 = fence[["mpl", "quarter", "upper_q", "lower_q", "upper_fence", "lower_fence"]].copy()

    table3 = table1.merge(
        table2[["mpl", "quarter", "upper_fence", "lower_fence"]],
        on=["mpl", "quarter"],
        how="left"
    )
    in_fence = (
        table3["unit_sold"].ge(table3["lower_fence"]) &
        table3["unit_sold"].le(table3["upper_fence"])
    )
    table3["outlier_flag"] = np.where(
        table3["lower_fence"].notna() & table3["upper_fence"].notna(),
        np.where(in_fence, "N", "Y"),
        "N"
    )

    weekday_ok = table3["transaction_date"].dt.weekday <= 4
    baseline_input = table3.loc[
        (table3["seasonality_flag"] == "N") &
        (table3["outlier_flag"] == "N") &
        weekday_ok &
        table3["unit_sold"].notna(),
        ["mpl", "quarter", "unit_sold"]
    ].copy()

    def agg_trimmean(s: pd.Series) -> float:
        return trimmed_mean(s.to_numpy(dtype=float), trim_ratio=trim_ratio)

    table4 = (
        baseline_input.groupby(["mpl", "quarter"], as_index=False)
                      .agg(baseline=("unit_sold", agg_trimmean),
                           n_obs=("unit_sold", "count"))
    )

    return {
        "table1_unit_by_mpl": table1,
        "table2_outlier_bank": table2,
        "table3_unit_by_mpl_cleaned": table3,
        "table4_baseline_daily_by_mpl": table4,
    }

# ============================================================
# UI
# ============================================================
st.title("Guardian Baseline Builder (Robust: EAN CODE + Seasonality Date)")

with st.sidebar:
    st.header("Parameters")
    upper_q = st.number_input("Upper fence percentile (0-1)", 0.0, 1.0, 0.80, 0.01)
    lower_q = st.number_input("Lower fence percentile (0-1)", 0.0, 1.0, 0.10, 0.01)
    trim_ratio = st.number_input("Trim ratio (0-0.8)", 0.0, 0.8, 0.20, 0.05)
    st.caption("Support% dibaca 23% -> 23 (skala 0-100).")

c1, c2 = st.columns(2)
with c1:
    f_unit = st.file_uploader("1) Unit sold file (EAN, Support%, Qty, Transaction Date, Quarter, MPL)", type=["csv", "xlsx", "xls", "parquet"])
with c2:
    f_seas = st.file_uploader("2) Seasonality file (Date, Holiday)", type=["csv", "xlsx", "xls", "parquet"])

if f_unit and f_seas:
    try:
        unit_raw = read_any_table(f_unit)
        seas_raw = read_any_table(f_seas)

        unit_raw = normalize_columns(unit_raw)
        seas_raw = normalize_columns(seas_raw)

        with st.expander("Debug: detected columns"):
            st.write("**Unit file columns:**", list(unit_raw.columns))
            st.write("**Seasonality file columns:**", list(seas_raw.columns))

        st.subheader("Preview")
        st.write("Unit sold file (top 10 rows):")
        st.dataframe(unit_raw.head(10), use_container_width=True)
        st.write("Seasonality file (top 10 rows):")
        st.dataframe(seas_raw.head(10), use_container_width=True)

        if st.button("Run baseline calculation", type="primary"):
            if lower_q >= upper_q:
                st.error("Lower percentile must be < Upper percentile.")
                st.stop()

            unit_df = standardize_unit_file(unit_raw)
            seas_df = standardize_seasonality_file(seas_raw)

            outputs = build_outputs(
                unit_df=unit_df,
                seasonality_df=seas_df,
                upper_q=float(upper_q),
                lower_q=float(lower_q),
                trim_ratio=float(trim_ratio),
            )

            st.success("Done!")

            tabs = st.tabs(["Table 1", "Table 2", "Table 3", "Table 4 (Baseline)"])
            keys = [
                "table1_unit_by_mpl",
                "table2_outlier_bank",
                "table3_unit_by_mpl_cleaned",
                "table4_baseline_daily_by_mpl",
            ]
            for t, k in zip(tabs, keys):
                with t:
                    st.dataframe(outputs[k], use_container_width=True)

            xbytes = to_excel_bytes(outputs)
            st.download_button(
                "Download all tables (Excel)",
                data=xbytes,
                file_name="guardian_baseline_outputs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        safe_error(
            "Processing failed. Biasanya karena header beda / tanggal tidak kebaca / ada kolom missing.",
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
else:
    st.info("Upload dua file: unit sold + seasonality.")

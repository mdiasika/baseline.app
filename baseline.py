import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Guardian Baseline Builder (EAN)", layout="wide")

# ============================================================
# Helpers
# ============================================================
def ensure_datetime_dmy(s: pd.Series) -> pd.Series:
    # your data: dd/mm/yy
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.normalize()

def parse_support_pct_0_100(s: pd.Series) -> pd.Series:
    """
    Input examples:
      '23%' -> 23
      '0%'  -> 0
      23    -> 23
      0.23  -> 0.23 (we keep as-is; but your expected input is 23%)
    """
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

# ============================================================
# Standardize input files to expected schema
# ============================================================
def standardize_unit_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expect EXACT headers from your sample:
      EAN CODE, Support%, Sum of Qty Sum, Transaction Date, Quarter, MPL 2026
    Output schema:
      transaction_date, quarter, ean, mpl, unit_sold, discount_pct, promo_type
    """
    df = df_raw.copy()

    expected = {
        "EAN CODE": "ean",
        "Support%": "discount_pct",
        "Sum of Qty Sum": "unit_sold",
        "Transaction Date": "transaction_date",
        "Quarter": "quarter_raw",
        "MPL 2026": "mpl",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di unit sold file: "
            + ", ".join(missing)
            + ". Pastikan header sama persis."
        )

    df = df.rename(columns=expected)

    # types
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Transaction Date tidak bisa diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")

    df["ean"] = df["ean"].astype(str).str.strip()
    df["mpl"] = df["mpl"].astype(str).str.strip()

    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = parse_support_pct_0_100(df["discount_pct"])  # 23% -> 23

    # quarter: build YYYYQ# using year from transaction_date
    q = pd.to_numeric(df["quarter_raw"], errors="coerce").astype("Int64")
    year = df["transaction_date"].dt.year.astype("Int64")
    df["quarter"] = year.astype(str) + "Q" + q.astype(str)

    # promo_type not provided -> set default
    df["promo_type"] = "UNKNOWN"

    return df[["transaction_date", "quarter", "ean", "mpl", "unit_sold", "discount_pct", "promo_type"]]

def standardize_seasonality_file(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Your seasonality file format:
      Date, Holiday
    We use Date only.
    """
    df = df_raw.copy()
    if "Date" not in df.columns:
        raise ValueError("Seasonality file harus punya kolom 'Date' (sesuai format kamu).")

    df = df.rename(columns={"Date": "transaction_date"})
    df["transaction_date"] = ensure_datetime_dmy(df["transaction_date"])
    if df["transaction_date"].isna().all():
        raise ValueError("Kolom Date di seasonality tidak bisa diparse. Pastikan format dd/mm/yy atau dd/mm/yyyy.")
    return df[["transaction_date"]].dropna().drop_duplicates()

# ============================================================
# Core processing
# ============================================================
def build_outputs(unit_df, seasonality_df, upper_q, lower_q, trim_ratio):
    # Table 1: by MPL daily + seasonality flag
    table1 = (
        unit_df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
               .agg(unit_sold=("unit_sold", "sum"))
    )
    seas_set = set(seasonality_df["transaction_date"].unique())
    table1["seasonality_flag"] = np.where(table1["transaction_date"].isin(seas_set), "Y", "N")

    # Table 2: outlier bank using only non-seasonality
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

    # Table 3: cleaned + outlier flag
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

    # Table 4: baseline (Mon-Fri only) + filters seasonality N & outlier N
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
st.title("Guardian Baseline Builder (Exact Headers: EAN CODE / Support% / MPL 2026)")

with st.sidebar:
    st.header("Parameters")
    upper_q = st.number_input("Upper fence percentile (0-1)", 0.0, 1.0, 0.80, 0.01)
    lower_q = st.number_input("Lower fence percentile (0-1)", 0.0, 1.0, 0.10, 0.01)
    trim_ratio = st.number_input("Trim ratio (0-0.8)", 0.0, 0.8, 0.20, 0.05)
    st.caption("Support% dibaca sebagai 23% -> 23 (0-100 scale).")

c1, c2 = st.columns(2)
with c1:
    f_unit = st.file_uploader(
        "1) Unit sold file (EAN CODE, Support%, Sum of Qty Sum, Transaction Date, Quarter, MPL 2026)",
        type=["csv", "xlsx", "xls", "parquet"]
    )
with c2:
    f_seas = st.file_uploader(
        "2) Seasonality file (Date, Holiday)",
        type=["csv", "xlsx", "xls", "parquet"]
    )

if f_unit and f_seas:
    unit_raw = read_any_table(f_unit)
    seas_raw = read_any_table(f_seas)

    st.subheader("Preview")
    st.write("Unit sold file (top 15 rows):")
    st.dataframe(unit_raw.head(15), use_container_width=True)

    st.write("Seasonality file (top 15 rows):")
    st.dataframe(seas_raw.head(15), use_container_width=True)

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

else:
    st.info("Upload dua file: unit sold + seasonality.")

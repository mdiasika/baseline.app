import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Guardian Baseline Builder", layout="wide")

# =========================
# Helpers
# =========================
def ensure_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.normalize()

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

def build_baseline_pipeline(
    unit_sold_daily: pd.DataFrame,
    seasonality_calendar: pd.DataFrame,
    plu_to_mpl: pd.DataFrame,
    upper_q: float,
    lower_q: float,
    trim_ratio: float = 0.20,
) -> dict[str, pd.DataFrame]:
    df = unit_sold_daily.copy()

    required = ["transaction_date", "quarter", "plu_code", "unit_sold", "discount_pct", "promo_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in unit_sold_daily: {missing}")

    if not {"plu_code", "mpl"}.issubset(plu_to_mpl.columns):
        raise ValueError("plu_to_mpl must have columns: ['plu_code','mpl']")

    seas = seasonality_calendar.copy()
    if "transaction_date" not in seas.columns:
        candidates = [c for c in seas.columns if "date" in c.lower()]
        if not candidates:
            raise ValueError("seasonality_calendar must have a date column (e.g., 'transaction_date')")
        seas = seas.rename(columns={candidates[0]: "transaction_date"})

    # types
    df["transaction_date"] = ensure_datetime(df["transaction_date"])
    seas["transaction_date"] = ensure_datetime(seas["transaction_date"])
    df["unit_sold"] = pd.to_numeric(df["unit_sold"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")

    # 1) PLU -> MPL + aggregate to MPL daily
    df = df.merge(plu_to_mpl[["plu_code", "mpl"]], on="plu_code", how="left")
    df["mpl"] = df["mpl"].fillna("UNKNOWN")

    table1 = (
        df.groupby(["transaction_date", "quarter", "mpl", "discount_pct", "promo_type"], as_index=False)
          .agg(unit_sold=("unit_sold", "sum"))
    )

    seas_set = set(seas["transaction_date"].dropna().unique())
    table1["seasonality_flag"] = np.where(table1["transaction_date"].isin(seas_set), "Y", "N")

    # 2) Outlier bank (MPL x Quarter) using non-seasonality only
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

    table2 = fence[["mpl", "quarter", "upper_q", "lower_q", "upper_fence", "lower_fence"]]

    # 3) Cleaned up (add outlier flag)
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

    # 4) Baseline (MPL x quarter)
    weekday_ok = table3["transaction_date"].dt.weekday <= 4  # Mon-Fri

    baseline_input = table3.loc[
        (table3["seasonality_flag"] == "N") &
        (table3["outlier_flag"] == "N") &
        weekday_ok &
        table3["unit_sold"].notna(),
        ["mpl", "quarter", "unit_sold"]
    ]

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

def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Supported: .csv, .xlsx/.xls, .parquet")

def to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for k, df in tables.items():
            sheet = k[:31]  # excel sheet name limit
            df.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()

# =========================
# UI
# =========================
st.title("Guardian Baseline Builder (MPL)")

with st.sidebar:
    st.header("Inputs")
    upper_q = st.number_input("Upper fence percentile (0-1)", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
    lower_q = st.number_input("Lower fence percentile (0-1)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    trim_ratio = st.number_input("Trim ratio (0-0.8)", min_value=0.0, max_value=0.8, value=0.20, step=0.05)

    st.caption("Upload files (CSV/XLSX/Parquet).")

colA, colB, colC = st.columns(3)

with colA:
    f1 = st.file_uploader("1) Unit sold daily (PLU-level)", type=["csv", "xlsx", "xls", "parquet"])
with colB:
    f2 = st.file_uploader("2) Seasonality calendar", type=["csv", "xlsx", "xls", "parquet"])
with colC:
    f3 = st.file_uploader("3) PLU → MPL mapping", type=["csv", "xlsx", "xls", "parquet"])

run = st.button("Run baseline calculation", type="primary", disabled=not (f1 and f2 and f3))

if run:
    if lower_q >= upper_q:
        st.error("Lower percentile must be < Upper percentile.")
        st.stop()

    try:
        unit_sold_daily = read_any_table(f1)
        seasonality_calendar = read_any_table(f2)
        plu_to_mpl = read_any_table(f3)

        tables = build_baseline_pipeline(
            unit_sold_daily=unit_sold_daily,
            seasonality_calendar=seasonality_calendar,
            plu_to_mpl=plu_to_mpl,
            upper_q=float(upper_q),
            lower_q=float(lower_q),
            trim_ratio=float(trim_ratio),
        )

        st.success("Done!")

        tabs = st.tabs(["Table 1", "Table 2", "Table 3", "Table 4 (Baseline)"])
        tab_keys = [
            "table1_unit_by_mpl",
            "table2_outlier_bank",
            "table3_unit_by_mpl_cleaned",
            "table4_baseline_daily_by_mpl",
        ]
        for t, k in zip(tabs, tab_keys):
            with t:
                st.dataframe(tables[k], use_container_width=True)

        # download as Excel
        xbytes = to_excel_bytes(tables)
        st.download_button(
            label="Download all tables (Excel)",
            data=xbytes,
            file_name="guardian_baseline_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)
"""
prepare_data.py
---------------
Reads raw downloads from regelleistung.net and SMARD.de,
cleans and reshapes them into the format the dashboard expects,
and saves four CSVs into the /data folder.

BEFORE RUNNING:
  1. Put your Excel file in this folder, named: regelleistung_data.xlsx
  2. Make sure it has four sheets:
       FCR_CAP_RESULT    ← from regelleistung.net FCR download
       aFRR_CAP_RESULT   ← from regelleistung.net aFRR download
       Renewable_gen     ← from SMARD.de Realisierte Erzeugung (actual generation)
       Actual_Load       ← from SMARD.de Realisierter Stromverbrauch (actual consumption)
  3. Run: python3 prepare_data.py
  4. Then run: streamlit run dashboard.py

OUTPUT FILES:
  data/fcr_tenders.csv       — daily FCR tender results
  data/afrr_tenders.csv      — 4-hour block aFRR tender results
  data/smard_renewable.csv   — daily renewable generation and share
  data/smard_load.csv        — daily grid load and residual load
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR   = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EXCEL_FILE = Path(__file__).parent / "regelleistung_data.xlsx"

# ── SMARD generation column definitions ─────────────────────────────────────

RENEWABLE_COLS = [
    "Biomasse [MWh]",
    "Wasserkraft [MWh]",
    "Wind Offshore [MWh]",
    "Wind Onshore [MWh]",
    "Photovoltaik [MWh]",
    "Sonstige Erneuerbare [MWh]",
]

NON_RENEWABLE_COLS = [
    "Kernenergie [MWh]",
    "Braunkohle [MWh]",
    "Steinkohle [MWh]",
    "Erdgas [MWh]",
    "Pumpspeicher [MWh]",
    "Sonstige Konventionelle [MWh]",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_dates(series: pd.Series) -> pd.Series:
    """Handle multiple date formats: DD.MM.YYYY, YYYY-MM-DD, DD/MM/YYYY"""
    for fmt in ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() > len(series) * 0.8:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def clean_numeric(series: pd.Series) -> pd.Series:
    """Handle European number formats: 1.234,56 → 1234.56"""
    if series.dtype == object:
        series = (series
                  .astype(str)
                  .str.replace(r"[^\d.,-]", "", regex=True)
                  .str.replace(r"\.(?=\d{3}[,\.])", "", regex=True)
                  .str.replace(",", ".", regex=False))
    return pd.to_numeric(series, errors="coerce")


def check_file(path: Path, label: str):
    if not path.exists():
        print(f"\n  ERROR: Cannot find {label}")
        print(f"  Expected location: {path}")
        print(f"  Please put the file there and run this script again.\n")
        sys.exit(1)


def check_sheet(xl: pd.ExcelFile, sheet: str) -> bool:
    if sheet not in xl.sheet_names:
        print(f"  WARNING: Sheet '{sheet}' not found. Skipping.")
        print(f"  Available sheets: {xl.sheet_names}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FCR
# ══════════════════════════════════════════════════════════════════════════════

def process_fcr(xl: pd.ExcelFile) -> pd.DataFrame:
    print("Processing FCR data...")
    if not check_sheet(xl, "FCR_CAP_RESULT"):
        return None

    raw = xl.parse("FCR_CAP_RESULT")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    raw["DATE_FROM"] = parse_dates(raw["DATE_FROM"])
    raw["DATE_TO"]   = parse_dates(raw["DATE_TO"])
    raw = raw.dropna(subset=["DATE_FROM"])

    price_col    = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    demand_col   = "GERMANY_DEMAND_[MW]"
    surplus_col  = "GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]"
    xb_price_col = "CROSSBORDER_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"

    for col in [price_col, demand_col, surplus_col, xb_price_col]:
        raw[col] = clean_numeric(raw[col])

    raw["offered_mw"] = raw[demand_col] + raw[surplus_col].fillna(0)

    fcr = pd.DataFrame({
        "tender_date":                raw["DATE_FROM"].dt.date,
        "delivery_date":              raw["DATE_TO"].dt.date,
        "awarded_mw":                 raw[demand_col].round(1),
        "offered_mw":                 raw["offered_mw"].round(1),
        "clearing_price_eur_mw_week": raw[price_col].round(4),
        "marginal_price_eur_mw_week": raw[xb_price_col].round(4),
        "num_bids":                   np.nan,
    })

    fcr = fcr.dropna(subset=["clearing_price_eur_mw_week"])
    fcr = fcr.sort_values("tender_date").reset_index(drop=True)

    print(f"  Clean rows: {len(fcr)}")
    print(f"  Date range: {fcr['tender_date'].min()} to {fcr['tender_date'].max()}")
    print(f"  Price range: {fcr['clearing_price_eur_mw_week'].min():.2f} to {fcr['clearing_price_eur_mw_week'].max():.2f} EUR/MW/week")
    return fcr


# ══════════════════════════════════════════════════════════════════════════════
# aFRR
# ══════════════════════════════════════════════════════════════════════════════

def process_afrr(xl: pd.ExcelFile) -> pd.DataFrame:
    print("\nProcessing aFRR data...")
    if not check_sheet(xl, "aFRR_CAP_RESULT"):
        return None

    raw = xl.parse("aFRR_CAP_RESULT")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")
    print(f"  TYPE_OF_RESERVES values found: {raw['TYPE_OF_RESERVES'].dropna().unique()[:8]}")

    raw["DATE_FROM"] = parse_dates(raw["DATE_FROM"])
    raw["DATE_TO"]   = parse_dates(raw["DATE_TO"])
    raw = raw.dropna(subset=["DATE_FROM"])

    def map_direction(series: pd.Series) -> np.ndarray:
        s = series.astype(str).str.upper()
        return np.where(
            s.str.contains("POS|POSITIVE|UP", regex=True), "positive",
            np.where(
                s.str.contains("NEG|NEGATIVE|DOWN", regex=True), "negative",
                None
            )
        )

    raw["direction"] = map_direction(raw["TYPE_OF_RESERVES"])

    missing = raw["direction"].isna()
    if missing.any() and "PRODUCT" in raw.columns:
        print(f"  TYPE_OF_RESERVES has no direction info — trying PRODUCT column...")
        print(f"  PRODUCT values found: {raw.loc[missing, 'PRODUCT'].dropna().unique()[:8]}")
        raw.loc[missing, "direction"] = map_direction(raw.loc[missing, "PRODUCT"])

    still_missing = raw["direction"].isna()
    if still_missing.any():
        print(f"  WARNING: Could not map these values: {raw.loc[still_missing, 'TYPE_OF_RESERVES'].unique()}")
        print(f"  These rows will be excluded.")

    raw = raw.dropna(subset=["direction"])

    avg_col      = "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
    marginal_col = "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"
    alloc_col    = "GERMANY_ALLOCATED_VOLUME_[MW]"
    offered_col  = "GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]"

    for col in [avg_col, marginal_col, alloc_col, offered_col]:
        raw[col] = clean_numeric(raw[col])

    afrr = pd.DataFrame({
        "tender_date":             raw["DATE_FROM"].dt.date,
        "delivery_week_start":     raw["DATE_FROM"].dt.date,
        "direction":               raw["direction"],
        "awarded_mw":              raw[alloc_col].round(1),
        "offered_mw":              raw[offered_col].round(1),
        "clearing_price_eur_mw_h": raw[avg_col].round(4),
        "marginal_price_eur_mw_h": raw[marginal_col].round(4),
        "num_bids":                np.nan,
    })

    afrr = afrr.dropna(subset=["clearing_price_eur_mw_h"])
    afrr = afrr.sort_values(["delivery_week_start", "direction"]).reset_index(drop=True)

    print(f"  Clean rows: {len(afrr)}")
    print(f"  Date range: {afrr['delivery_week_start'].min()} to {afrr['delivery_week_start'].max()}")
    pos = afrr[afrr["direction"] == "positive"]["clearing_price_eur_mw_h"]
    neg = afrr[afrr["direction"] == "negative"]["clearing_price_eur_mw_h"]
    print(f"  aFRR+ price range: {pos.min():.2f} to {pos.max():.2f} EUR/MW/h")
    print(f"  aFRR- price range: {neg.min():.2f} to {neg.max():.2f} EUR/MW/h")
    return afrr


# ══════════════════════════════════════════════════════════════════════════════
# SMARD — RENEWABLE GENERATION (now includes wind split)
# ══════════════════════════════════════════════════════════════════════════════

def process_smard(xl: pd.ExcelFile) -> pd.DataFrame:
    print("\nProcessing SMARD renewable generation data...")
    if not check_sheet(xl, "Renewable_gen"):
        return None

    raw = xl.parse("Renewable_gen")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    date_col = "Datum von"
    if date_col not in raw.columns:
        date_col = raw.columns[0]
        print(f"  'Datum von' not found — using first column: '{date_col}'")

    raw[date_col] = parse_dates(raw[date_col])
    raw = raw.dropna(subset=[date_col])

    all_gen_cols = RENEWABLE_COLS + NON_RENEWABLE_COLS
    for col in all_gen_cols:
        if col in raw.columns:
            raw[col] = clean_numeric(raw[col])
        else:
            print(f"  WARNING: Column not found: '{col}' — filling with 0")
            raw[col] = 0

    raw[all_gen_cols] = raw[all_gen_cols].fillna(0)

    raw["renewable_generation_mwh"] = raw[RENEWABLE_COLS].sum(axis=1)
    raw["total_generation_mwh"]     = raw[all_gen_cols].sum(axis=1)
    raw["renewable_share_pct"]      = (
        raw["renewable_generation_mwh"] / raw["total_generation_mwh"] * 100
    ).round(1)

    # Wind split — offshore and onshore separately, plus combined
    raw["wind_offshore_mwh"] = raw["Wind Offshore [MWh]"]
    raw["wind_onshore_mwh"]  = raw["Wind Onshore [MWh]"]
    raw["wind_total_mwh"]    = raw["wind_offshore_mwh"] + raw["wind_onshore_mwh"]
    raw["solar_mwh"]         = raw["Photovoltaik [MWh]"]

    # Wind share of total generation
    raw["wind_share_pct"]  = (raw["wind_total_mwh"]  / raw["total_generation_mwh"] * 100).round(1)
    raw["solar_share_pct"] = (raw["solar_mwh"] / raw["total_generation_mwh"] * 100).round(1)

    smard = pd.DataFrame({
        "date":                     raw[date_col].dt.date,
        "total_generation_mwh":     raw["total_generation_mwh"].round(0),
        "renewable_generation_mwh": raw["renewable_generation_mwh"].round(0),
        "renewable_share_pct":      raw["renewable_share_pct"],
        "wind_offshore_mwh":        raw["wind_offshore_mwh"].round(0),
        "wind_onshore_mwh":         raw["wind_onshore_mwh"].round(0),
        "wind_total_mwh":           raw["wind_total_mwh"].round(0),
        "solar_mwh":                raw["solar_mwh"].round(0),
        "wind_share_pct":           raw["wind_share_pct"],
        "solar_share_pct":          raw["solar_share_pct"],
    })

    smard = smard.dropna(subset=["renewable_share_pct"])
    smard = smard[smard["renewable_share_pct"].between(1, 99)]
    smard = smard.sort_values("date").reset_index(drop=True)

    print(f"  Clean rows: {len(smard)}")
    print(f"  Date range: {smard['date'].min()} to {smard['date'].max()}")
    print(f"  Renewable share range: {smard['renewable_share_pct'].min()}% to {smard['renewable_share_pct'].max()}%")
    print(f"  Wind share range: {smard['wind_share_pct'].min()}% to {smard['wind_share_pct'].max()}%")
    print(f"  Solar share range: {smard['solar_share_pct'].min()}% to {smard['solar_share_pct'].max()}%")
    return smard


# ══════════════════════════════════════════════════════════════════════════════
# SMARD — ACTUAL LOAD
# ══════════════════════════════════════════════════════════════════════════════

def process_load(xl: pd.ExcelFile) -> pd.DataFrame:
    print("\nProcessing SMARD actual load data...")
    if not check_sheet(xl, "Actual_Load"):
        return None

    raw = xl.parse("Actual_Load")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")
    print(f"  Columns found: {list(raw.columns)}")

    date_col = "Datum von"
    if date_col not in raw.columns:
        date_col = raw.columns[0]
        print(f"  'Datum von' not found — using first column: '{date_col}'")

    raw[date_col] = parse_dates(raw[date_col])
    raw = raw.dropna(subset=[date_col])

    # Column names from SMARD load export
    load_col     = "Netzlast [MWh]"
    residual_col = "Residuallast [MWh]"
    load_ps_col  = "Netzlast inkl. Pumpspeicher [MWh]"

    for col in [load_col, residual_col, load_ps_col]:
        if col in raw.columns:
            raw[col] = clean_numeric(raw[col])
        else:
            print(f"  WARNING: Column not found: '{col}' — filling with NaN")
            raw[col] = np.nan

    load = pd.DataFrame({
        "date":                      raw[date_col].dt.date,
        "grid_load_mwh":             raw[load_col].round(0),
        "grid_load_incl_pumped_mwh": raw[load_ps_col].round(0),
        "residual_load_mwh":         raw[residual_col].round(0),
    })

    load = load.dropna(subset=["grid_load_mwh"])
    load = load.sort_values("date").reset_index(drop=True)

    print(f"  Clean rows: {len(load)}")
    print(f"  Date range: {load['date'].min()} to {load['date'].max()}")
    print(f"  Grid load range: {load['grid_load_mwh'].min():,.0f} to {load['grid_load_mwh'].max():,.0f} MWh")
    print(f"  Residual load range: {load['residual_load_mwh'].min():,.0f} to {load['residual_load_mwh'].max():,.0f} MWh")
    return load


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  German Balancing Market — Data Preparation")
    print("=" * 60)

    check_file(EXCEL_FILE, "regelleistung_data.xlsx")

    xl = pd.ExcelFile(EXCEL_FILE)
    print(f"\nSheets found in Excel file: {xl.sheet_names}")

    fcr   = process_fcr(xl)
    afrr  = process_afrr(xl)
    smard = process_smard(xl)
    load  = process_load(xl)

    print("\n" + "=" * 60)
    print("  Saving outputs...")
    print("=" * 60)

    if fcr is not None:
        fcr.to_csv(DATA_DIR / "fcr_tenders.csv", index=False)
        print(f"  Saved → data/fcr_tenders.csv  ({len(fcr)} rows)")

    if afrr is not None:
        afrr.to_csv(DATA_DIR / "afrr_tenders.csv", index=False)
        print(f"  Saved → data/afrr_tenders.csv  ({len(afrr)} rows)")

    if smard is not None:
        smard.to_csv(DATA_DIR / "smard_renewable.csv", index=False)
        print(f"  Saved → data/smard_renewable.csv  ({len(smard)} rows)")

    if load is not None:
        load.to_csv(DATA_DIR / "smard_load.csv", index=False)
        print(f"  Saved → data/smard_load.csv  ({len(load)} rows)")

    print("\n" + "=" * 60)
    print("  Done. Run: streamlit run dashboard.py")
    print("=" * 60)
"""
prepare_data.py
---------------
Reads the raw Excel download (regelleistung_data.xlsx with 4 sheets)
and produces four clean CSVs for the dashboard.

Input sheets:
    aFRR_CAP_RESULT  : 12 rows/day (6 blocks x 2 directions)
    FCR_CAP_RESULT   : 1 row/day
    Renewable_gen    : 1 row/day (SMARD generation by source)
    Actual_Load      : 1 row/day (SMARD load data)

Output files:
    data/afrr_tenders.csv    : one row per 4-hour block per direction
    data/fcr_tenders.csv     : one row per day
    data/smard_renewable.csv : one row per day
    data/smard_load.csv      : one row per day

Run:  python3 prepare_data.py
Then: streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EXCEL_FILE = Path(__file__).parent / "regelleistung_data.xlsx"

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_numeric(s: pd.Series) -> pd.Series:
    """Convert European-formatted strings (1.234,56) to float."""
    if s.dtype == object:
        s = (s.astype(str)
             .str.replace(r"[^\d.,-]", "", regex=True)     # strip stray chars
             .str.replace(r"\.(?=\d{3}[,.])", "", regex=True)  # remove thousands dots
             .str.replace(",", ".", regex=False))           # decimal comma → dot
    return pd.to_numeric(s, errors="coerce")


def parse_dates(s: pd.Series) -> pd.Series:
    """Try common date formats, fall back to dayfirst."""
    for fmt in ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(s) * 0.8:
            return parsed
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def check_sheet(xl: pd.ExcelFile, name: str) -> bool:
    if name not in xl.sheet_names:
        print(f"  WARNING: Sheet '{name}' not found. Available: {xl.sheet_names}")
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# 1. aFRR — one row per 4-hour block per direction
# ═════════════════════════════════════════════════════════════════════════════

def process_afrr(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    print("\n── aFRR ──")
    if not check_sheet(xl, "aFRR_CAP_RESULT"):
        return None

    raw = xl.parse("aFRR_CAP_RESULT")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    # Parse delivery date from DATE_FROM
    raw["delivery_date"] = pd.to_datetime(raw["DATE_FROM"]).dt.normalize()

    # Parse direction and block hours from PRODUCT  (e.g. POS_00_04, NEG_16_20)
    product_parts = raw["PRODUCT"].str.extract(r"(POS|NEG)_(\d{2})_(\d{2})")
    raw["direction"] = product_parts[0].map({"POS": "positive", "NEG": "negative"})
    raw["block_hour"] = product_parts[1].astype(int)

    # Build block_start as an actual datetime
    raw["block_start"] = raw["delivery_date"] + pd.to_timedelta(raw["block_hour"], unit="h")

    # Extract Germany-specific columns
    price_col = "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"
    avg_price_col = "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
    awarded_col = "GERMANY_ALLOCATED_VOLUME_[MW]"
    offered_col = "GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]"

    # aFRR capacity is pay-as-bid (not pay-as-cleared like FCR).
    # Average price = volume-weighted mean of all accepted bids = primary analysis column.
    #   Represents broad market tightness. When this is high, the whole stack shifted up.
    # Marginal price = highest accepted bid = kept for reference only.
    #   Noisy in pay-as-bid: one outlier bid can spike it without reflecting real scarcity.
    out = pd.DataFrame({
        "delivery_date":          raw["delivery_date"],
        "block_start":            raw["block_start"],
        "block_hour":             raw["block_hour"],
        "direction":              raw["direction"],
        "avg_price_eur_mw_h":      clean_numeric(raw[avg_price_col]),
        "marginal_price_eur_mw_h": clean_numeric(raw[price_col]),
        "awarded_mw":             clean_numeric(raw[awarded_col]),
        "offered_mw":             clean_numeric(raw[offered_col]),
    })

    out = out.dropna(subset=["delivery_date", "direction"])
    out = out.sort_values(["delivery_date", "block_hour", "direction"]).reset_index(drop=True)

    n_days = out["delivery_date"].nunique()
    n_pos = (out["direction"] == "positive").sum()
    n_neg = (out["direction"] == "negative").sum()
    print(f"  Output: {len(out)} rows  ({n_days} days × 6 blocks × 2 directions)")
    print(f"  Positive: {n_pos}  |  Negative: {n_neg}")
    print(f"  Date range: {out['delivery_date'].min().date()} → {out['delivery_date'].max().date()}")
    print(f"  Price range (avg, primary): {out['avg_price_eur_mw_h'].min():.2f} – {out['avg_price_eur_mw_h'].max():.2f} EUR/MW/h")
    print(f"  Price range (marginal, ref): {out['marginal_price_eur_mw_h'].min():.2f} – {out['marginal_price_eur_mw_h'].max():.2f} EUR/MW/h")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. FCR — one row per day
# ═════════════════════════════════════════════════════════════════════════════

def process_fcr(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    print("\n── FCR ──")
    if not check_sheet(xl, "FCR_CAP_RESULT"):
        return None

    raw = xl.parse("FCR_CAP_RESULT")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    # Filter to tender 1 only — tender 2 is a supplementary cross-border round
    # with empty Germany columns, not useful for our analysis.
    tender_col = "TENDER_NUMBER"
    if tender_col in raw.columns:
        raw[tender_col] = clean_numeric(raw[tender_col])
        tender2_count = (raw[tender_col] == 2).sum()
        raw = raw[raw[tender_col] == 1].copy()
        print(f"  Filtered to tender 1 (dropped {tender2_count} tender 2 rows)")

    raw["delivery_date"] = pd.to_datetime(raw["DATE_FROM"]).dt.normalize()

    # Parse block hours from PRODUCT column (NEGPOS_00_04, etc.)
    # FCR column is PRODUCTNAME (not PRODUCT like aFRR)
    product_col = "PRODUCTNAME" if "PRODUCTNAME" in raw.columns else "PRODUCT"
    product_parts = raw[product_col].str.extract(r"(\d{2})_(\d{2})")
    raw["block_hour"] = product_parts[0].astype(int)
    raw["block_start"] = raw["delivery_date"] + pd.to_timedelta(raw["block_hour"], unit="h")

    # FCR is pay-as-cleared — settlement price is what every provider gets paid.
    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    demand_col = "GERMANY_DEMAND_[MW]"
    surplus_col = "GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]"

    out = pd.DataFrame({
        "delivery_date":              raw["delivery_date"],
        "block_start":                raw["block_start"],
        "block_hour":                 raw["block_hour"],
        "clearing_price_eur_mw_week": clean_numeric(raw[price_col]),
        "demand_mw":                  clean_numeric(raw[demand_col]),
        "surplus_mw":                 clean_numeric(raw[surplus_col]),
    })

    out = out.dropna(subset=["delivery_date"])
    out = out.sort_values(["delivery_date", "block_hour"]).reset_index(drop=True)

    n_days = out["delivery_date"].nunique()
    print(f"  Output: {len(out)} rows  ({n_days} days × 6 blocks)")
    print(f"  Date range: {out['delivery_date'].min().date()} → {out['delivery_date'].max().date()}")
    print(f"  Price range: {out['clearing_price_eur_mw_week'].min():.2f} – {out['clearing_price_eur_mw_week'].max():.2f} EUR/MW/week")
    print(f"  Surplus range: {out['surplus_mw'].min():.0f} – {out['surplus_mw'].max():.0f} MW")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. SMARD Renewable Generation — one row per day
# ═════════════════════════════════════════════════════════════════════════════

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

def process_renewable(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    print("\n── SMARD Renewable Generation ──")
    if not check_sheet(xl, "Renewable_gen"):
        return None

    raw = xl.parse("Renewable_gen")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    raw["date"] = parse_dates(raw["Datum von"] if "Datum von" in raw.columns else raw["Datum"])

    # Clean all generation columns
    all_gen_cols = RENEWABLE_COLS + NON_RENEWABLE_COLS
    for col in all_gen_cols:
        if col in raw.columns:
            raw[col] = clean_numeric(raw[col])

    # Compute totals
    ren_cols_present = [c for c in RENEWABLE_COLS if c in raw.columns]
    non_ren_cols_present = [c for c in NON_RENEWABLE_COLS if c in raw.columns]

    raw["renewable_generation_mwh"] = raw[ren_cols_present].sum(axis=1)
    raw["non_renewable_mwh"] = raw[non_ren_cols_present].sum(axis=1)
    raw["total_generation_mwh"] = raw["renewable_generation_mwh"] + raw["non_renewable_mwh"]

    # Wind and solar
    wind_off = raw.get("Wind Offshore [MWh]", 0)
    wind_on = raw.get("Wind Onshore [MWh]", 0)
    raw["wind_total_mwh"] = wind_off + wind_on
    raw["solar_mwh"] = raw.get("Photovoltaik [MWh]", 0)

    # Shares (avoid division by zero)
    total = raw["total_generation_mwh"].replace(0, np.nan)
    raw["renewable_share_pct"] = (raw["renewable_generation_mwh"] / total * 100).round(2)
    raw["wind_share_pct"] = (raw["wind_total_mwh"] / total * 100).round(2)
    raw["solar_share_pct"] = (raw["solar_mwh"] / total * 100).round(2)

    out = raw[["date", "total_generation_mwh", "renewable_generation_mwh",
               "renewable_share_pct", "wind_total_mwh", "solar_mwh",
               "wind_share_pct", "solar_share_pct"]].copy()

    # Also include offshore/onshore split for deeper analysis
    if "Wind Offshore [MWh]" in raw.columns:
        out["wind_offshore_mwh"] = raw["Wind Offshore [MWh]"]
    if "Wind Onshore [MWh]" in raw.columns:
        out["wind_onshore_mwh"] = raw["Wind Onshore [MWh]"]

    out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)

    print(f"  Output: {len(out)} rows")
    print(f"  Date range: {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"  Renewable share: {out['renewable_share_pct'].min():.1f}% – {out['renewable_share_pct'].max():.1f}%")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. SMARD Load — one row per day
# ═════════════════════════════════════════════════════════════════════════════

def process_load(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    print("\n── SMARD Load ──")
    if not check_sheet(xl, "Actual_Load"):
        return None

    raw = xl.parse("Actual_Load")
    raw.columns = raw.columns.str.strip()
    print(f"  Raw rows: {len(raw)}")

    raw["date"] = parse_dates(raw["Datum von"])

    out = pd.DataFrame({
        "date":              raw["date"],
        "grid_load_mwh":     clean_numeric(raw["Netzlast [MWh]"]),
        "residual_load_mwh": clean_numeric(raw["Residuallast [MWh]"]),
    })

    out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)

    print(f"  Output: {len(out)} rows")
    print(f"  Date range: {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"  Residual load range: {out['residual_load_mwh'].min():.0f} – {out['residual_load_mwh'].max():.0f} MWh")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if not EXCEL_FILE.exists():
        print(f"\n  ERROR: Cannot find {EXCEL_FILE}")
        print(f"  Place regelleistung_data.xlsx in this folder and run again.\n")
        sys.exit(1)

    xl = pd.ExcelFile(EXCEL_FILE)
    print(f"Loaded: {EXCEL_FILE.name}")
    print(f"Sheets found: {xl.sheet_names}")

    processors = {
        "afrr_tenders.csv":    process_afrr,
        "fcr_tenders.csv":     process_fcr,
        "smard_renewable.csv": process_renewable,
        "smard_load.csv":      process_load,
    }

    for filename, func in processors.items():
        df = func(xl)
        if df is not None:
            path = DATA_DIR / filename
            df.to_csv(path, index=False)
            print(f"  ✓ Saved → {path}")

    print("\n── Done ──")
    print("Run:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
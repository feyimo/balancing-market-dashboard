"""
dashboard.py
------------
When Renewables Create Scarcity: Identifying High-Value Windows for VPP Operators

Five analytical views answering:
  "When does the German grid most urgently need more power at short notice
   — and can you see it coming?"

Data: regelleistung.net (FCR + aFRR tenders) + SMARD.de (generation + load)
Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VPP High-Value Windows — German Balancing Markets",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ───────────────────────────────────────────────────────────

TEAL      = "#0F6E56"
TEAL_L    = "#14A37F"
TEAL_BG   = "#E8F5F1"
AMBER     = "#E08C00"
AMBER_L   = "#F5A623"
RED       = "#C0392B"
SLATE     = "#3D4A5C"
GREY_LINE = "#DDE2EA"
BLACK     = "#1A1A1A"

SEASON_COLOURS = {
    "Winter": "#5B8DB8",
    "Spring": TEAL_L,
    "Summer": AMBER,
    "Autumn": "#C0722A",
}

# ── shared layout helper ─────────────────────────────────────────────────────

def apply_base_layout(fig, height=380, title=""):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Calibri, Arial, sans-serif", color=BLACK, size=12),
        title_font=dict(color=BLACK, size=13),
        legend=dict(orientation="h", y=1.08, font=dict(color=BLACK, size=11)),
        legend_title_text="",
        height=height,
        margin=dict(t=55, b=40, l=10, r=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK))
    fig.update_yaxes(gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK))
    return fig


# ── load and prepare data ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"

    fcr   = pd.read_csv(base / "fcr_tenders.csv",     parse_dates=["tender_date", "delivery_date"])
    afrr  = pd.read_csv(base / "afrr_tenders.csv",    parse_dates=["tender_date", "delivery_week_start", "block_start"])
    smard = pd.read_csv(base / "smard_renewable.csv", parse_dates=["date"])
    load  = pd.read_csv(base / "smard_load.csv",      parse_dates=["date"])

    # ── Build aFRR spread table (one row per time block) ──
    afrr_pivot = afrr.pivot_table(
        index=["tender_date", "delivery_week_start"],
        columns="direction",
        values="clearing_price_eur_mw_h",
        aggfunc="mean",
    ).reset_index()
    afrr_pivot.columns.name = None
    if "positive" in afrr_pivot.columns and "negative" in afrr_pivot.columns:
        afrr_pivot["spread"] = afrr_pivot["positive"] - afrr_pivot["negative"]
    else:
        afrr_pivot["spread"] = np.nan

    # ── Merge generation and load onto spread table ──
    smard["merge_date"] = smard["date"].dt.date
    load["merge_date"]  = load["date"].dt.date
    afrr_pivot["merge_date"] = afrr_pivot["delivery_week_start"].dt.date
    fcr["merge_date"] = fcr["delivery_date"].dt.date

    afrr_pivot = afrr_pivot.merge(
        smard[["merge_date", "renewable_share_pct", "wind_share_pct",
               "solar_share_pct", "wind_total_mwh", "solar_mwh"]],
        on="merge_date", how="left"
    )
    afrr_pivot = afrr_pivot.merge(
        load[["merge_date", "grid_load_mwh", "residual_load_mwh"]],
        on="merge_date", how="left"
    )

    # Daily avg FCR price merged onto spread table
    fcr_daily = fcr.groupby("merge_date")["clearing_price_eur_mw_week"].mean().reset_index()
    fcr_daily.columns = ["merge_date", "fcr_price"]
    afrr_pivot = afrr_pivot.merge(fcr_daily, on="merge_date", how="left")

    # ── Tag high-asymmetry events (top 10% spread) ──
    threshold = afrr_pivot["spread"].quantile(0.90)
    afrr_pivot["high_asymmetry"] = afrr_pivot["spread"] >= threshold

    # ── Time features ──
    afrr_pivot["month"]      = afrr_pivot["delivery_week_start"].dt.month
    afrr_pivot["month_name"] = afrr_pivot["delivery_week_start"].dt.strftime("%b")
    afrr_pivot["season"]     = afrr_pivot["month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Spring",  4:"Spring",  5:"Spring",
        6:"Summer",  7:"Summer",  8:"Summer",
        9:"Autumn",  10:"Autumn", 11:"Autumn",
    })

    # ── Merge smard onto FCR for correlation view ──
    smard_for_fcr = smard[["merge_date", "renewable_share_pct", "wind_share_pct", "solar_share_pct"]].copy()
    fcr = fcr.merge(smard_for_fcr, on="merge_date", how="left")
    fcr = fcr.merge(load[["merge_date", "grid_load_mwh", "residual_load_mwh"]], on="merge_date", how="left")

    return fcr, afrr, afrr_pivot, smard, load, threshold


fcr, afrr, afrr_pivot, smard, load, threshold_90 = load_data()

# ── Filtered subsets ─────────────────────────────────────────────────────────

afrr_pos = afrr[afrr["direction"] == "positive"]
afrr_neg = afrr[afrr["direction"] == "negative"]
high_events = afrr_pivot[afrr_pivot["high_asymmetry"]]
normal_events = afrr_pivot[~afrr_pivot["high_asymmetry"]]

# ── sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Filters")

min_date = afrr_pivot["delivery_week_start"].min().date()
max_date = afrr_pivot["delivery_week_start"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = date_range
else:
    d_start, d_end = min_date, max_date

afrr_f   = afrr_pivot[(afrr_pivot["delivery_week_start"].dt.date >= d_start) & (afrr_pivot["delivery_week_start"].dt.date <= d_end)]
fcr_f    = fcr[(fcr["tender_date"].dt.date >= d_start) & (fcr["tender_date"].dt.date <= d_end)]
smard_f  = smard[(smard["date"].dt.date >= d_start) & (smard["date"].dt.date <= d_end)]
load_f   = load[(load["date"].dt.date >= d_start) & (load["date"].dt.date <= d_end)]

high_f   = afrr_f[afrr_f["high_asymmetry"]]
normal_f = afrr_f[~afrr_f["high_asymmetry"]]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- [regelleistung.net](https://www.regelleistung.net) — FCR & aFRR tender results\n"
    "- [SMARD.de](https://www.smard.de) — Generation and load data (Bundesnetzagentur)\n"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**High-asymmetry threshold**\n\n"
    f"Top 10% of spread values: **≥ {threshold_90:.2f} EUR/MW/h**\n\n"
    f"Events in selected period: **{len(high_f)}** of {len(afrr_f)} time blocks"
)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <h1 style='color:{TEAL}; margin-bottom:0'>
        When Renewables Create Scarcity
    </h1>
    <p style='color:{TEAL}; font-size:1.1rem; font-weight:600; margin-top:4px; margin-bottom:2px'>
        Identifying High-Value Windows for VPP Operators
    </p>
    <p style='color:rgba(255,255,255,0.6); font-size:0.95rem; margin-top:0'>
        German FCR & aFRR balancing markets &nbsp;·&nbsp;
        {d_start.strftime('%d %b %Y')} to {d_end.strftime('%d %b %Y')}
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────

avg_fcr        = fcr_f["clearing_price_eur_mw_week"].mean()
peak_fcr       = fcr_f["clearing_price_eur_mw_week"].max()
avg_afrr_pos   = afrr_f["positive"].mean() if "positive" in afrr_f.columns else np.nan
avg_spread     = afrr_f["spread"].mean()
n_high         = len(high_f)
avg_ren        = smard_f["renewable_share_pct"].mean()

def kpi_card(col, label, value, unit, fmt=".1f"):
    col.markdown(
        f"""
        <div style='padding:10px 0 6px 0;'>
            <div style='font-size:0.78rem; color:rgba(255,255,255,0.55); font-weight:600;
                        letter-spacing:0.03em; margin-bottom:3px; text-transform:uppercase'>{label}</div>
            <div style='font-size:1.85rem; font-weight:700; color:#ffffff; line-height:1.1'>{value:{fmt}}</div>
            <div style='font-size:0.78rem; color:rgba(255,255,255,0.45); margin-top:2px'>{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_card(k1, "Avg FCR Price",          avg_fcr,      "€ / MW / week")
kpi_card(k2, "FCR Price Peak",         peak_fcr,     "€ / MW / week")
kpi_card(k3, "Avg aFRR+ Price",        avg_afrr_pos, "€ / MW / h")
kpi_card(k4, "Avg Spread (aFRR+ − −)", avg_spread,   "€ / MW / h")
kpi_card(k5, "High-Asymmetry Events",  n_high,       "time blocks", fmt=".0f")
kpi_card(k6, "Avg Renewable Share",    avg_ren,       "%")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 1 — HOW URGENT DOES IT GET?
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### How Urgent Does It Get?")
st.caption(
    "The scale of the premium — how much more expensive does upward flexibility become "
    "during high-stress events, and when did the largest spikes occur?"
)

c1a, c1b = st.columns([3, 2])

# aFRR+ price over time — one point per 4-hour block, x-axis is exact block datetime
afrr_pos_ts = afrr[afrr["direction"] == "positive"].copy()
afrr_pos_ts = afrr_pos_ts.sort_values("block_start")
afrr_pos_ts["rolling"] = afrr_pos_ts["clearing_price_eur_mw_h"].rolling(30, min_periods=1).mean()

fig1a = go.Figure()
fig1a.add_trace(go.Scatter(
    x=afrr_pos_ts["block_start"], y=afrr_pos_ts["clearing_price_eur_mw_h"],
    name="aFRR+ price (4h block)",
    mode="lines", line=dict(color=GREY_LINE, width=1),
    hovertemplate="%{x|%d %b %Y %H:%M}<br>Price: €%{y:.2f}/MW/h<extra></extra>",
))
fig1a.add_trace(go.Scatter(
    x=afrr_pos_ts["block_start"], y=afrr_pos_ts["rolling"],
    name="30-block rolling avg",
    mode="lines", line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%d %b %Y %H:%M}<br>Avg: €%{y:.2f}/MW/h<extra></extra>",
))
fig1a.update_layout(
    title="aFRR+ (upward regulation) clearing price — each point is one 4-hour block",
    xaxis_title="Block start time",
    yaxis_title="Clearing price (€/MW/h)",
    hovermode="closest",
)
apply_base_layout(fig1a, height=360)
c1a.plotly_chart(fig1a, use_container_width=True)

# Premium comparison bar
if len(high_f) > 0 and len(normal_f) > 0:
    premium_data = pd.DataFrame({
        "Period": ["High-asymmetry events", "All other periods"],
        "avg_price": [high_f["positive"].mean(), normal_f["positive"].mean()],
        "colour": [AMBER, TEAL],
    })
    fig1b = go.Figure()
    for _, row in premium_data.iterrows():
        fig1b.add_trace(go.Bar(
            x=[row["Period"]], y=[row["avg_price"]],
            name=row["Period"],
            marker_color=row["colour"],
            text=[f"€{row['avg_price']:.1f}"],
            textposition="outside",
            textfont=dict(color=BLACK, size=12),
        ))
    multiple = high_f["positive"].mean() / normal_f["positive"].mean()
    fig1b.update_layout(
        title=f"aFRR+ price: high-asymmetry vs baseline ({multiple:.1f}x premium)",
        yaxis_title="Avg clearing price (€/MW/h)",
        showlegend=False,
        bargap=0.4,
    )
    apply_base_layout(fig1b, height=360)
    c1b.plotly_chart(fig1b, use_container_width=True)

# FCR price tracker underneath
fcr_f_sorted = fcr_f.sort_values("tender_date").copy()
fcr_f_sorted["rolling_30d"] = fcr_f_sorted["clearing_price_eur_mw_week"].rolling(30, min_periods=1).mean()

fig1c = go.Figure()
fig1c.add_trace(go.Scatter(
    x=fcr_f_sorted["tender_date"], y=fcr_f_sorted["clearing_price_eur_mw_week"],
    name="Daily FCR price", mode="lines",
    line=dict(color=GREY_LINE, width=1),
    hovertemplate="%{x|%d %b %Y}<br>€%{y:.2f}/MW/wk<extra></extra>",
))
fig1c.add_trace(go.Scatter(
    x=fcr_f_sorted["tender_date"], y=fcr_f_sorted["rolling_30d"],
    name="30-day rolling avg", mode="lines",
    line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>Avg: €%{y:.2f}/MW/wk<extra></extra>",
))
fig1c.update_layout(
    title="FCR clearing price over time (supporting indicator)",
    xaxis_title="Tender date",
    yaxis_title="Clearing price (€/MW/week)",
)
apply_base_layout(fig1c, height=300)
st.plotly_chart(fig1c, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 2 — WHEN DOES URGENCY PEAK?
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### When Does Urgency Peak?")
st.caption(
    "The seasonal pattern of upward flexibility scarcity — "
    "when do high-asymmetry events cluster, and how consistent is the pattern across both years?"
)

c2a, c2b = st.columns([1, 1])

# Monthly average spread
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_spread = (afrr_pivot.groupby(["month", "month_name"])["spread"]
                  .mean().reset_index()
                  .sort_values("month"))
monthly_spread["colour"] = monthly_spread["month"].map({
    12: SEASON_COLOURS["Winter"], 1: SEASON_COLOURS["Winter"], 2: SEASON_COLOURS["Winter"],
    3: SEASON_COLOURS["Spring"],  4: SEASON_COLOURS["Spring"],  5: SEASON_COLOURS["Spring"],
    6: SEASON_COLOURS["Summer"],  7: SEASON_COLOURS["Summer"],  8: SEASON_COLOURS["Summer"],
    9: SEASON_COLOURS["Autumn"], 10: SEASON_COLOURS["Autumn"], 11: SEASON_COLOURS["Autumn"],
})

fig2a = go.Figure()
fig2a.add_trace(go.Bar(
    x=monthly_spread["month_name"],
    y=monthly_spread["spread"],
    marker_color=monthly_spread["colour"],
    hovertemplate="%{x}<br>Avg spread: €%{y:.2f}/MW/h<extra></extra>",
    name="",
))
fig2a.add_hline(y=0, line_dash="dash", line_color=GREY_LINE, line_width=1.5)
fig2a.update_layout(
    title="Average aFRR price spread by month (aFRR+ minus aFRR–)",
    xaxis_title="Month",
    yaxis_title="Avg spread (€/MW/h)",
    showlegend=False,
    xaxis=dict(categoryorder="array", categoryarray=month_order),
)
apply_base_layout(fig2a, height=360)
c2a.plotly_chart(fig2a, use_container_width=True)

# High-asymmetry event timeline
if len(high_events) > 0:
    high_events_plot = high_events.copy()
    high_events_plot["season_colour"] = high_events_plot["season"].map(SEASON_COLOURS)

    fig2b = go.Figure()
    for season, colour in SEASON_COLOURS.items():
        subset = high_events_plot[high_events_plot["season"] == season]
        if len(subset) == 0:
            continue
        fig2b.add_trace(go.Scatter(
            x=subset["delivery_week_start"],
            y=subset["spread"],
            mode="markers",
            name=season,
            marker=dict(color=colour, size=12, symbol="circle",
                        line=dict(color="white", width=1.5)),
            hovertemplate="%{x|%d %b %Y}<br>Spread: €%{y:.2f}/MW/h<extra></extra>",
        ))
    fig2b.add_hline(
        y=threshold_90, line_dash="dot", line_color=RED, line_width=1.5,
        annotation_text="90th percentile threshold",
        annotation_position="top right",
        annotation_font=dict(color=RED, size=10),
    )
    fig2b.update_layout(
        title="High-asymmetry events across the dataset (top 10% spread weeks)",
        xaxis_title="Delivery date",
        yaxis_title="Spread (€/MW/h)",
    )
    apply_base_layout(fig2b, height=360)
    c2b.plotly_chart(fig2b, use_container_width=True)

# aFRR spread over full time period
afrr_spread_ts = afrr_pivot.sort_values("delivery_week_start").copy()
afrr_spread_ts["spread_rolling"] = afrr_spread_ts["spread"].rolling(30, min_periods=1).mean()

fig2c = go.Figure()
fig2c.add_trace(go.Scatter(
    x=afrr_spread_ts["delivery_week_start"], y=afrr_spread_ts["spread"],
    name="aFRR+ minus aFRR– price",
    mode="lines", line=dict(color=GREY_LINE, width=1),
    hovertemplate="%{x|%d %b %Y}<br>Spread: €%{y:.2f}/MW/h<extra></extra>",
))
fig2c.add_trace(go.Scatter(
    x=afrr_spread_ts["delivery_week_start"], y=afrr_spread_ts["spread_rolling"],
    name="30-period rolling avg",
    mode="lines", line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>Avg: €%{y:.2f}/MW/h<extra></extra>",
))
fig2c.add_hline(
    y=0, line_dash="dash", line_color=AMBER, line_width=1.5,
    annotation_text="Parity (aFRR+ = aFRR–)",
    annotation_position="top left",
    annotation_font=dict(color=AMBER, size=11),
)
fig2c.update_layout(
    title="aFRR price spread over time: upward minus downward regulation",
    xaxis_title="Delivery date",
    yaxis_title="Price spread (€/MW/h)",
)
apply_base_layout(fig2c, height=300)
st.plotly_chart(fig2c, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 3 — WHAT CAUSES IT?
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### What Causes It?")
st.caption(
    "The driving conditions — how do renewable share, wind output, and residual load "
    "relate to moments when upward flexibility becomes scarce and expensive?"
)

c3a, c3b, c3c = st.columns(3)

# Scatter: spread vs renewable share
corr_fcr = afrr_pivot[["renewable_share_pct", "spread"]].dropna()
r_ren = corr_fcr.corr().iloc[0, 1]

fig3a = px.scatter(
    corr_fcr, x="renewable_share_pct", y="spread",
    trendline="ols",
    title=f"Spread vs renewable share (r = {r_ren:.2f})",
    labels={"renewable_share_pct": "Renewable share (%)", "spread": "Spread (€/MW/h)"},
    color_discrete_sequence=[TEAL],
)
fig3a.update_traces(marker=dict(opacity=0.4, size=5), selector=dict(mode="markers"))
fig3a.data[1].line.color = RED
apply_base_layout(fig3a, height=340)
c3a.plotly_chart(fig3a, use_container_width=True)

# Scatter: spread vs wind share
corr_wind = afrr_pivot[["wind_share_pct", "spread"]].dropna()
r_wind = corr_wind.corr().iloc[0, 1]

fig3b = px.scatter(
    corr_wind, x="wind_share_pct", y="spread",
    trendline="ols",
    title=f"Spread vs wind share (r = {r_wind:.2f})",
    labels={"wind_share_pct": "Wind share of generation (%)", "spread": "Spread (€/MW/h)"},
    color_discrete_sequence=[TEAL_L],
)
fig3b.update_traces(marker=dict(opacity=0.4, size=5), selector=dict(mode="markers"))
fig3b.data[1].line.color = RED
apply_base_layout(fig3b, height=340)
c3b.plotly_chart(fig3b, use_container_width=True)

# Scatter: spread vs residual load
corr_res = afrr_pivot[["residual_load_mwh", "spread"]].dropna()
r_res = corr_res.corr().iloc[0, 1]
corr_res_plot = corr_res.copy()
corr_res_plot["residual_load_twh"] = corr_res_plot["residual_load_mwh"] / 1_000_000

fig3c = px.scatter(
    corr_res_plot, x="residual_load_twh", y="spread",
    trendline="ols",
    title=f"Spread vs residual load (r = {r_res:.2f})",
    labels={"residual_load_twh": "Residual load (TWh)", "spread": "Spread (€/MW/h)"},
    color_discrete_sequence=[AMBER],
)
fig3c.update_traces(marker=dict(opacity=0.4, size=5), selector=dict(mode="markers"))
fig3c.data[1].line.color = RED
apply_base_layout(fig3c, height=340)
c3c.plotly_chart(fig3c, use_container_width=True)

# Conditions profile: high vs normal
if len(high_f) > 0 and len(normal_f) > 0:
    indicators = {
        "Renewable share (%)": ("renewable_share_pct", 1),
        "Wind share (%)": ("wind_share_pct", 1),
        "Solar share (%)": ("solar_share_pct", 1),
        "Residual load (TWh)": ("residual_load_mwh", 1_000_000),
    }

    profile_rows = []
    for label, (col, divisor) in indicators.items():
        h_val = high_f[col].mean() / divisor if col in high_f.columns else np.nan
        n_val = normal_f[col].mean() / divisor if col in normal_f.columns else np.nan
        profile_rows.append({"Indicator": label, "High-asymmetry": h_val, "All other periods": n_val})

    profile_df = pd.DataFrame(profile_rows)

    fig3d = go.Figure()
    fig3d.add_trace(go.Bar(
        name="High-asymmetry events",
        x=profile_df["Indicator"],
        y=profile_df["High-asymmetry"],
        marker_color=AMBER,
    ))
    fig3d.add_trace(go.Bar(
        name="All other periods",
        x=profile_df["Indicator"],
        y=profile_df["All other periods"],
        marker_color=TEAL,
    ))
    fig3d.update_layout(
        title="Conditions profile: high-asymmetry events vs all other periods",
        barmode="group",
        yaxis_title="Average value",
    )
    apply_base_layout(fig3d, height=320)
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 4 — CAN YOU SEE IT COMING?
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### Can You See It Coming?")
st.caption(
    "Leading indicator analysis — do conditions in the days before a high-asymmetry event "
    "differ from normal periods in ways that could provide advance warning?"
)

# Day-before conditions: merge previous day's generation/load onto each event
afrr_pivot_sorted = afrr_pivot.sort_values("delivery_week_start").copy()
afrr_pivot_sorted["prev_ren_share"] = afrr_pivot_sorted["renewable_share_pct"].shift(1)
afrr_pivot_sorted["prev_wind_share"] = afrr_pivot_sorted["wind_share_pct"].shift(1)
afrr_pivot_sorted["prev_residual_load"] = afrr_pivot_sorted["residual_load_mwh"].shift(1)

high_lead  = afrr_pivot_sorted[afrr_pivot_sorted["high_asymmetry"]]
norm_lead  = afrr_pivot_sorted[~afrr_pivot_sorted["high_asymmetry"]]

c4a, c4b = st.columns([1, 1])

# Previous day renewable share distribution
if len(high_lead) > 0 and len(norm_lead) > 0:
    fig4a = go.Figure()
    fig4a.add_trace(go.Box(
        y=high_lead["prev_ren_share"].dropna(),
        name="Before high-asymmetry",
        marker_color=AMBER, boxmean=True,
    ))
    fig4a.add_trace(go.Box(
        y=norm_lead["prev_ren_share"].dropna(),
        name="Before normal periods",
        marker_color=TEAL, boxmean=True,
    ))
    fig4a.update_layout(
        title="Prior day renewable share: before high-asymmetry vs normal",
        yaxis_title="Renewable share (%)",
        showlegend=True,
    )
    apply_base_layout(fig4a, height=360)
    c4a.plotly_chart(fig4a, use_container_width=True)

    # Previous day wind share distribution
    fig4b = go.Figure()
    fig4b.add_trace(go.Box(
        y=high_lead["prev_wind_share"].dropna(),
        name="Before high-asymmetry",
        marker_color=AMBER, boxmean=True,
    ))
    fig4b.add_trace(go.Box(
        y=norm_lead["prev_wind_share"].dropna(),
        name="Before normal periods",
        marker_color=TEAL, boxmean=True,
    ))
    fig4b.update_layout(
        title="Prior day wind share: before high-asymmetry vs normal",
        yaxis_title="Wind share (%)",
        showlegend=True,
    )
    apply_base_layout(fig4b, height=360)
    c4b.plotly_chart(fig4b, use_container_width=True)

# Simple rule derivation
if len(high_lead) > 0:
    ren_threshold  = afrr_pivot_sorted["renewable_share_pct"].quantile(0.60)
    wind_threshold = afrr_pivot_sorted["wind_share_pct"].quantile(0.60)

    afrr_pivot_sorted["flag"] = (
        (afrr_pivot_sorted["renewable_share_pct"] >= ren_threshold) &
        (afrr_pivot_sorted["wind_share_pct"] >= wind_threshold)
    )

    flagged       = afrr_pivot_sorted[afrr_pivot_sorted["flag"]]
    not_flagged   = afrr_pivot_sorted[~afrr_pivot_sorted["flag"]]
    precision     = flagged["high_asymmetry"].mean() * 100 if len(flagged) > 0 else 0
    recall        = (flagged["high_asymmetry"].sum() / afrr_pivot_sorted["high_asymmetry"].sum() * 100
                     if afrr_pivot_sorted["high_asymmetry"].sum() > 0 else 0)

    st.info(
        f"**Simple leading indicator rule:** When renewable share ≥ {ren_threshold:.0f}% "
        f"AND wind share ≥ {wind_threshold:.0f}%, conditions favour a high-asymmetry event. "
        f"This rule identifies **{recall:.0f}%** of high-asymmetry events in the dataset "
        f"with a precision of **{precision:.0f}%** (i.e. {precision:.0f}% of flagged periods "
        f"were genuinely high-asymmetry). "
        f"Integrating demand forecasts and wind forecasts would sharpen this signal further."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 5 — WHAT SHOULD A VPP DO ABOUT IT?
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### What Should a VPP Do About It?")
st.caption(
    "Translating the analytical findings into operational and strategic implications "
    "for a VPP operator participating in the German balancing markets."
)

col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown(f"#### The Pattern")
    season_counts = high_events["season"].value_counts()
    peak_month    = afrr_pivot.groupby("month_name")["spread"].mean().idxmax()
    worst_month   = afrr_pivot.groupby("month_name")["spread"].mean().idxmin()

    st.markdown(
        f"High-value windows for upward flexibility occur predominantly in "
        f"**{season_counts.index[0]}** and **{season_counts.index[1] if len(season_counts) > 1 else 'Spring'}**, "
        f"with the spread peaking on average in **{peak_month}** and reaching its lowest point in **{worst_month}**. "
        f"This is counterintuitive — winter, when demand is highest, is actually the cheapest period for "
        f"upward regulation because thermal plants are still online. The premium appears when renewables "
        f"are high enough to push thermal generation offline, removing the grid's fast-response backup."
    )

    st.markdown(f"#### The Scale")
    if len(high_events) > 0 and len(normal_events) > 0:
        multiple = high_events["positive"].mean() / normal_events["positive"].mean()
        st.markdown(
            f"During high-asymmetry events, aFRR+ prices averaged "
            f"**€{high_events['positive'].mean():.1f}/MW/h** — "
            f"**{multiple:.1f}x** the baseline of "
            f"€{normal_events['positive'].mean():.1f}/MW/h during normal periods. "
            f"FCR prices were also elevated during these windows, suggesting a "
            f"broad market tightening rather than an aFRR-specific phenomenon."
        )

with col_r:
    st.markdown(f"#### The Action")
    st.markdown(
        f"A VPP operator running aggregated home battery assets should consider three adjustments "
        f"based on these findings:\n\n"
        f"**1. Seasonal bidding strategy** — Prioritise aFRR+ capacity in the May to August window. "
        f"This is when upward flexibility commands the highest premium. In December and January, "
        f"the spread is near zero or negative — the value is in aFRR– or day-ahead arbitrage instead.\n\n"
        f"**2. State-of-charge management** — During high-renewable periods (renewable share above "
        f"~{afrr_pivot['renewable_share_pct'].quantile(0.60):.0f}%), "
        f"protect battery state-of-charge for upward dispatch rather than depleting it on "
        f"day-ahead arbitrage. The balancing premium justifies it.\n\n"
        f"**3. Forward signal monitoring** — When renewable share and wind share are both elevated, "
        f"the probability of a high-asymmetry event in the coming time blocks increases. "
        f"This is the 'can you see it coming' signal — and the data suggests it is visible "
        f"with at least same-day advance notice."
    )

    st.markdown(f"#### The Limitation")
    st.markdown(
        "This analysis is based on two years of historical data. The market is evolving — "
        "battery storage capacity is growing rapidly, which will eventually compress these premiums "
        "as more assets compete for the same aFRR+ contracts. The window of opportunity "
        "may narrow over time. Acting on this insight now, while the premium is still structural, "
        "is more valuable than acting on it in three years when the market has adjusted."
    )

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<p style='color:{SLATE}; font-size:0.82rem;'>"
    "Built by Feyi Monehin &nbsp;|&nbsp; "
    "Data: regelleistung.net (FCR/aFRR tenders) + SMARD.de (Bundesnetzagentur) &nbsp;|&nbsp; "
    f"<a href='https://www.linkedin.com/in/feyisogo-monehin-33a60212b/' style='color:{TEAL}'>LinkedIn</a>"
    f" &nbsp;·&nbsp; "
    f"<a href='https://feyimonehin.framer.website/' style='color:{TEAL}'>Portfolio</a>"
    "</p>",
    unsafe_allow_html=True,
)
"""
dashboard.py  —  German Balancing Market Analytics Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three analytical views:
  1. FCR Price Tracker       — clearing price trends + seasonal patterns
  2. aFRR Market Depth       — awarded capacity, bid competition, pos vs neg spread
  3. Renewables vs Tightness — correlation between RES share and reserve prices

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="German Balancing Market Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ───────────────────────────────────────────────────────────

TEAL      = "#0F6E56"
TEAL_L    = "#14A37F"
AMBER     = "#E08C00"
SLATE     = "#3D4A5C"
RED       = "#C0392B"
GREY_LINE = "#DDE2EA"
BLACK     = "#1A1A1A"

# ── shared layout helper ─────────────────────────────────────────────────────

def apply_base_layout(fig, height=380):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Calibri, Arial, sans-serif", color=BLACK, size=12),
        title_font=dict(color=BLACK, size=13),
        legend=dict(orientation="h", y=1.08, font=dict(color=BLACK, size=11)),
        height=height,
        margin=dict(t=50, b=40, l=10, r=10),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GREY_LINE,
        title_font=dict(color=BLACK),
        tickfont=dict(color=BLACK),
    )
    fig.update_yaxes(
        gridcolor=GREY_LINE,
        title_font=dict(color=BLACK),
        tickfont=dict(color=BLACK),
    )
    return fig


# ── load data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    smard = pd.read_csv(base / "smard_renewable.csv", parse_dates=["date"])
    fcr   = pd.read_csv(base / "fcr_tenders.csv",    parse_dates=["tender_date", "delivery_date"])
    afrr  = pd.read_csv(base / "afrr_tenders.csv",   parse_dates=["tender_date", "delivery_week_start"])

    smard_indexed = smard.set_index("date")[["renewable_share_pct"]].copy()
    fcr = fcr.join(smard_indexed.rename(columns={"renewable_share_pct": "ren_share_pct"}),
                   on="delivery_date", how="left")

    smard["week_start"] = smard["date"] - pd.to_timedelta(smard["date"].dt.weekday, unit="d")
    weekly_ren = smard.groupby("week_start")["renewable_share_pct"].mean().reset_index()
    weekly_ren.columns = ["delivery_week_start", "ren_share_pct"]
    afrr = afrr.merge(weekly_ren, on="delivery_week_start", how="left")

    return smard, fcr, afrr


smard, fcr, afrr = load_data()

# ── sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Filters")

min_date = fcr["tender_date"].min().date()
max_date = fcr["tender_date"].max().date()

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

fcr_f   = fcr[(fcr["tender_date"].dt.date >= d_start) & (fcr["tender_date"].dt.date <= d_end)]
afrr_f  = afrr[(afrr["delivery_week_start"].dt.date >= d_start) & (afrr["delivery_week_start"].dt.date <= d_end)]
smard_f = smard[(smard["date"].dt.date >= d_start) & (smard["date"].dt.date <= d_end)]

afrr_pos = afrr_f[afrr_f["direction"] == "positive"]
afrr_neg = afrr_f[afrr_f["direction"] == "negative"]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- [regelleistung.net](https://www.regelleistung.net) — FCR & aFRR tender results\n"
    "- [SMARD.de](https://www.smard.de) — Bundesnetzagentur generation data\n"
)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <h1 style='color:{TEAL}; margin-bottom:0'>German Balancing Market Analytics</h1>
    <p style='color:rgba(255,255,255,0.65); font-size:1.05rem; margin-top:4px'>
    FCR & aFRR tender performance vs. renewable penetration &nbsp;·&nbsp;
    {d_start.strftime('%d %b %Y')} to {d_end.strftime('%d %b %Y')}
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────

avg_fcr_price = fcr_f["clearing_price_eur_mw_week"].mean()
max_fcr_price = fcr_f["clearing_price_eur_mw_week"].max()
avg_ren       = smard_f["renewable_share_pct"].mean()
avg_pos_afrr  = afrr_pos["clearing_price_eur_mw_h"].mean()
avg_neg_afrr  = afrr_neg["clearing_price_eur_mw_h"].mean()

def kpi_card(col, label, value, unit):
    col.markdown(
        f"""
        <div style='padding:12px 0 8px 0;'>
            <div style='font-size:0.82rem; color:rgba(255,255,255,0.6); font-weight:600;
                        letter-spacing:0.02em; margin-bottom:4px'>{label}</div>
            <div style='font-size:2rem; font-weight:700; color:#ffffff;
                        line-height:1.1'>{value:.1f}</div>
            <div style='font-size:0.82rem; color:rgba(255,255,255,0.5); margin-top:2px'>{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "Avg FCR Price",       avg_fcr_price, "€ / MW / week")
kpi_card(k2, "FCR Price Peak",      max_fcr_price, "€ / MW / week")
kpi_card(k3, "Avg Renewable Share", avg_ren,        "%")
kpi_card(k4, "Avg aFRR+ Price",     avg_pos_afrr,  "€ / MW / h")
kpi_card(k5, "Avg aFRR– Price",     avg_neg_afrr,  "€ / MW / h")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 1 — FCR PRICE TRACKER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### FCR Clearing Price Tracker")
st.caption(
    "Frequency Containment Reserve (FCR) is tendered daily. "
    "Price reflects the marginal cost of holding 600 MW of symmetric response capacity in Germany."
)

fcr_f = fcr_f.copy()
fcr_f["rolling_30d"] = fcr_f["clearing_price_eur_mw_week"].rolling(30, min_periods=1).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=fcr_f["tender_date"], y=fcr_f["clearing_price_eur_mw_week"],
    name="Daily clearing price",
    mode="lines",
    line=dict(color=GREY_LINE, width=1),
    hovertemplate="%{x|%d %b %Y}<br>Price: €%{y:.2f}/MW/wk<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=fcr_f["tender_date"], y=fcr_f["rolling_30d"],
    name="30-day rolling avg",
    mode="lines",
    line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>30d avg: €%{y:.2f}/MW/wk<extra></extra>",
))
fig1.update_layout(
    title="FCR daily clearing price and 30-day rolling average",
    xaxis_title="Tender date",
    yaxis_title="Clearing price (€/MW/week)",
)
apply_base_layout(fig1, height=380)

c1a, c1b = st.columns([2, 1])
c1a.plotly_chart(fig1, use_container_width=True)

fcr_f["month"] = fcr_f["tender_date"].dt.to_period("M").astype(str)
monthly_fcr = fcr_f.groupby("month")["clearing_price_eur_mw_week"].mean().reset_index()
monthly_fcr.columns = ["month", "avg_price"]

fig1b = px.bar(
    monthly_fcr, x="month", y="avg_price",
    title="Monthly average FCR price",
    color="avg_price",
    color_continuous_scale=[[0, "#D5EDE8"], [0.5, TEAL_L], [1, TEAL]],
    labels={"avg_price": "€/MW/wk", "month": ""},
)
fig1b.update_layout(coloraxis_showscale=False, xaxis=dict(tickangle=-45))
apply_base_layout(fig1b, height=380)
c1b.plotly_chart(fig1b, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 2 — aFRR MARKET DEPTH
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### aFRR Market Depth: Positive vs Negative Regulation")
st.caption(
    "Automatic Frequency Restoration Reserve (aFRR) is tendered in 4-hour blocks. "
    "The spread between positive (upward) and negative (downward) prices reflects the directional "
    "cost of balancing, a signal of how asymmetric the market is becoming as renewable share grows."
)

c2a, c2b = st.columns([3, 2])

fig2a = go.Figure()
fig2a.add_trace(go.Scatter(
    x=afrr_pos["delivery_week_start"], y=afrr_pos["clearing_price_eur_mw_h"],
    name="aFRR+ (upward)",
    mode="lines+markers",
    marker=dict(size=3),
    line=dict(color=TEAL, width=1.5),
    hovertemplate="%{x|%d %b %Y}<br>Price: €%{y:.2f}/MW/h<extra></extra>",
))
fig2a.add_trace(go.Scatter(
    x=afrr_neg["delivery_week_start"], y=afrr_neg["clearing_price_eur_mw_h"],
    name="aFRR– (downward)",
    mode="lines+markers",
    marker=dict(size=3),
    line=dict(color=AMBER, width=1.5),
    hovertemplate="%{x|%d %b %Y}<br>Price: €%{y:.2f}/MW/h<extra></extra>",
))
fig2a.update_layout(
    title="aFRR clearing prices over time",
    xaxis_title="Delivery date",
    yaxis_title="Clearing price (€/MW/h)",
)
apply_base_layout(fig2a, height=360)
c2a.plotly_chart(fig2a, use_container_width=True)

fig2b = go.Figure()
fig2b.add_trace(go.Box(
    y=afrr_pos["clearing_price_eur_mw_h"],
    name="aFRR+",
    marker_color=TEAL,
    boxmean=True,
))
fig2b.add_trace(go.Box(
    y=afrr_neg["clearing_price_eur_mw_h"],
    name="aFRR–",
    marker_color=AMBER,
    boxmean=True,
))
fig2b.update_layout(
    title="Price distribution by direction",
    yaxis_title="Clearing price (€/MW/h)",
    showlegend=False,
)
apply_base_layout(fig2b, height=360)
c2b.plotly_chart(fig2b, use_container_width=True)

# aFRR price spread: aFRR+ minus aFRR– per time block
# Pivot to get positive and negative side by side, then calculate spread
afrr_spread = afrr_f.pivot_table(
    index=["tender_date", "delivery_week_start"],
    columns="direction",
    values="clearing_price_eur_mw_h",
    aggfunc="mean",
).reset_index()

afrr_spread.columns.name = None

if "positive" in afrr_spread.columns and "negative" in afrr_spread.columns:
    afrr_spread["spread"] = afrr_spread["positive"] - afrr_spread["negative"]
    afrr_spread = afrr_spread.dropna(subset=["spread"])

    # Rolling average to smooth the noise
    afrr_spread = afrr_spread.sort_values("delivery_week_start")
    afrr_spread["spread_30d"] = afrr_spread["spread"].rolling(30, min_periods=1).mean()

    fig2c = go.Figure()
    fig2c.add_trace(go.Scatter(
        x=afrr_spread["delivery_week_start"],
        y=afrr_spread["spread"],
        name="aFRR+ minus aFRR– price",
        mode="lines",
        line=dict(color=GREY_LINE, width=1),
        hovertemplate="%{x|%d %b %Y}<br>Spread: €%{y:.2f}/MW/h<extra></extra>",
    ))
    fig2c.add_trace(go.Scatter(
        x=afrr_spread["delivery_week_start"],
        y=afrr_spread["spread_30d"],
        name="30-period rolling avg",
        mode="lines",
        line=dict(color=TEAL, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>30-period avg: €%{y:.2f}/MW/h<extra></extra>",
    ))
    # Zero reference line
    fig2c.add_hline(
        y=0,
        line_dash="dash",
        line_color=AMBER,
        line_width=1.5,
        annotation_text="Parity (aFRR+ = aFRR–)",
        annotation_position="top left",
        annotation_font=dict(color=AMBER, size=11),
    )
    fig2c.update_layout(
        title="aFRR price spread: upward minus downward regulation (€/MW/h)",
        xaxis_title="Delivery date",
        yaxis_title="Price spread (€/MW/h)",
    )
    apply_base_layout(fig2c, height=320)
    st.plotly_chart(fig2c, use_container_width=True)

    st.caption(
        "When the spread is positive, upward flexibility (aFRR+) commands a premium over downward, "
        "a signal that thermal generation headroom is tight and dispatchable upward capacity is scarce. "
        "Spikes above zero are the highest-value moments for a VPP with upward dispatch capability."
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# VIEW 3 — RENEWABLES vs RESERVE MARKET TIGHTNESS
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("### Renewable Penetration vs. Reserve Market Tightness")
st.caption(
    "As renewable share grows, how do FCR and aFRR clearing prices respond? "
    "The correlation here shows whether the market is becoming structurally tighter "
    "or whether battery storage supply is keeping pace."
)

c3a, c3b = st.columns([1, 1])

corr_fcr = fcr_f[["ren_share_pct", "clearing_price_eur_mw_week"]].dropna()
r_fcr = corr_fcr.corr().iloc[0, 1]

fig3a = px.scatter(
    corr_fcr,
    x="ren_share_pct",
    y="clearing_price_eur_mw_week",
    trendline="ols",
    title=f"FCR price vs renewable share  (r = {r_fcr:.2f})",
    labels={
        "ren_share_pct": "Daily avg renewable share (%)",
        "clearing_price_eur_mw_week": "FCR clearing price (€/MW/wk)",
    },
    color_discrete_sequence=[TEAL],
)
fig3a.update_traces(marker=dict(opacity=0.4, size=4), selector=dict(mode="markers"))
fig3a.data[1].line.color = RED
apply_base_layout(fig3a, height=380)
c3a.plotly_chart(fig3a, use_container_width=True)

corr_pos = afrr_pos[["ren_share_pct", "clearing_price_eur_mw_h"]].dropna()
r_pos = corr_pos.corr().iloc[0, 1]

fig3b = px.scatter(
    corr_pos,
    x="ren_share_pct",
    y="clearing_price_eur_mw_h",
    trendline="ols",
    title=f"aFRR+ price vs renewable share  (r = {r_pos:.2f})",
    labels={
        "ren_share_pct": "Weekly avg renewable share (%)",
        "clearing_price_eur_mw_h": "aFRR+ clearing price (€/MW/h)",
    },
    color_discrete_sequence=[AMBER],
)
fig3b.update_traces(marker=dict(opacity=0.5, size=5), selector=dict(mode="markers"))
fig3b.data[1].line.color = RED
apply_base_layout(fig3b, height=380)
c3b.plotly_chart(fig3b, use_container_width=True)

smard_monthly = smard_f.copy()
smard_monthly["month"] = smard_monthly["date"].dt.to_period("M").dt.to_timestamp()
smard_monthly = smard_monthly.groupby("month")["renewable_share_pct"].mean().reset_index()

fcr_monthly = fcr_f.copy()
fcr_monthly["month"] = fcr_monthly["tender_date"].dt.to_period("M").dt.to_timestamp()
fcr_monthly = fcr_monthly.groupby("month")["clearing_price_eur_mw_week"].mean().reset_index()

combined = smard_monthly.merge(fcr_monthly, on="month", how="inner")

fig3c = make_subplots(specs=[[{"secondary_y": True}]])
fig3c.add_trace(
    go.Bar(x=combined["month"], y=combined["renewable_share_pct"],
           name="Renewable share (%)", marker_color="#D5EDE8", opacity=0.8),
    secondary_y=False,
)
fig3c.add_trace(
    go.Scatter(x=combined["month"], y=combined["clearing_price_eur_mw_week"],
               name="FCR avg price (€/MW/wk)", line=dict(color=TEAL, width=2.5),
               mode="lines+markers", marker=dict(size=5)),
    secondary_y=True,
)
fig3c.update_layout(title="Monthly renewable share vs FCR clearing price")
apply_base_layout(fig3c, height=360)
fig3c.update_yaxes(
    title_text="Renewable share (%)", secondary_y=False,
    gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK),
)
fig3c.update_yaxes(
    title_text="FCR price (€/MW/wk)", secondary_y=True,
    gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK),
)
st.plotly_chart(fig3c, use_container_width=True)

# ── insight callout ────────────────────────────────────────────────────────────

r_dir = "negatively" if r_fcr < 0 else "positively"
if r_fcr > 0:
    trend_note = (
        f"FCR clearing prices are {r_dir} correlated with renewable share (r = {r_fcr:.2f}), "
        "suggesting that battery storage supply has not yet kept pace with growing reserve demand "
        "and that the market was structurally tight in this period."
    )
else:
    trend_note = (
        f"FCR clearing prices are {r_dir} correlated with renewable share (r = {r_fcr:.2f}), "
        "consistent with growing battery storage supply entering the FCR market faster than demand — "
        "a sign that the market is becoming more competitive."
    )

st.info(
    f"**Key finding:** {trend_note} "
    "The asymmetry between aFRR+ and aFRR\u2013 prices on high-stress days is the sharpest "
    "commercial signal, showing which direction of flexibility commands a premium and when."
)

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<p style='color:{SLATE}; font-size:0.85rem;'>"
    "Built by Feyi Monehin &nbsp;|&nbsp; "
    "Data: regelleistung.net (FCR/aFRR tenders) + SMARD.de (renewable generation) &nbsp;|&nbsp; "
    f"<a href='https://www.linkedin.com/in/feyisogo-monehin-33a60212b/' style='color:{TEAL}'>LinkedIn</a> &nbsp;·&nbsp; "
    f"<a href='https://drive.google.com/file/d/1tpBlCOGJZdMHKgGCp94LoQ4YCjJT3YjH/view' style='color:{TEAL}'>Portfolio</a>"
    "</p>",
    unsafe_allow_html=True,
)

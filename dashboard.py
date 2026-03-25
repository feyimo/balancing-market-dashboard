"""
dashboard.py
------------
When Renewables Create Scarcity: Identifying High-Value Windows for VPP Operators

Four analytical views:
  1. The Two Scarcity Stories: upward vs downward flexibility
  2. The Evening Cliff: month x block heatmap of spread
  3. What Creates the Squeeze: conditions behind high-spread events
  4. Can You See It Coming: leading indicators

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
    page_title="When Renewables Create Scarcity: Identifying High-Value Windows for VPP Operators",
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

SEASON_COLOURS = {"Winter": "#5B8DB8", "Spring": TEAL_L, "Summer": AMBER, "Autumn": "#C0722A"}

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
BLOCK_LABELS = ["00–04","04–08","08–12","12–16","16–20","20–24"]
BLOCK_HOURS = [0, 4, 8, 12, 16, 20]

# ── shared layout helper ─────────────────────────────────────────────────────
def base_layout(fig, height=380, title=""):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Calibri, Arial, sans-serif", color=BLACK, size=12),
        title_font=dict(color=BLACK, size=13),
        legend=dict(orientation="h", y=1.08, font=dict(color=BLACK, size=11)),
        legend_title_text="",
        height=height, margin=dict(t=55, b=40, l=10, r=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK))
    fig.update_yaxes(gridcolor=GREY_LINE, title_font=dict(color=BLACK), tickfont=dict(color=BLACK))
    return fig

# ── load and prepare data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"

    # ── Raw tables ──
    afrr = pd.read_csv(base / "afrr_tenders.csv", parse_dates=["delivery_date", "block_start"])
    fcr  = pd.read_csv(base / "fcr_tenders.csv",  parse_dates=["delivery_date", "block_start"])
    ren  = pd.read_csv(base / "smard_renewable.csv", parse_dates=["date"])
    load = pd.read_csv(base / "smard_load.csv",    parse_dates=["date"])

    # ── Block-level spread table (one row per 4h block) ──
    pos = afrr[afrr["direction"] == "positive"].copy()
    neg = afrr[afrr["direction"] == "negative"].copy()

    spread = pos[["delivery_date", "block_start", "block_hour"]].copy()
    spread["pos_avg"]  = pos["avg_price_eur_mw_h"].values
    spread["pos_marg"] = pos["marginal_price_eur_mw_h"].values

    neg_indexed = neg.set_index(["delivery_date", "block_hour"])
    spread = spread.set_index(["delivery_date", "block_hour"])
    spread["neg_avg"]  = neg_indexed["avg_price_eur_mw_h"]
    spread["neg_marg"] = neg_indexed["marginal_price_eur_mw_h"]
    spread = spread.reset_index()

    spread["spread_avg"]  = spread["pos_avg"]  - spread["neg_avg"]
    spread["spread_marg"] = spread["pos_marg"] - spread["neg_marg"]

    # ── Time features ──
    spread["month"]      = spread["delivery_date"].dt.month
    spread["month_name"] = spread["delivery_date"].dt.strftime("%b")
    spread["year"]       = spread["delivery_date"].dt.year
    spread["season"]     = spread["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Autumn",10:"Autumn",11:"Autumn",
    })
    spread["block_label"] = spread["block_hour"].map(dict(zip(BLOCK_HOURS, BLOCK_LABELS)))

    # ── Merge SMARD (daily) onto block-level spread ──
    ren["merge_date"]  = ren["date"].dt.date
    load["merge_date"] = load["date"].dt.date
    spread["merge_date"] = spread["delivery_date"].dt.date

    spread = spread.merge(
        ren[["merge_date","renewable_share_pct","wind_share_pct","solar_share_pct",
             "wind_total_mwh","solar_mwh"]],
        on="merge_date", how="left"
    )
    spread = spread.merge(
        load[["merge_date","grid_load_mwh","residual_load_mwh"]],
        on="merge_date", how="left"
    )

    # ── Merge FCR onto spread (block-level) ──
    fcr["merge_date"] = fcr["delivery_date"].dt.date
    fcr_block = fcr[["merge_date","block_hour","clearing_price_eur_mw_week","surplus_mw"]].copy()
    fcr_block = fcr_block.rename(columns={"clearing_price_eur_mw_week":"fcr_price","surplus_mw":"fcr_surplus"})
    spread = spread.merge(fcr_block, on=["merge_date","block_hour"], how="left")

    # ── High-asymmetry flag (top 10% of spread_avg) ──
    threshold = spread["spread_avg"].quantile(0.90)
    spread["high_asym"] = spread["spread_avg"] >= threshold

    return afrr, fcr, spread, ren, load, threshold

afrr, fcr, spread, ren, load, threshold_90 = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
min_date = spread["delivery_date"].min().date()
max_date = spread["delivery_date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = date_range
else:
    d_start, d_end = min_date, max_date

sp = spread[(spread["delivery_date"].dt.date >= d_start) & (spread["delivery_date"].dt.date <= d_end)].copy()
high_sp = sp[sp["high_asym"]]
norm_sp = sp[~sp["high_asym"]]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- [regelleistung.net](https://www.regelleistung.net) — FCR & aFRR tenders\n"
    "- [SMARD.de](https://www.smard.de) — Generation & load (Bundesnetzagentur)\n"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**High-asymmetry threshold**\n\n"
    f"Top 10 % spread (avg price): **≥ {threshold_90:.2f} €/MW/h**\n\n"
    f"Events in period: **{len(high_sp)}** of {len(sp)} blocks"
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{TEAL}; margin-bottom:0'>When Renewables Create Scarcity</h1>"
    f"<p style='color:{TEAL}; font-size:1.1rem; font-weight:600; margin-top:4px; margin-bottom:2px'>"
    f"Identifying High-Value Windows for VPP Operators</p>"
    f"<p style='color:{SLATE}; font-size:0.95rem; margin-top:0'>"
    f"German FCR &amp; aFRR balancing markets · "
    f"{d_start.strftime('%d %b %Y')} to {d_end.strftime('%d %b %Y')}</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ──────────────────────────────────────────────────────────────────
def kpi(col, label, value, unit, fmt=".1f"):
    col.markdown(
        f"<div style='padding:10px 0 6px 0;'>"
        f"<div style='font-size:0.78rem; color:rgba(255,255,255,0.55); font-weight:600; "
        f"letter-spacing:0.03em; text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:1.85rem; font-weight:700; color:#ffffff; line-height:1.1'>{value:{fmt}}</div>"
        f"<div style='font-size:0.78rem; color:rgba(255,255,255,0.45); margin-top:2px'>{unit}</div></div>",
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, "Avg aFRR+ Price", sp["pos_avg"].mean(), "€/MW/h")
kpi(k2, "Avg aFRR– Price", sp["neg_avg"].mean(), "€/MW/h")
kpi(k3, "Avg Spread", sp["spread_avg"].mean(), "€/MW/h")
kpi(k4, "High-Asym Events", len(high_sp), "blocks", fmt=".0f")
kpi(k5, "Avg Renewable Share", sp["renewable_share_pct"].mean(), "%")
kpi(k6, "Avg FCR Price", sp["fcr_price"].mean(), "€/MW/wk")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# VIEW 1 — THE TWO SCARCITY STORIES
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · The Two Scarcity Stories")
st.caption(
    "Upward and downward flexibility are different markets with different timing. "
    "Summer is expensive for downward regulation (absorbing excess solar). "
    "Autumn and winter evenings are expensive for upward regulation (replacing lost solar)."
)

c1a, c1b = st.columns([3, 2])

# aFRR+ and aFRR- side by side over time (block-level, rolling avg)
sp_sorted = sp.sort_values("block_start")
win = 60  # ~10 days of blocks

fig1a = go.Figure()
fig1a.add_trace(go.Scatter(
    x=sp_sorted["block_start"],
    y=sp_sorted["pos_avg"].rolling(win, min_periods=1).mean(),
    name="aFRR+ (upward)", mode="lines",
    line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%d %b %Y %H:%M}<br>aFRR+: €%{y:.2f}/MW/h<extra></extra>",
))
fig1a.add_trace(go.Scatter(
    x=sp_sorted["block_start"],
    y=sp_sorted["neg_avg"].rolling(win, min_periods=1).mean(),
    name="aFRR– (downward)", mode="lines",
    line=dict(color=AMBER, width=2.5),
    hovertemplate="%{x|%d %b %Y %H:%M}<br>aFRR–: €%{y:.2f}/MW/h<extra></extra>",
))
fig1a.update_layout(
    title="aFRR capacity prices: upward vs downward (60-block rolling avg)",
    xaxis_title="", yaxis_title="Avg price (€/MW/h)",
)
base_layout(fig1a, height=360)
c1a.plotly_chart(fig1a, use_container_width=True)

# Monthly comparison bars — pos vs neg
monthly = sp.groupby("month_name").agg(
    pos=("pos_avg","mean"), neg=("neg_avg","mean"), spread=("spread_avg","mean"),
    month=("month","first"),
).reset_index().sort_values("month")

fig1b = go.Figure()
fig1b.add_trace(go.Bar(x=monthly["month_name"], y=monthly["pos"], name="aFRR+ (upward)",
                        marker_color=TEAL))
fig1b.add_trace(go.Bar(x=monthly["month_name"], y=monthly["neg"], name="aFRR– (downward)",
                        marker_color=AMBER))
fig1b.update_layout(
    title="Monthly avg: upward vs downward price",
    xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
    yaxis_title="Avg price (€/MW/h)", barmode="group", bargap=0.3,
)
base_layout(fig1b, height=360)
c1b.plotly_chart(fig1b, use_container_width=True)

# Spread time series with FCR overlay
fig1c = make_subplots(specs=[[{"secondary_y": True}]])
fig1c.add_trace(go.Scatter(
    x=sp_sorted["block_start"],
    y=sp_sorted["spread_avg"].rolling(win, min_periods=1).mean(),
    name="aFRR spread (60-block avg)", mode="lines",
    line=dict(color=TEAL, width=2.5),
), secondary_y=False)

fcr_sorted = sp_sorted.dropna(subset=["fcr_price"]).copy()
fig1c.add_trace(go.Scatter(
    x=fcr_sorted["block_start"],
    y=fcr_sorted["fcr_price"].rolling(win, min_periods=1).mean(),
    name="FCR price (60-block avg)", mode="lines",
    line=dict(color=AMBER, width=1.5, dash="dot"),
), secondary_y=True)

fig1c.add_hline(y=0, line_dash="dash", line_color=GREY_LINE, line_width=1.5)
fig1c.update_layout(title="aFRR spread + FCR price (confirming indicator)")
fig1c.update_yaxes(title_text="Spread (€/MW/h)", secondary_y=False)
fig1c.update_yaxes(title_text="FCR (€/MW/wk)", secondary_y=True)
base_layout(fig1c, height=300)
st.plotly_chart(fig1c, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# VIEW 2 — THE EVENING CLIFF
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 2 · The Evening Cliff")
st.caption(
    "When solar output drops in the evening, thermal plants have not ramped back up, "
    "and the grid scrambles for upward power. This heatmap shows exactly which "
    "month and time-of-day combinations carry the highest spread."
)

c2a, c2b = st.columns([3, 2])

# Heatmap: month × block → avg spread
heat_data = sp.groupby(["month_name","block_label"])["spread_avg"].mean().reset_index()
heat_pivot = heat_data.pivot(index="block_label", columns="month_name", values="spread_avg")
heat_pivot = heat_pivot.reindex(index=BLOCK_LABELS, columns=MONTH_ORDER)

fig2a = go.Figure(data=go.Heatmap(
    z=heat_pivot.values,
    x=heat_pivot.columns.tolist(),
    y=heat_pivot.index.tolist(),
    colorscale=[[0, RED], [0.45, "#FFFFFF"], [1, TEAL]],
    zmid=0,
    text=np.round(heat_pivot.values, 1),
    texttemplate="%{text}",
    textfont=dict(size=11),
    hovertemplate="Month: %{x}<br>Block: %{y}<br>Spread: €%{z:.2f}/MW/h<extra></extra>",
    colorbar=dict(title="€/MW/h", len=0.8),
))
fig2a.update_layout(
    title="Average spread by month and time-of-day (€/MW/h)",
    xaxis_title="", yaxis_title="4-hour block",
    yaxis=dict(autorange="reversed"),
)
base_layout(fig2a, height=380)
c2a.plotly_chart(fig2a, use_container_width=True)

# Top-10% event rate by block
block_rates = sp.groupby("block_label").agg(
    total=("high_asym","count"), high=("high_asym","sum"),
).reset_index()
block_rates["rate"] = block_rates["high"] / block_rates["total"] * 100
block_rates = block_rates.set_index("block_label").reindex(BLOCK_LABELS).reset_index()

fig2b = go.Figure()
fig2b.add_trace(go.Bar(
    x=block_rates["block_label"], y=block_rates["rate"],
    marker_color=[TEAL if r > 10 else GREY_LINE for r in block_rates["rate"]],
    text=[f"{r:.1f}%" for r in block_rates["rate"]],
    textposition="outside", textfont=dict(size=12, color=BLACK),
    hovertemplate="Block: %{x}<br>Rate: %{y:.1f}%<extra></extra>",
))
fig2b.update_layout(
    title="High-asymmetry event rate by time-of-day",
    xaxis_title="4-hour block", yaxis_title="% of blocks in top 10%",
    showlegend=False,
)
base_layout(fig2b, height=380)
c2b.plotly_chart(fig2b, use_container_width=True)

# Scatter timeline of high-asym events coloured by season
if len(high_sp) > 0:
    fig2c = go.Figure()
    for season, colour in SEASON_COLOURS.items():
        sub = high_sp[high_sp["season"] == season]
        if len(sub) == 0:
            continue
        fig2c.add_trace(go.Scatter(
            x=sub["block_start"], y=sub["spread_avg"],
            mode="markers", name=season,
            marker=dict(color=colour, size=8, opacity=0.7,
                        line=dict(color="white", width=1)),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Spread: €%{y:.2f}<br>Block: " +
                          sub["block_label"] + "<extra></extra>",
        ))
    fig2c.add_hline(y=threshold_90, line_dash="dot", line_color=RED, line_width=1.5,
                     annotation_text=f"90th pctl: €{threshold_90:.1f}",
                     annotation_position="top right",
                     annotation_font=dict(color=RED, size=10))

    # Annotate the largest spike
    if len(high_sp) > 0:
        peak_row = high_sp.loc[high_sp["spread_avg"].idxmax()]
        fig2c.add_annotation(
            x=peak_row["block_start"], y=peak_row["spread_avg"],
            text=f"{peak_row['block_start'].strftime('%d %b %Y %H:%M')}<br>€{peak_row['spread_avg']:.0f}/MW/h",
            showarrow=True, arrowhead=2, arrowcolor=SLATE,
            ax=-60, ay=-40,
            font=dict(size=10, color=SLATE),
        )
    fig2c.update_layout(
        title="High-asymmetry events timeline (top 10% spread blocks)",
        xaxis_title="", yaxis_title="Spread (€/MW/h)",
    )
    base_layout(fig2c, height=300)
    st.plotly_chart(fig2c, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# VIEW 3 — WHAT CREATES THE SQUEEZE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · What Creates the Squeeze")
st.caption(
    "The drivers are conditional on time of day. For evening blocks (16:00–24:00), "
    "residual load is the strongest predictor — high demand after solar drops off. "
    "For daytime blocks (08:00–16:00), solar share drives aFRR– up, pushing spread negative. "
    "Wind share has little effect in either case."
)

c3a, c3b, c3c = st.columns(3)

# Split scatters: evening blocks (16-24) vs daytime blocks (08-16)
evening = sp[sp["block_hour"].isin([16, 20])].copy()
daytime = sp[sp["block_hour"].isin([8, 12])].copy()

# Scatter: spread vs residual load — evening blocks only
corr_eve = evening[["residual_load_mwh","spread_avg"]].dropna().copy()
r_eve = corr_eve.corr().iloc[0,1]
corr_eve["residual_gwh"] = corr_eve["residual_load_mwh"] / 1000
fig3a = px.scatter(corr_eve, x="residual_gwh", y="spread_avg", trendline="ols",
                    title=f"Evening blocks: spread vs residual load (r = {r_eve:.2f})",
                    labels={"residual_gwh":"Residual load (GWh)","spread_avg":"Spread (€/MW/h)"},
                    color_discrete_sequence=[TEAL])
fig3a.update_traces(marker=dict(opacity=0.3, size=4), selector=dict(mode="markers"))
fig3a.data[1].line.color = RED
base_layout(fig3a, height=340)
c3a.plotly_chart(fig3a, use_container_width=True)

# Scatter: spread vs solar share — daytime blocks only
corr_day = daytime[["solar_share_pct","spread_avg"]].dropna()
r_day = corr_day.corr().iloc[0,1]
fig3b = px.scatter(corr_day, x="solar_share_pct", y="spread_avg", trendline="ols",
                    title=f"Daytime blocks: spread vs solar share (r = {r_day:.2f})",
                    labels={"solar_share_pct":"Solar share (%)","spread_avg":"Spread (€/MW/h)"},
                    color_discrete_sequence=[AMBER])
fig3b.update_traces(marker=dict(opacity=0.3, size=4), selector=dict(mode="markers"))
fig3b.data[1].line.color = RED
base_layout(fig3b, height=340)
c3b.plotly_chart(fig3b, use_container_width=True)

# Scatter: spread vs wind share — all blocks
corr_wind = sp[["wind_share_pct","spread_avg"]].dropna()
r_wind = corr_wind.corr().iloc[0,1]
fig3c = px.scatter(corr_wind, x="wind_share_pct", y="spread_avg", trendline="ols",
                    title=f"All blocks: spread vs wind share (r = {r_wind:.2f})",
                    labels={"wind_share_pct":"Wind share (%)","spread_avg":"Spread (€/MW/h)"},
                    color_discrete_sequence=[TEAL_L])
fig3c.update_traces(marker=dict(opacity=0.3, size=4), selector=dict(mode="markers"))
fig3c.data[1].line.color = RED
base_layout(fig3c, height=340)
c3c.plotly_chart(fig3c, use_container_width=True)

# Conditions profile: split into two charts to fix scale problem
if len(high_sp) > 0 and len(norm_sp) > 0:
    c3d_l, c3d_r = st.columns(2)

    # Chart 1: percentage indicators
    pct_indicators = {
        "Renewable %":  "renewable_share_pct",
        "Wind %":       "wind_share_pct",
        "Solar %":      "solar_share_pct",
    }
    pct_profile = []
    for label, col in pct_indicators.items():
        pct_profile.append({"Indicator": label,
                            "High-asymmetry": high_sp[col].mean(),
                            "Normal": norm_sp[col].mean()})
    pct_df = pd.DataFrame(pct_profile)

    fig3d = go.Figure()
    fig3d.add_trace(go.Bar(name="High-asymmetry (top 10%)", x=pct_df["Indicator"],
                            y=pct_df["High-asymmetry"], marker_color=AMBER,
                            text=[f"{v:.1f}%" for v in pct_df["High-asymmetry"]],
                            textposition="outside", textfont=dict(size=14, color=BLACK)))
    fig3d.add_trace(go.Bar(name="Normal periods", x=pct_df["Indicator"],
                            y=pct_df["Normal"], marker_color=TEAL,
                            text=[f"{v:.1f}%" for v in pct_df["Normal"]],
                            textposition="outside", textfont=dict(size=14, color=BLACK)))
    fig3d.update_layout(title="Generation mix: high-asymmetry vs normal",
                         barmode="group", yaxis_title="Share (%)", bargap=0.3,
                         yaxis=dict(range=[0, max(pct_df["High-asymmetry"].max(), pct_df["Normal"].max()) * 1.25]))
    base_layout(fig3d, height=340)
    c3d_l.plotly_chart(fig3d, use_container_width=True)

    # Chart 2: residual load
    h_load = high_sp["residual_load_mwh"].mean() / 1000
    n_load = norm_sp["residual_load_mwh"].mean() / 1000
    load_df = pd.DataFrame({
        "Period": ["High-asymmetry", "Normal"],
        "load": [h_load, n_load],
        "colour": [AMBER, TEAL],
    })
    fig3e = go.Figure()
    for _, row in load_df.iterrows():
        fig3e.add_trace(go.Bar(
            x=[row["Period"]], y=[row["load"]], name=row["Period"],
            marker_color=row["colour"],
            text=[f"{row['load']:.0f}"], textposition="outside",
            textfont=dict(size=16, color=BLACK),
        ))
    delta = h_load - n_load
    fig3e.update_layout(
        title=f"Residual load: +{delta:.0f} GWh during high-asymmetry",
        yaxis_title="Avg residual load (GWh)", showlegend=False, bargap=0.5,
        yaxis=dict(range=[0, max(h_load, n_load) * 1.2]),
    )
    base_layout(fig3e, height=340)
    c3d_r.plotly_chart(fig3e, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# VIEW 4 — CAN YOU SEE IT COMING?
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 4 · Can You See It Coming?")
st.caption(
    "The signal is not just 'high renewables'. It is the combination of time-of-day "
    "(evening blocks), season (Oct-Feb), and conditions (high residual load, moderate "
    "renewables, low solar). These are observable before the block starts."
)

c4a, c4b = st.columns(2)

# Previous-block conditions
sp_shift = sp.sort_values("block_start").copy()
sp_shift["prev_ren"]  = sp_shift["renewable_share_pct"].shift(1)
sp_shift["prev_wind"] = sp_shift["wind_share_pct"].shift(1)
sp_shift["prev_resid"] = sp_shift["residual_load_mwh"].shift(1)
sp_shift["prev_solar"] = sp_shift["solar_share_pct"].shift(1)

high_lead = sp_shift[sp_shift["high_asym"]]
norm_lead = sp_shift[~sp_shift["high_asym"]]

if len(high_lead) > 0 and len(norm_lead) > 0:
    # Previous block solar share
    fig4a = go.Figure()
    fig4a.add_trace(go.Box(y=high_lead["prev_solar"].dropna(), name="Before high-asym",
                            marker_color=AMBER, boxmean=True))
    fig4a.add_trace(go.Box(y=norm_lead["prev_solar"].dropna(), name="Before normal",
                            marker_color=TEAL, boxmean=True))
    fig4a.update_layout(title="Solar share the block before", yaxis_title="Solar share (%)")
    base_layout(fig4a, height=360)
    c4a.plotly_chart(fig4a, use_container_width=True)

    # Previous block residual load
    fig4b = go.Figure()
    fig4b.add_trace(go.Box(y=high_lead["prev_resid"].dropna()/1000, name="Before high-asym",
                            marker_color=AMBER, boxmean=True))
    fig4b.add_trace(go.Box(y=norm_lead["prev_resid"].dropna()/1000, name="Before normal",
                            marker_color=TEAL, boxmean=True))
    fig4b.update_layout(title="Residual load the block before", yaxis_title="Residual load (GWh)")
    base_layout(fig4b, height=360)
    c4b.plotly_chart(fig4b, use_container_width=True)

# Combined signal: evening block + season + conditions
evening_blocks = [16, 20]
autumn_winter_months = [10, 11, 12, 1, 2]
resid_p60 = sp["residual_load_mwh"].quantile(0.60)

sp["signal"] = (
    sp["block_hour"].isin(evening_blocks) &
    sp["month"].isin(autumn_winter_months) &
    (sp["residual_load_mwh"] >= resid_p60)
)

flagged = sp[sp["signal"]]
not_flagged = sp[~sp["signal"]]

if len(flagged) > 0 and sp["high_asym"].sum() > 0:
    precision = flagged["high_asym"].mean() * 100
    recall = flagged["high_asym"].sum() / sp["high_asym"].sum() * 100
    st.info(
        f"**Combined signal:** Evening block (16:00–24:00) + autumn/winter (Oct–Feb) + "
        f"residual load ≥ {resid_p60/1000:.0f} GWh → "
        f"captures **{recall:.0f}%** of high-asymmetry events "
        f"with **{precision:.0f}%** precision. "
        f"This is observable from day-ahead renewable forecasts and load forecasts."
    )

# Year-over-year: is it getting worse?
yearly_monthly = sp.groupby(["year","month_name","month"]).agg(
    spread=("spread_avg","mean"), n=("spread_avg","count"),
).reset_index().sort_values(["year","month"])

years = sorted(sp["year"].unique())
if len(years) >= 2:
    fig4c = go.Figure()
    colors = {years[0]: TEAL_L, years[1]: TEAL if len(years) > 1 else TEAL}
    if len(years) > 2:
        colors[years[2]] = AMBER
    for y in years:
        yd = yearly_monthly[yearly_monthly["year"] == y]
        if len(yd) < 3:
            continue
        fig4c.add_trace(go.Scatter(
            x=yd["month_name"], y=yd["spread"],
            name=str(y), mode="lines+markers",
            line=dict(color=colors.get(y, SLATE), width=2.5),
            marker=dict(size=7),
        ))
    fig4c.add_hline(y=0, line_dash="dash", line_color=GREY_LINE)
    fig4c.update_layout(
        title="Monthly spread by year: is the pattern intensifying?",
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
        yaxis_title="Avg spread (€/MW/h)",
    )
    base_layout(fig4c, height=320)
    st.plotly_chart(fig4c, use_container_width=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# CLOSING
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div style='background:{TEAL_BG}; padding:20px 24px; border-radius:8px; margin-top:8px'>"
    f"<p style='color:{TEAL}; font-size:1rem; font-weight:700; margin:0 0 8px 0'>"
    f"What does this mean for a VPP operator?</p>"
    f"<p style='color:{BLACK}; font-size:0.92rem; margin:0; line-height:1.6'>"
    f"The upward flexibility premium concentrates in autumn/winter evening blocks (16:00-24:00, Oct-Feb), "
    f"driven by the solar cliff and high residual load. Summer carries a different opportunity: "
    f"expensive downward regulation during midday solar peaks. Both patterns intensified from 2024 to 2025. "
    f"Full analysis and VPP strategy recommendations → "
    f"<a href='https://feyimonehin.framer.website/' style='color:{TEAL}; font-weight:700'>project write-up</a>."
    f"</p></div>",
    unsafe_allow_html=True,
)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='color:{SLATE}; font-size:0.82rem;'>"
    "Built by Feyi Monehin · "
    "Data: regelleistung.net (FCR/aFRR tenders) + SMARD.de (Bundesnetzagentur) · "
    f"<a href='https://www.linkedin.com/in/feyisogo-monehin-33a60212b/' style='color:{TEAL}'>LinkedIn</a>"
    f" · <a href='https://feyimonehin.framer.website/' style='color:{TEAL}'>Portfolio</a>"
    "</p>",
    unsafe_allow_html=True,
)
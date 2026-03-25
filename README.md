# When Renewables Create Scarcity: Identifying High-Value Windows for VPP Operators

> Use this as the GitHub README.md, the full technical write-up for anyone who wants to go deep.

## The Question

When does the German electricity grid most urgently need more power at short notice, and can you see it coming in the data?

Germany is replacing coal and nuclear plants with wind and solar. That is good for emissions, but it creates a problem: when the sun goes down or the wind drops, the grid loses generation fast. The backup plants that used to cover these gaps are being retired. When supply falls short, grid operators pay a premium for anyone who can ramp up quickly, and that premium is the signal this project tracks.

For a Virtual Power Plant (VPP) operator aggregating thousands of home batteries, the question is not academic. It determines when to bid into the balancing market, when to hold battery charge in reserve, and which direction (upward or downward) to prioritise in each season.

## The Data

Four datasets covering January 2024 to February 2026:

**aFRR capacity tenders** from regelleistung.net. 9,480 rows. One row per 4-hour block per direction (upward/downward), six blocks per day. This is the primary dataset. The aFRR+ (upward) price measures what the grid is willing to pay for someone to generate more. The aFRR- (downward) price measures the cost of absorbing excess. The spread between them (aFRR+ minus aFRR-) is the core metric: when the spread is high, upward flexibility is scarce relative to downward.

**FCR capacity tenders** from regelleistung.net. 4,740 rows. Same 4-hour block structure. FCR is the fastest-response reserve product. It serves as a confirming indicator: when FCR prices spike alongside aFRR, it signals broad system stress, not just noise in one market.

**Renewable generation** from SMARD.de (Bundesnetzagentur). 790 daily rows. Wind onshore, wind offshore, solar, biomass, hydro, and all conventional sources. From this I calculate renewable share, wind share, solar share, and total generation.

**Grid load** from SMARD.de. 813 daily rows. Total grid load and residual load (load minus renewables). Residual load is the sharpest single indicator of how hard conventional plants are working, and how little spare capacity they have.

An important technical note on the aFRR data: the German aFRR capacity auction is **pay-as-bid**, not pay-as-cleared. Every accepted provider gets their own bid price, not a uniform clearing price. The data includes both the marginal price (highest accepted bid, noisy since one outlier can spike it) and the average price (volume-weighted mean of all accepted bids, reflects broad market tightness). I use the average price for all spread calculations because it captures genuine, broad-based scarcity rather than single-bid outliers. The marginal price is kept in the dataset for reference.

## Method

The dashboard analyses the aFRR spread at block level, preserving the 4-hour granularity that daily averages would destroy. The analysis follows five steps:

**1. Decompose upward vs downward pricing.** Instead of looking at aFRR+ in isolation, I track both directions side by side. This immediately revealed that summer's negative spread is not caused by cheap upward flexibility. aFRR+ prices are actually highest in summer. The spread goes negative because aFRR- (downward) prices are even higher. Summer is expensive in both directions, but the grid's greater need is for someone to absorb excess solar, not generate more.

**2. Map spread to month x time-of-day.** A heatmap with months on one axis and 4-hour blocks on the other shows where value concentrates. This was the sharpest finding: the spread is not just seasonal. Time of day matters more than month. The 16:00-24:00 evening window in October-February is where upward flexibility commands the highest premium.

**3. Profile the conditions behind high-spread events.** I tagged the top 10% of spread blocks as "high-asymmetry events" and compared their conditions to the rest. The result was counterintuitive: high-spread events do not happen during high renewable output. They happen during moderate renewables combined with high demand. Enough to displace some thermal plants during the day, but not enough to cover the evening load once solar drops off.

**4. Test leading indicators.** I examined whether conditions observable before a block starts could predict high-asymmetry events. Splitting the scatter analysis by block type (evening vs daytime) dramatically improved the correlations. For evening blocks, residual load is the strongest predictor (r = 0.34). For daytime blocks, solar share drives the spread negative (r = -0.51). Wind share has almost no explanatory power in either case.

**5. Check year-over-year consistency.** The seasonal pattern holds across both 2024 and 2025, but the extremes deepened significantly. Summer 2025's negative spread was far more pronounced than 2024's. Autumn 2025's positive spread was wider. The structural dynamics are accelerating as more solar enters the system.

## Finding

**The upward flexibility premium concentrates in autumn/winter evening blocks, driven by the "solar cliff."**

During the day, solar floods the grid and pushes thermal plants offline. When solar output collapses in the late afternoon, the grid needs fast upward response, but the thermal plants that would normally provide it have been displaced. This creates a predictable scarcity window every evening during months with meaningful solar generation followed by high demand.

The numbers: 47% of all top-10% spread events occur in the 16:00-24:00 window. Zero occur in the 00:00-04:00 window. October and November have the highest concentration rate (19.1% and 16.1% respectively). The average spread in the 16:00-20:00 block in October is EUR 36.9/MW/h, compared to the 12:00-16:00 block in June at EUR -48.6/MW/h.

Summer has its own scarcity story, but it is a *downward* flexibility opportunity. The 12:00-16:00 block from May to August carries spreads of EUR -34 to EUR -50/MW/h, meaning aFRR- is far more expensive than aFRR+. For a VPP with batteries, this means the summer play is absorbing excess solar, not generating more.

The pattern intensified from 2024 to 2025. Summer 2025's worst monthly spread (May: EUR -11.56/MW/h) was six times deeper than 2024's (Aug: EUR -1.83/MW/h). Autumn 2025's best (Oct: EUR +14.70/MW/h) exceeded 2024's peak (Nov: EUR +11.66/MW/h).

## Implication

A VPP operator running aggregated home battery assets should adjust strategy along three dimensions:

**Seasonal bidding.** Prioritise aFRR+ (upward) capacity bids in October-February, targeting the 16:00-24:00 blocks where the premium is highest. In May-August, shift to aFRR- (downward) bids during the 12:00-16:00 solar peak, that is where the grid most urgently needs flexibility, and the premium reflects it.

**State-of-charge management.** On autumn/winter days with high daytime solar forecasts, protect battery state-of-charge through the afternoon. Do not deplete on day-ahead arbitrage if the evening aFRR+ premium is likely to exceed it. The observable signal: high daytime solar + high residual load forecast + evening block approaching.

**Urgency.** As of late 2025, approximately 580 MW of battery storage was pre-qualified for aFRR in Germany, about 30% of operational BESS. Total BESS capacity is projected to reach 5.7 GW by end of 2026. If 35% of that fleet qualifies for aFRR, it would meet the full ~2 GW of capacity German TSOs procure. The premiums identified in this analysis are structural today, but they will compress as more flexible assets compete for the same contracts. The window for acting on this insight is open now. It will not stay open indefinitely.

## Output

**[Live dashboard](https://when-renewables-create-scarcity-dashboard-by-feyimo.streamlit.app)**

**[GitHub repo](https://github.com/feyimo/when-renewables-create-scarcity-dashboard)**

Built with Python, pandas, Plotly, and Streamlit. Data sourced from regelleistung.net (FCR and aFRR tenders) and SMARD.de (Bundesnetzagentur).
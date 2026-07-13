# Demo Video Script (~2.5 minutes)

Record a screen capture of `streamlit run app/app.py` in a browser. Suggested narration:

**0:00–0:15 — Hook**
"Uttar Pradesh is home to 240 million people, and it's warming faster than almost anywhere
on Earth. But raw temperature doesn't tell you when heat becomes deadly — Wet-Bulb
Temperature does. Above 35°C WBT, the human body can't cool itself even at rest. We built a
system that forecasts WBT 10 days out, across 125 districts."

**0:15–0:45 — The map**
Point at the region-wide risk map. Move the day_index slider across a few summer dates in
the test period (2015–2025) to show districts flipping from Safe → Caution → Danger.
"This is our live risk map — every district, color-coded by the worst WBT expected in the
next 10 days. A district turning red here is a signal for local authorities to act, days in
advance instead of hours."

**0:45–1:30 — District drill-down**
Pick a district that's in Danger/Fatal on the chosen day. Show the 10-day line chart with
the shaded danger bands.
"Drilling into a single district, we see the full 10-day trajectory, not just a single
number — so responders know whether the danger window is 2 days or 8 days long."

**1:30–2:00 — Model performance**
Open the "Model performance & methodology" panel. Show the WRMSE metric and the per-day
RMSE bar chart.
"Under the hood, this is a LightGBM model trained on 31 years of NASA POWER data, with
lag and rolling-window features per district. Our validation weighted-RMSE is
<INSERT metrics.json 'wrmse_10day' VALUE>°C, and — matching how the competition scores
this — our error is lowest on Day 1, when accuracy matters most operationally."

**2:00–2:20 — Why this matters**
"Current forecasting systems lose accuracy on heat-humidity variables past 2–3 days. By
pushing that horizon to 10 days, this system gives district authorities, hospitals, and
outdoor workers a real planning window before a heatwave turns dangerous."

**2:20–2:30 — Close**
"Built for the IIIT Lucknow Climate Intelligence Challenge 2026, by [team name]."

---

### Recording tips
- Use OBS Studio or the Windows Game Bar (`Win+Alt+R`) to capture the browser window.
- Pre-pick 2-3 day_index values that land in a hot month (May–June) so the map actually
  shows Danger/Fatal districts — check `app/data/predictions.parquet` for good examples:
  `df[df.risk_level.isin(["Danger","Fatal"])].day_index.value_counts().head()`.
- Fill in the actual WRMSE number from `model/metrics.json` before recording.

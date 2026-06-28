# BriefBench dashboard

Interactive, Python-backed results explorer (Streamlit + Plotly) — "Power BI, but Python".
Sliders/filters for model, dataset, depth, arm, and metric; live bars, depth-crossover
curves, efficiency Pareto, leaderboards, statistics, the competitor landscape, and a
downloadable raw-data table. Reads the measured data in `results/data/`.

## Run locally
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Deploy a public URL (free, for marketing)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → point at `dashboard/app.py`.
3. Share the URL.

Honest by design: Brief's wins are on its design axes (with/without context, depth,
robustness, efficiency); raw general-retrieval results (mid-pack) are shown straight.

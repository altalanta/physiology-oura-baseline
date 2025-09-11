Baseline Physiological Analytics (Oura-Style)

Overview
- Goal: Demonstrate end-to-end handling of physiological wearable data similar to Oura: generate a realistic synthetic dataset, compute nightly features (RHR, HRV rMSSD, sleep staging, duration), daily activity metrics, temperature deviations, and a simple readiness-like score with visualizations.
- Contents: data generator, analysis pipeline, outputs, and notes on extensions.

Why this is useful
- Shows comfort with time-series preprocessing, feature engineering, rolling baselines, and physiology-aware metrics.
- Clean structure and reproducibility: deterministic synthetic data, CLI scripts, and saved figures/CSV.

Project Structure
- `scripts/generate_synthetic_wearable_data.py` — Generates 14–30 days of 5-min aggregated wearable data.
- `analysis/baseline_oura_style_analysis.py` — Computes nightly/daily features and a readiness-like score; saves plots and `daily_summary.csv`.
- `data/sample_wearable_timeseries.csv` — Synthetic dataset (created by the generator).
- `outputs/` — Plots and summary table written here.

Quick Start
1) Generate synthetic data:
   - `python scripts/generate_synthetic_wearable_data.py --days 21 --out data/sample_wearable_timeseries.csv`
2) Run analysis:
   - `python analysis/baseline_oura_style_analysis.py --data data/sample_wearable_timeseries.csv --outdir outputs`

Dependencies
- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn` (optional: `seaborn` for heatmaps)
- Install: `pip install -r requirements.txt` (or install the above individually)

What the analysis computes
- Nightly metrics (sleep periods inferred from `sleep_stage` in synthetic data):
  - Resting heart rate (RHR): 5th percentile HR during sleep
  - HRV (rMSSD): computed from successive RR interval differences
  - Sleep duration, time in stages, sleep efficiency
- Daily metrics:
  - Total steps, active minutes, high-intensity minutes
  - Temperature deviation vs. rolling baseline
  - Readiness-like score combining HRV vs. baseline, RHR, sleep, and temperature

Outputs
- `outputs/daily_summary.csv` — One row per calendar day with features and readiness-like score
- `outputs/fig_*.png` —
  - `fig_timeseries.png` — HR, steps, temperature across time
  - `fig_hypnogram.png` — Sleep hypnogram-like plot
  - `fig_hrv_rhr.png` — Nightly rMSSD and RHR
  - `fig_readiness.png` — Readiness-like score over days
  - `fig_correlations.png` — Feature correlation heatmap (if seaborn available)

Notes and extensions
- Replace the synthetic CSV with real exports to demonstrate production handling (privacy-preserving sanitization recommended).
- Add cosinor-based circadian metrics, chronotype estimate (mid-sleep on free days), or a predictive model for next-day readiness.
- Integrate quality control (artifact detection for RR intervals) and missing-data handling.

License
- MIT (see LICENSE)

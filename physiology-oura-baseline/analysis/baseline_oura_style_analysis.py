#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False


def rmssd(rr_ms: np.ndarray) -> float:
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[~np.isnan(rr)]
    if len(rr) < 3:
        return np.nan
    diffs = np.diff(rr)
    return np.sqrt(np.mean(diffs**2))


def summarize_sleep_nightly(df: pd.DataFrame) -> pd.DataFrame:
    # Identify sleep segments via explicit sleep_stage != 'wake'
    df = df.copy()
    df['is_sleep'] = df['sleep_stage'].astype(str) != 'wake'
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    nightly = []
    for day, group in df.groupby('date'):
        # Night spans across midnight: take window from 18:00 of day to 12:00 next day
        start = pd.Timestamp(day) + pd.Timedelta(hours=18)
        end = pd.Timestamp(day) + pd.Timedelta(days=1, hours=12)
        win = df[(pd.to_datetime(df['timestamp']) >= start) & (pd.to_datetime(df['timestamp']) < end)].copy()
        if win.empty:
            continue

        asleep = win[win['is_sleep']]
        if asleep.empty:
            # no sleep detected in window
            nightly.append({
                'date': day,
                'sleep_start': pd.NaT,
                'sleep_end': pd.NaT,
                'sleep_minutes': 0,
                'sleep_efficiency': np.nan,
                'rhr': np.nan,
                'rmssd': np.nan,
                't_deep': 0,
                't_rem': 0,
                't_light': 0,
                't_wake': int(len(win) * 5)
            })
            continue

        sleep_start = pd.to_datetime(asleep['timestamp']).min()
        sleep_end = pd.to_datetime(asleep['timestamp']).max() + (pd.to_datetime(win['timestamp']).diff().median() or pd.Timedelta(minutes=5))
        sleep_minutes = int(asleep.shape[0] * 5)

        # Sleep efficiency: asleep minutes / time in bed (from first to last sleep-labelled sample)
        tib_minutes = int(((sleep_end - sleep_start).total_seconds())/60)
        sleep_eff = sleep_minutes / tib_minutes if tib_minutes > 0 else np.nan

        # RHR as 5th percentile HR during sleep (robust)
        rhr = np.nanpercentile(asleep['heart_rate'], 5) if not asleep['heart_rate'].empty else np.nan

        # Nightly rMSSD from RR intervals during sleep
        r = rmssd(asleep['rr_ms'].values)

        # Time in stages (minutes)
        t_deep = int((asleep['sleep_stage'] == 'deep').sum() * 5)
        t_rem = int((asleep['sleep_stage'] == 'rem').sum() * 5)
        t_light = int((asleep['sleep_stage'] == 'light').sum() * 5)
        t_wake = int((win['sleep_stage'] == 'wake').sum() * 5)

        nightly.append({
            'date': day,
            'sleep_start': sleep_start,
            'sleep_end': sleep_end,
            'sleep_minutes': sleep_minutes,
            'sleep_efficiency': sleep_eff,
            'rhr': float(rhr) if rhr == rhr else np.nan,
            'rmssd': float(r) if r == r else np.nan,
            't_deep': t_deep,
            't_rem': t_rem,
            't_light': t_light,
            't_wake': t_wake,
        })

    return pd.DataFrame(nightly).sort_values('date')


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['date'] = pd.to_datetime(d['timestamp']).dt.date
    g = d.groupby('date')
    daily = g.agg(
        steps=('steps', 'sum'),
        active_minutes=('activity', lambda x: int((x > 0.4).sum() * 5)),
        hi_minutes=('activity', lambda x: int((x > 0.8).sum() * 5)),
        mean_hr=('heart_rate', 'mean'),
        temp_dev_mean=('temp_dev_c', 'mean')
    ).reset_index()

    # Rolling baseline for HRV and temperature
    daily['temp_baseline'] = daily['temp_dev_mean'].rolling(7, min_periods=3).median()
    daily['temp_delta'] = daily['temp_dev_mean'] - daily['temp_baseline']

    return daily


def compute_readiness(daily: pd.DataFrame, nightly: pd.DataFrame) -> pd.DataFrame:
    df = daily.merge(nightly, on='date', how='left')

    # HRV baseline (rolling median)
    df['rmssd_baseline'] = df['rmssd'].rolling(7, min_periods=3).median()
    df['rmssd_ratio'] = df['rmssd'] / df['rmssd_baseline']

    # Normalize components to ~0..1 (clamped)
    def nz(x):
        return np.where(np.isfinite(x), x, np.nan)

    sleep_score = np.clip(nz(df['sleep_minutes'])/ (7.5*60), 0, 1)
    eff_score = np.clip(nz(df['sleep_efficiency']), 0, 1)
    hrv_score = np.clip(nz(df['rmssd_ratio']), 0.6, 1.4)
    hrv_score = (hrv_score - 0.6) / (1.4 - 0.6)
    # Lower RHR is better: compare to 56 bpm ref within [44, 70]
    rhr_norm = np.clip((70 - nz(df['rhr'])) / (70 - 44), 0, 1)
    # Temperature: negative delta better; penalize positive deltas
    temp_penalty = np.clip(1 - np.maximum(nz(df['temp_delta']), 0)*6, 0, 1)

    # Weighted readiness-like score (0..100)
    # Emphasis on HRV, RHR, and sleep; temperature acts as penalty multiplier.
    base = 0.35*hrv_score + 0.25*rhr_norm + 0.25*sleep_score + 0.15*eff_score
    readiness = 100 * temp_penalty * base

    out = df.copy()
    out['readiness'] = readiness
    return out


def plot_timeseries(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ts = pd.to_datetime(df['timestamp'])
    ax[0].plot(ts, medfilt(df['heart_rate'], kernel_size=5), label='HR (bpm)')
    ax[0].set_ylabel('HR (bpm)')
    ax[0].legend(loc='upper right')

    ax[1].plot(ts, df['steps'].rolling(3).sum(), color='tab:green', label='Steps (5-min)')
    ax[1].set_ylabel('Steps')
    ax[1].legend(loc='upper right')

    ax[2].plot(ts, df['temp_dev_c'], color='tab:red', label='Temp Δ (°C)')
    ax[2].set_ylabel('Temp Δ (°C)')
    ax[2].legend(loc='upper right')

    ax[2].set_xlabel('Time')
    fig.tight_layout()
    fp = outdir / 'fig_timeseries.png'
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_hypnogram(df: pd.DataFrame, outdir: Path):
    # Plot last night hypnogram-like chart
    d = pd.to_datetime(df['timestamp']).dt.date
    last = d.max()
    mask = d == last
    sub = df.loc[mask].copy()
    if sub.empty:
        return None

    stage_map = {'wake': 3, 'light': 2, 'rem': 1, 'deep': 0}
    y = sub['sleep_stage'].map(stage_map).fillna(3)
    ts = pd.to_datetime(sub['timestamp'])

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.step(ts, y, where='post')
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(['deep','rem','light','wake'])
    ax.set_title('Hypnogram (last night)')
    ax.set_xlabel('Time')
    fig.tight_layout()
    fp = outdir / 'fig_hypnogram.png'
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_hrv_rhr(nightly: pd.DataFrame, outdir: Path):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(nightly['date'])
    ax1.plot(x, nightly['rmssd'], marker='o', color='tab:blue', label='rMSSD (ms)')
    ax1.set_ylabel('rMSSD (ms)')
    ax2 = ax1.twinx()
    ax2.plot(x, nightly['rhr'], marker='s', color='tab:red', label='RHR (bpm)')
    ax2.set_ylabel('RHR (bpm)')
    ax1.set_title('Nightly HRV (rMSSD) and RHR')
    fig.tight_layout()
    fp = outdir / 'fig_hrv_rhr.png'
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_readiness(daily: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = pd.to_datetime(daily['date'])
    ax.plot(x, daily['readiness'], marker='o')
    ax.set_ylabel('Readiness (0-100)')
    ax.set_title('Readiness-like Score')
    fig.tight_layout()
    fp = outdir / 'fig_readiness.png'
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_correlations(df: pd.DataFrame, outdir: Path):
    cols = ['readiness','rmssd','rhr','sleep_minutes','sleep_efficiency','steps','active_minutes','hi_minutes','temp_delta']
    sub = df[cols].copy()
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    if HAVE_SEABORN:
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    else:
        im = ax.imshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        fig.colorbar(im, ax=ax)
    ax.set_title('Feature Correlations')
    fig.tight_layout()
    fp = outdir / 'fig_correlations.png'
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description='Baseline physiological analytics on synthetic wearable data')
    ap.add_argument('--data', type=str, default='physiology-oura-baseline/data/sample_wearable_timeseries.csv', help='Input CSV path')
    ap.add_argument('--outdir', type=str, default='physiology-oura-baseline/outputs', help='Output directory')
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if 'timestamp' not in df.columns:
        raise ValueError('CSV must include a timestamp column')

    nightly = summarize_sleep_nightly(df)
    daily = summarize_daily(df)
    features = compute_readiness(daily, nightly)

    # Save summary
    features.to_csv(outdir / 'daily_summary.csv', index=False)

    # Plots
    plot_timeseries(df, outdir)
    plot_hypnogram(df, outdir)
    plot_hrv_rhr(nightly, outdir)
    plot_readiness(features, outdir)
    plot_correlations(features, outdir)

    print(f"Wrote daily summary and figures to {outdir}")


if __name__ == '__main__':
    main()


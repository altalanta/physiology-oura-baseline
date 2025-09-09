#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic(days=14, freq="5min", seed=42):
    rng = np.random.default_rng(seed)

    # Time index
    start = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=days)
    idx = pd.date_range(start=start, periods=int((24*60)/5*days) if freq=="5min" else days*24*60, freq=freq)

    # Daily sleep schedule (approximate; add jitter per day)
    # Sleep around 23:30–07:30
    base_sleep_start = pd.Timedelta(hours=23, minutes=30)
    base_sleep_end = pd.Timedelta(hours=7, minutes=30)

    df = pd.DataFrame(index=idx)
    df["date"] = df.index.date

    # Circadian baseline for HR and temperature
    t = (df.index.view(np.int64) // 10**9).astype(float)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

    # Activity proxy and steps: higher during daytime, with random activity bouts
    activity = np.zeros(len(df))
    steps = np.zeros(len(df))
    heart_rate = np.zeros(len(df))
    temp_dev = np.zeros(len(df))
    sleep_stage = np.array(["wake"]*len(df), dtype=object)

    current_day = None
    sleep_intervals = []

    for i, ts in enumerate(df.index):
        if current_day != ts.date():
            current_day = ts.date()
            # Jitter sleep window per day
            jitter_start = rng.normal(0, 20)  # +/-20 min
            jitter_end = rng.normal(0, 20)
            sleep_start = pd.Timestamp(current_day) + base_sleep_start + pd.Timedelta(minutes=jitter_start)
            # if start past midnight, adjust date
            if sleep_start.time() < datetime.min.time():
                sleep_start = sleep_start + pd.Timedelta(days=1)
            sleep_end = pd.Timestamp(current_day) + pd.Timedelta(days=1) + base_sleep_end + pd.Timedelta(minutes=jitter_end)
            sleep_intervals.append((sleep_start, sleep_end))

            # Temperature deviation baseline drift; occasional spike (e.g., mild sickness)
            temp_offset = rng.normal(0, 0.03)
            if rng.random() < 0.05:
                temp_offset += rng.normal(0.3, 0.05)  # one-off feverish day

        # Determine asleep
        asleep = False
        for (ss, se) in sleep_intervals[-1:]:  # only check latest interval
            if ss <= ts < se:
                asleep = True
                break

        # Activity pattern: low during sleep, moderate day, peaks in morning/evening windows
        hour = ts.hour + ts.minute/60
        daytime = (hour >= 7) and (hour <= 22)
        base_act = 0.05 if asleep else (0.3 + 0.4*np.sin((hour-9)/24*2*np.pi)**2)
        # Random bouts of high activity during day
        if daytime and rng.random() < 0.03:
            base_act += rng.uniform(0.5, 1.0)
        activity[i] = max(0.0, base_act + rng.normal(0, 0.05))
        steps[i] = max(0, int(activity[i] * rng.integers(0, 60)))

        # Heart rate model: lower at sleep, higher with activity, with circadian component
        circadian = 5*np.sin((hour-3)/24*2*np.pi)
        hr_base = 50 if asleep else 65
        hr = hr_base + 25*activity[i] + circadian + rng.normal(0, 1.5)
        heart_rate[i] = np.clip(hr, 38, 190)

        # Temperature deviation (deg C) vs personal baseline
        temp_dev[i] = temp_offset + (0.03 if asleep else 0) + 0.02*np.sin((hour-5)/24*2*np.pi) + rng.normal(0, 0.01)

        # Sleep stage during sleep window
        if asleep:
            # Simple hypnogram cycling
            cycle = int(((ts - ss).total_seconds()/60)//90) % 4
            # randomize stage a bit
            r = rng.random()
            if r < 0.05:
                st = "wake"
            else:
                st = ["light", "deep", "rem", "light"][cycle]
            sleep_stage[i] = st

    df["heart_rate"] = np.round(heart_rate, 1)
    # RR intervals (ms) approximated from HR with added variability; higher variability during sleep
    rr_nominal = 60000 / np.clip(df["heart_rate"].values, 35, None)
    rr_noise = rng.normal(0, 12, size=len(df)) + np.where(sleep_stage != "wake", rng.normal(0, 18, size=len(df)), 0)
    df["rr_ms"] = np.clip(rr_nominal + rr_noise, 300, 2000)

    df["steps"] = steps.astype(int)
    df["activity"] = activity
    df["temp_dev_c"] = np.round(temp_dev, 3)
    df["sleep_stage"] = sleep_stage

    # Keep tidy columns
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic wearable timeseries data (Oura-style)")
    ap.add_argument("--days", type=int, default=14, help="Number of days to generate")
    ap.add_argument("--freq", type=str, default="5min", help="Sampling frequency (e.g., 5min)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", type=str, default="data/sample_wearable_timeseries.csv", help="Output CSV path")
    args = ap.parse_args()

    from pathlib import Path
    df = generate_synthetic(days=args.days, freq=args.freq, seed=args.seed)
    # Ensure output dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic dataset to {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()

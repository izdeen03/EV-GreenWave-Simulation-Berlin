import os
import pandas as pd
import matplotlib.pyplot as plt  # make sure to import this!

# 1. Load CSVs
base = pd.read_csv("perf_baseline.csv")
wave = pd.read_csv("perf_greenwave.csv")

# 2. Extract EV times
ev_time_base = base['evTravelTime'].dropna().iloc[0]  
t_start_base = base.loc[base['evTravelTime'].notna(), 'time'].min() - ev_time_base
t_end_base   = t_start_base + ev_time_base

ev_time_wave = wave['evTravelTime'].dropna().iloc[0]
t_start_wave = wave.loc[wave['evTravelTime'].notna(), 'time'].min() - ev_time_wave
t_end_wave   = t_start_wave + ev_time_wave

# 3. Slice to EV window
bwin = base[(base.time >= t_start_base) & (base.time <= t_end_base)]
wwin = wave[(wave.time >= t_start_wave) & (wave.time <= t_end_wave)]

# 4. Compute metrics
def summarize(df):
    out = {}
    out['mean_queue'] = df.groupby('edge').vehCount.mean().mean()
    out['peak_queue'] = df.groupby('edge').vehCount.max().max()
    out['mean_wait']  = df.waitingTime.mean()
    out['ev_stops']   = df.evStopped.sum()
    return pd.Series(out)

summary = pd.DataFrame({
    'baseline': summarize(bwin),
    'greenwave': summarize(wwin)
}).T

print(summary)

# 5. Plot time series of mean queue
ts_base = bwin.groupby('time').vehCount.mean()
ts_wave = wwin.groupby('time').vehCount.mean()

plt.figure(figsize=(8,5))
plt.plot(ts_base.index, ts_base.values, label='Baseline')
plt.plot(ts_wave.index, ts_wave.values, label='Green‑wave')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Mean Queue Length')
plt.title('Mean Queue Length vs. Time (EV Travel Window)')

# 6. Ensure output folder exists and save the plot
os.makedirs("statistics", exist_ok=True)
plt.savefig("statistics/sarah.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved to statistics/mean_queue.png")

# import xml.etree.ElementTree as ET
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

# # =================== CONFIG ===================
# XML_PATH_BaseLine = "outputt/azoz_output_BaseLine/queue_emergency.xml"
# XML_PATH_GW       = "outputt/azoz_output_GW/queue_emergency.xml"  # <- set to your GW file

# # Use LANES directly (NO aggregation to edges; we only sum across the chosen lanes)
# # >>> Fill these with real lane IDs from your network <<<
# mainline_lanes_raw = ["E12_1","E12_2", "E21_1", "E21_2", "E22_1","E22_2"]  # EXAMPLE
# side_lanes_raw     = ["68135997#1_1", "-776093907#2_1", "-E7_1", "E0_1", "-816324771#2_1", "-1136034180#1_1"]  # EXAMPLE

# # Queue attribute to use (meters)
# QUEUE_ATTR_PRIMARY  = "queueing_length"
# QUEUE_ATTR_FALLBACK = "queueing_length_experimental"  # used if primary absent

# # EV departure marker at 16:30
# EV_S = 16*3600 + 30*60  # 59400 s

# # Optional zoom window around EV (None = full run)
# WINDOW_START_S = 16*3600 + 25*60  # 59100 (16:25)
# WINDOW_END_S   = 16*3600 + 35*60  # 59700 (16:35)
# # ==============================================

# def parse_lane_queue_xml(path, q_attr_primary=QUEUE_ATTR_PRIMARY, q_attr_fallback=QUEUE_ATTR_FALLBACK):
#     """
#     Parse lane-based queue file:
#       <data timestep="58505.00"><lanes><lane id="..." queueing_length="..."/></lanes></data>
#     Returns df with columns: ['t','lane_id','queue_len_m'] for NON-junction lanes (skip ids starting with ':').
#     """
#     rows = []
#     tree = ET.parse(path)
#     root = tree.getroot()

#     for data in root.findall(".//data"):
#         t_str = data.attrib.get("timestep") or data.attrib.get("time")
#         if t_str is None:
#             continue
#         t = float(t_str)

#         lanes = data.find("lanes")
#         if lanes is None:
#             continue

#         for lane in lanes.findall("lane"):
#             lane_id = lane.attrib.get("id")
#             if not lane_id or lane_id.startswith(":"):
#                 continue  # ignore junction/internal lanes

#             # Prefer primary attr; else fallback; else 0
#             if q_attr_primary in lane.attrib:
#                 q = float(lane.attrib[q_attr_primary])
#             elif q_attr_fallback in lane.attrib:
#                 q = float(lane.attrib[q_attr_fallback])
#             else:
#                 q = 0.0

#             rows.append((t, lane_id, q))

#     df = pd.DataFrame(rows, columns=["t", "lane_id", "queue_len_m"]).sort_values("t")
#     if df.empty:
#         raise ValueError(f"No lane data parsed from {path}. "
#                          f"Check it contains <data timestep=...><lanes><lane ... {q_attr_primary} ...></lanes></data> blocks.")
#     return df

# def aggregate_lane_sets(df, lane_list):
#     """Sum queue length across the specified lanes per timestep (meters)."""
#     if not lane_list:
#         return pd.Series(dtype=float)
#     subset = df[df["lane_id"].isin(lane_list)]
#     return subset.groupby("t")["queue_len_m"].sum()

# def aggregate_overall(df):
#     """Sum queue length across ALL non-junction lanes per timestep (meters)."""
#     return df.groupby("t")["queue_len_m"].sum()

# def apply_window_to_series(times, series_dict, start_s=None, end_s=None):
#     """Reindex multiple series to a common timeline and optionally slice to [start_s, end_s]."""
#     # Make union timeline
#     all_times = times
#     for s in series_dict.values():
#         all_times = np.union1d(all_times, s.index.values if isinstance(s.index.values, np.ndarray) else np.array(s.index))
#     # Optional slice
#     if start_s is not None:
#         all_times = all_times[all_times >= start_s]
#     if end_s is not None:
#         all_times = all_times[all_times <= end_s]
#     # Reindex
#     out = {k: v.reindex(all_times, fill_value=0.0) for k, v in series_dict.items()}
#     return all_times, out

# def sec_to_hhmm(x, pos):
#     x = int(round(x))
#     h = x // 3600
#     m = (x % 3600) // 60
#     return f"{h:02d}:{m:02d}"

# def series_stats(s):
#     return float(s.mean()), float(s.max())

# # ------------- Load both scenarios -------------
# if XML_PATH_BaseLine == XML_PATH_GW:
#     print("⚠️  Warning: XML_PATH_BaseLine and XML_PATH_GW are identical. Update XML_PATH_GW to your Green Wave file.")

# df_b = parse_lane_queue_xml(XML_PATH_BaseLine)
# df_g = parse_lane_queue_xml(XML_PATH_GW)

# # Sanity: warn on lane IDs not present
# missing_main_b = sorted(set(mainline_lanes_raw) - set(df_b["lane_id"].unique()))
# missing_side_b = sorted(set(side_lanes_raw)     - set(df_b["lane_id"].unique()))
# missing_main_g = sorted(set(mainline_lanes_raw) - set(df_g["lane_id"].unique()))
# missing_side_g = sorted(set(side_lanes_raw)     - set(df_g["lane_id"].unique()))
# if missing_main_b or missing_side_b or missing_main_g or missing_side_g:
#     print("ℹ️ Missing lanes (not found in one or both files):")
#     if missing_main_b: print("  Baseline - main:", missing_main_b)
#     if missing_side_b: print("  Baseline - side:", missing_side_b)
#     if missing_main_g: print("  GreenWave - main:", missing_main_g)
#     if missing_side_g: print("  GreenWave - side:", missing_side_g)

# # Build per-scenario series (meters)
# main_b = aggregate_lane_sets(df_b, mainline_lanes_raw)
# side_b = aggregate_lane_sets(df_b, side_lanes_raw)
# tot_b  = aggregate_overall(df_b)

# main_g = aggregate_lane_sets(df_g, mainline_lanes_raw)
# side_g = aggregate_lane_sets(df_g, side_lanes_raw)
# tot_g  = aggregate_overall(df_g)

# # Common timeline + optional window
# t_all = np.array([], dtype=float)
# t_plot, S = apply_window_to_series(
#     t_all,
#     {"main_b": main_b, "side_b": side_b, "tot_b": tot_b,
#      "main_g": main_g, "side_g": side_g, "tot_g": tot_g},
#     start_s=WINDOW_START_S, end_s=WINDOW_END_S
# )
# main_b = S["main_b"]; side_b = S["side_b"]; tot_b = S["tot_b"]
# main_g = S["main_g"]; side_g = S["side_g"]; tot_g = S["tot_g"]

# # ------------- Plot: Side streets (Baseline vs Green Wave) -------------
# fig, ax = plt.subplots(figsize=(11, 6))
# ax.plot(t_plot, side_b.values, label="Side streets — Baseline", linewidth=2)
# ax.plot(t_plot, side_g.values, label="Side streets — Green Wave", linewidth=2)

# # EV marker at 16:30
# ax.axvline(EV_S, linestyle=":", linewidth=2, label="EV departure (16:30)")

# ax.set_title("Side Streets Queue Length — Baseline vs Green Wave")
# ax.set_xlabel("Time [HH:MM]")
# ax.set_ylabel("Queue length [m]")
# ax.grid(True)
# ax.legend()
# ax.xaxis.set_major_formatter(FuncFormatter(sec_to_hhmm))
# fig.tight_layout()
# plt.show()

# # ------------- Optional: Difference plot (GW - Baseline) -------------
# delta_side = side_g.reindex(t_plot, fill_value=0.0) - side_b.reindex(t_plot, fill_value=0.0)
# fig2, ax2 = plt.subplots(figsize=(11, 3.5))
# ax2.plot(t_plot, delta_side.values, linewidth=2)
# ax2.axhline(0, color="black", linewidth=1)
# ax2.axvline(EV_S, linestyle=":", linewidth=2, label="EV departure (16:30)")
# ax2.set_title("Δ Side Streets Queue (Green Wave − Baseline)")
# ax2.set_xlabel("Time [HH:MM]")
# ax2.set_ylabel("Δ queue length [m]")
# ax2.grid(True)
# ax2.legend()
# ax2.xaxis.set_major_formatter(FuncFormatter(sec_to_hhmm))
# fig2.tight_layout()
# plt.show()

# # ------------- Summary stats (Main street + Overall) -------------
# main_avg_b, main_max_b = series_stats(main_b)
# main_avg_g, main_max_g = series_stats(main_g)
# tot_avg_b,  tot_max_b  = series_stats(tot_b)
# tot_avg_g,  tot_max_g  = series_stats(tot_g)

# summary = pd.DataFrame({
#     "Scenario": ["Baseline","Green Wave","Baseline","Green Wave"],
#     "Group":    ["Main street","Main street","Overall","Overall"],
#     "Average queue length [m]": [main_avg_b, main_avg_g, tot_avg_b, tot_avg_g],
#     "Max queue length [m]":     [main_max_b, main_max_g,  tot_max_b,  tot_max_g],
#     "Δ vs Baseline":            [0.0, main_avg_g - main_avg_b, 0.0, tot_avg_g - tot_avg_b],
# })
# print("\n=== Summary (Average & Max queue length) ===")
# print(summary.to_string(index=False))


# ##############################


# # --- 1) Are Baseline & GW identical before 16:30?
# pre_mask = side_b.index < EV_S
# delta_pre = side_g.reindex(side_b.index, fill_value=0.0)[pre_mask] - side_b[pre_mask]
# max_abs_diff = float(np.nanmax(np.abs(delta_pre))) if len(delta_pre) else float('nan')
# print(f"Pre-16:30 max |GW − Baseline| on SIDE streets = {max_abs_diff:.2f} m")
# if max_abs_diff == 0.0:
#     print("→ They are exactly the same before 16:30 (as expected if GW only activates at EV).")

# # --- 2) Highlight the 16:26–16:28 window on your existing side-streets plot
# W0, W1 = 16*3600 + 26*60, 16*3600 + 28*60
# try:
#     ax.axvspan(W0, W1, alpha=0.08, label="16:26–16:28 window")  # re-use the axes from your plot
#     ax.legend()
#     fig.canvas.draw_idle()
# except NameError:
#     pass  # if you're running headless or ax/fig not in scope, ignore

# # --- 3) Which lanes are driving the spike in that window?
# def lane_breakdown(df, lane_list, t0, t1, top_k=10):
#     win = df[(df["t"] >= t0) & (df["t"] <= t1) & (df["lane_id"].isin(lane_list))].copy()
#     if win.empty:
#         return pd.Series(dtype=float), 0.0
#     # average queue length per lane across the window (meters)
#     lane_avg = win.groupby("lane_id")["queue_len_m"].mean().sort_values(ascending=False)
#     # time-averaged total across all chosen lanes (meters)
#     timeline = win["t"].unique()
#     total_avg = win.groupby("t")["queue_len_m"].sum().reindex(timeline, fill_value=0.0).mean()
#     return lane_avg.head(top_k), total_avg

# print("\n--- Lane contributions in 16:26–16:28 (Baseline) ---")
# top_b, total_b_win = lane_breakdown(df_b, side_lanes_raw, W0, W1)
# print(top_b.to_string())
# if len(top_b) > 0:
#     share_b = 100.0 * top_b.iloc[0] / top_b.sum()
#     print(f"Top lane share of side total (avg over window): {share_b:.1f}%")

# print("\n--- Lane contributions in 16:26–16:28 (Green Wave) ---")
# top_g, total_g_win = lane_breakdown(df_g, side_lanes_raw, W0, W1)
# print(top_g.to_string())
# if len(top_g) > 0:
#     share_g = 100.0 * top_g.iloc[0] / top_g.sum()
#     print(f"Top lane share of side total (avg over window): {share_g:.1f}%")





# ################



# import numpy as np
# import pandas as pd

# # Windows
# PRE0, PRE1   = 16*3600 + 25*60, 16*3600 + 30*60   # 16:25–16:30
# POST0, POST1 = 16*3600 + 30*60, 16*3600 + 35*60   # 16:30–16:35

# def win_stats(series, a, b):
#     s = series[(series.index >= a) & (series.index < b)]
#     return float(s.mean()), float(s.max())

# # Side streets (meters)
# pre_avg_b,  pre_max_b  = win_stats(side_b, PRE0, PRE1)
# pre_avg_g,  pre_max_g  = win_stats(side_g, PRE0, PRE1)
# post_avg_b, post_max_b = win_stats(side_b, POST0, POST1)
# post_avg_g, post_max_g = win_stats(side_g, POST0, POST1)

# def pct(delta, base):
#     return (delta / base * 100.0) if base else np.nan

# summary_side = pd.DataFrame({
#     "Window": ["16:25–16:30", "16:30–16:35"],
#     "Baseline avg [m]": [pre_avg_b,  post_avg_b],
#     "GreenWave avg [m]": [pre_avg_g,  post_avg_g],
#     "Δ avg (GW−BL) [m]": [pre_avg_g - pre_avg_b, post_avg_g - post_avg_b],
#     "Δ avg %": [pct(pre_avg_g - pre_avg_b, pre_avg_b), pct(post_avg_g - post_avg_b, post_avg_b)],
#     "Baseline max [m]": [pre_max_b,  post_max_b],
#     "GreenWave max [m]": [pre_max_g,  post_max_g],
#     "Δ max (GW−BL) [m]": [pre_max_g - pre_max_b, post_max_g - post_max_b],
#     "Δ max %": [pct(pre_max_g - pre_max_b, pre_max_b), pct(post_max_g - post_max_b, post_max_b)],
# })
# print("\n=== Side streets: pre/post EV comparison ===")
# print(summary_side.to_string(index=False))


#اللي فوق هو اصل الميمعة تاعت سايد ستريت



import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ---- (Optional) pick an interactive backend BEFORE importing pyplot ----
try:
    import matplotlib
    matplotlib.use("TkAgg")  # or "Qt5Agg"
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =================== CONFIG ===================
XML_PATH_BaseLine = "outputt/azoz_output_BaseLine/queue_emergency.xml"
XML_PATH_GW       = "outputt/azoz_output_GW/queue_emergency.xml"   # <- set to GW file

# Use LANES directly (no edge aggregation). Fill with your actual lanes.
mainline_lanes = ["E12_1","E12_2","E21_1","E21_2","E22_1","E22_2"]  # example
side_lanes     = ["68135997#1_1","-776093907#2_1","-E7_1","E0_1","-816324771#2_1","-1136034180#1_1"]  # example

# Queue attribute to use (meters)
QUEUE_ATTR_PRIMARY  = "queueing_length"
QUEUE_ATTR_FALLBACK = "queueing_length_experimental"

# Data window (data is 16:30–17:00; labels show one hour earlier)
START_S = 16*3600 #+ 30*60   # 59400 = 16:30
END_S   = 17*3600           # 61200 = 17:00
#END_S   = 16*3600  + 32*60 + 15         # 61200 = 17:00
LABEL_TIME_SHIFT_S = -3600  # display one hour earlier

# Visible axis start at 15:29 (i.e., one minute before 15:30 label)
#XMIN_LABEL = 15*3600 + 29*60           # 15:29 label time
XMIN_LABEL = 15*3600 #+ 30*60 
XMIN_ACTUAL = XMIN_LABEL - LABEL_TIME_SHIFT_S  # convert back to actual (adds 1h)

# EV events (same departure; different arrivals)
EV_DEP    = 16*3600 + 30*60        # 16:30
EV_ARR_BL = EV_DEP + 132           # +135s -> 16:32:15
EV_ARR_GW = EV_DEP + 89            # +89s  -> 16:31:29

# Output
OUTDIR = "queue_cmp_report"
os.makedirs(OUTDIR, exist_ok=True)
# ==============================================

def parse_lane_queue_xml(path, q_primary=QUEUE_ATTR_PRIMARY, q_fallback=QUEUE_ATTR_FALLBACK):
    """Parse lane-based queue XML and return df[t, lane_id, queue_len_m] (non-junction lanes only)."""
    rows = []
    root = ET.parse(path).getroot()
    for data in root.findall(".//data"):
        t_str = data.attrib.get("timestep") or data.attrib.get("time")
        if t_str is None:
            continue
        t = float(t_str)
        lanes = data.find("lanes")
        if lanes is None:
            continue
        for lane in lanes.findall("lane"):
            lane_id = lane.attrib.get("id")
            if not lane_id or lane_id.startswith(":"):
                continue  # ignore internal/junction lanes
            if q_primary in lane.attrib:
                q = float(lane.attrib[q_primary])
            elif q_fallback in lane.attrib:
                q = float(lane.attrib[q_fallback])
            else:
                q = 0.0
            rows.append((t, lane_id, q))
    df = pd.DataFrame(rows, columns=["t","lane_id","queue_len_m"]).sort_values("t")
    if df.empty:
        raise RuntimeError(f"No lane data parsed from {path}.")
    return df

def sum_over_lanes(df, lane_ids):
    if not lane_ids:
        return pd.Series(dtype=float)
    sub = df[df["lane_id"].isin(lane_ids)]
    return sub.groupby("t")["queue_len_m"].sum()

def sum_over_all(df):
    return df.groupby("t")["queue_len_m"].sum()

def union_index(*series):
    idx = np.array([], dtype=float)
    for s in series:
        idx = np.union1d(idx, s.index.values if hasattr(s.index, "values") else np.array(s.index))
    return idx

def slice_series(s, a, b):
    return s[(s.index >= a) & (s.index <= b)]

def fmt_mmss(sec):
    sec = int(round(sec))
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def mmss_formatter(x, pos): 
    return fmt_mmss(x)


def plot_group(title, times, s_bl, s_gw, filename):
    """Create one comparison figure (Baseline vs Green Wave) for a group and leave it open."""
    t_plot = times + LABEL_TIME_SHIFT_S  # shift only for display
    y_bl = s_bl.reindex(times, fill_value=0.0).values
    y_gw = s_gw.reindex(times, fill_value=0.0).values

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    try: fig.canvas.manager.set_window_title(title)
    except Exception: pass

    ax.plot(t_plot, y_bl, label="Baseline", linewidth=2,color="blue",alpha=0.9)
    ax.plot(t_plot, y_gw, label="Green Wave", linewidth=2,color="green",alpha=0.9)

    # One shared GRAY departure line, plus two colored arrival lines
    dep_plot    = EV_DEP    + LABEL_TIME_SHIFT_S
    arr_bl_plot = EV_ARR_BL + LABEL_TIME_SHIFT_S
    arr_gw_plot = EV_ARR_GW + LABEL_TIME_SHIFT_S

    ax.axvline(dep_plot,    linestyle=":",  linewidth=2, color="gray",   label="EV depart (15:30)")
    #ax.axvline(dep_plot,    label="EV depart (15:30)")
    ax.axvline(arr_bl_plot, linestyle="--", linewidth=1.8, color="#4a7cf9", label="EV BL arrive (+132s)",alpha = 0.5)
    ax.axvline(arr_gw_plot, linestyle="--", linewidth=1.8, color="#39ff5e", label="EV GW arrive (+89s)", alpha = 0.5)

    # Force x-axis: start at 15:29, end at 16:00 (labels)
    xmin = XMIN_ACTUAL + LABEL_TIME_SHIFT_S   # this equals 15:29 label
    xmax = END_S + LABEL_TIME_SHIFT_S         # 16:00 label
    ax.set_xlim(xmin, xmax)

    ax.set_title(title)
    ax.set_xlabel("Time [MM:SS]")
    ax.set_ylabel("Queue length [m]")
    ax.grid(True)
    ax.legend(ncol=2)
    ax.xaxis.set_major_formatter(FuncFormatter(mmss_formatter))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, filename), dpi=150)
    return fig

def summarize_block(name, s_bl, s_gw, a, b):
    bl = slice_series(s_bl, a, b)
    gw = slice_series(s_gw, a, b)
    return pd.DataFrame({
        "Group":    [name, name],
        "Scenario": ["Baseline", "Green Wave"],
        "Average [m]": [float(bl.mean()), float(gw.mean())],
        "Max [m]":     [float(bl.max()),  float(gw.max())],
        #"Min [m]":     [float(bl.min()),  float(gw.min())],
        "Δ avg (GW−BL) [m]": [0.0, float(gw.mean() - bl.mean())],
        "Δ max (GW−BL) [m]": [0.0, float(gw.max()  - bl.max())],
    })

# --------- Load files ----------
if XML_PATH_BaseLine == XML_PATH_GW:
    print("⚠️  WARNING: XML_PATH_BaseLine and XML_PATH_GW are identical.")

df_b = parse_lane_queue_xml(XML_PATH_BaseLine)
df_g = parse_lane_queue_xml(XML_PATH_GW)

# Optional: warn missing lanes
for tag, lanes, df in [("Baseline main", mainline_lanes, df_b),
                       ("Baseline side", side_lanes, df_b),
                       ("GW main", mainline_lanes, df_g),
                       ("GW side", side_lanes, df_g)]:
    missing = sorted(set(lanes) - set(df["lane_id"].unique()))
    if missing:
        print(f"ℹ️ Missing in {tag}: {missing}")

# Build time series (meters)
main_b = sum_over_lanes(df_b, mainline_lanes)
side_b = sum_over_lanes(df_b, side_lanes)
tot_b  = sum_over_all(df_b)

main_g = sum_over_lanes(df_g, mainline_lanes)
side_g = sum_over_lanes(df_g, side_lanes)
tot_g  = sum_over_all(df_g)

# Common timeline: union -> slice to [16:30, 17:00] (data range)
times_union = union_index(main_b, side_b, tot_b, main_g, side_g, tot_g)
mask = (times_union >= START_S) & (times_union <= END_S)
times = times_union[mask]

# ---------------- PLOTS (keep windows open) ----------------
fig1 = plot_group(
    "Cross Streets — Queue Length: Baseline vs Green Wave",
    times, side_b, side_g, "plot_cross_streets.png"
)
fig2 = plot_group(
    "Mainline (Sonnenallee) — Queue Length: Baseline vs Green Wave",
    times, main_b, main_g, "plot_mainline.png"
)
fig3 = plot_group(
    "Overall Network — Queue Length: Baseline vs Green Wave",
    times, tot_b,  tot_g,  "plot_overall.png"
)

# ---------------- TABLES ----------------
summary_side = summarize_block("Cross streets", side_b, side_g, START_S, END_S)
summary_main = summarize_block("Mainline",    main_b, main_g, START_S, END_S)
summary_tot  = summarize_block("Overall",     tot_b,  tot_g,  START_S, END_S)

summary_all = pd.concat([summary_side, summary_main, summary_tot], ignore_index=True)
summary_all = summary_all[["Group","Scenario","Average [m]","Max [m]","Δ avg (GW−BL) [m]","Δ max (GW−BL) [m]"]]

print("\n=== Summary (15:30–16:00) — Queue length [m] ===")
print(summary_all.to_string(index=False))

summary_csv = os.path.join(OUTDIR, "summary_1630_1700.csv")
summary_all.to_csv(summary_csv, index=False)
print(f"\nSaved tables & plots in: {os.path.abspath(OUTDIR)}")

# ---- BLOCK so the 3 windows stay open until you close them ----
plt.ioff()
try:
    plt.show(block=True)
except TypeError:
    plt.show()
    import time
    while plt.get_fignums():
        plt.pause(0.1)


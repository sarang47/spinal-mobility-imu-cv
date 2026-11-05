import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Paths
base = Path(__file__).parent.parent.parent  # or set manually
qualisys_path = base / "data" / "sample" / "STS_Hip_(L)_Angle.txt"
camera_path = base / "data" / "sample" / "pose2sim_2d_angles.csv"
output_table = base / "results" / "sample_outputs" / "hip_full_comparison_table.csv"
plot_path = base / "results" / "sample_outputs" / "hip_ascent_to_descent_overlay.png"

# Load Qualisys
q_lines = qualisys_path.read_text().splitlines()
q_data = []
for line in q_lines:
    parts = line.strip().split()
    if len(parts) == 3 and parts[0].isdigit():
        q_data.append([float(parts[1]), float(parts[2])])
q_df = pd.DataFrame(q_data, columns=["time_s", "hip_angle"])

# Load Camera
c_df = pd.read_csv(camera_path)
fps = 30
c_df["time_s"] = c_df.index / fps
c_df = c_df.rename(columns={"left_hip_angle_deg": "hip_angle"})

# Peak-based alignment
q_peaks, _ = find_peaks(q_df["hip_angle"], distance=80, height=140)
c_peaks, _ = find_peaks(c_df["hip_angle"], distance=25, height=140)
n = min(10, len(q_peaks), len(c_peaks))
offsets = c_df["time_s"].iloc[c_peaks[:n]].values - q_df["time_s"].iloc[q_peaks[:n]].values
constant_offset = np.median(offsets)
c_df["time_aligned"] = c_df["time_s"] - constant_offset
print(f"Offset used: {constant_offset:.2f} s")

# Valleys for ascent-to-descent
valleys, _ = find_peaks(-q_df["hip_angle"], distance=40, height=-120)

# Prepare plot
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
axes = axes.flatten()
t_norm = np.linspace(0, 100, 101)

results = []
stand = 0

for i in range(len(q_peaks)-1):
    peak = q_peaks[i]
    next_valleys = valleys[valleys > peak]
    if len(next_valleys) == 0:
        continue
    start_ascent = next_valleys[0]
    later_valleys = valleys[valleys > start_ascent]
    if len(later_valleys) == 0:
        continue
    end_descent = later_valleys[0]

    q_cycle = q_df.iloc[start_ascent:end_descent].copy()
    if len(q_cycle) < 15:
        continue

    t0 = q_cycle["time_s"].iloc[0]
    t1 = q_cycle["time_s"].iloc[-1]
    c_cycle = c_df[(c_df["time_aligned"] >= t0) & (c_df["time_aligned"] <= t1)].copy()

    # Time-normalise
    tq = np.linspace(0, 100, len(q_cycle))
    fq = interp1d(tq, q_cycle["hip_angle"].values, kind="linear", fill_value="extrapolate")
    q_norm = fq(t_norm)

    if len(c_cycle) > 5:
        tc = np.linspace(0, 100, len(c_cycle))
        fc = interp1d(tc, c_cycle["hip_angle"].values, kind="linear", fill_value="extrapolate")
        c_norm = fc(t_norm)
    else:
        continue

    # Metrics
    peak_q = np.max(q_norm)
    peak_c = np.max(c_norm)
    diff = peak_c - peak_q
    rmse = np.sqrt(mean_squared_error(q_norm, c_norm))
    mae = mean_absolute_error(q_norm, c_norm)
    icc = np.corrcoef(q_norm, c_norm)[0, 1]

    stand += 1
    results.append({
        "stand": stand,
        "hip_angle_qualisys_peak_deg": round(peak_q, 1),
        "hip_angle_camera_peak_deg": round(peak_c, 1),
        "difference_deg": round(diff, 1),
        "ICC": round(icc, 3),
        "RMSE_deg": round(rmse, 2),
        "MAE_deg": round(mae, 2)
    })

    # Plot
    ax = axes[stand-1]
    ax.plot(t_norm, c_norm, color="steelblue", label="Camera", linewidth=1.5)
    ax.plot(t_norm, q_norm, color="darkorange", label="Qualisys", linewidth=1.5)
    ax.set_title(f"Stand {stand} ICC {icc:.3f}")
    ax.grid(True, alpha=0.3)
    if stand == 1:
        ax.legend(fontsize=8)
    if stand > 5:
        ax.set_xlabel("Cycle (%)")
    if (stand-1) % 5 == 0:
        ax.set_ylabel("Hip angle (°)")

    if stand >= 10:
        break

# Hide unused axes
for j in range(stand, 10):
    axes[j].axis("off")

# Save plot
plt.suptitle("Ascent-to-descent windows (time-normalised) – camera vs Qualisys", fontsize=14)
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
print(f"Overlay plot saved → {plot_path}")

# Table with mean row
if results:
    res_df = pd.DataFrame(results)
    mean_row = {
        "stand": "Mean",
        "hip_angle_qualisys_peak_deg": round(res_df["hip_angle_qualisys_peak_deg"].mean(), 1),
        "hip_angle_camera_peak_deg": round(res_df["hip_angle_camera_peak_deg"].mean(), 1),
        "difference_deg": round(res_df["difference_deg"].mean(), 1),
        "ICC": round(res_df["ICC"].mean(), 3),
        "RMSE_deg": round(res_df["RMSE_deg"].mean(), 2),
        "MAE_deg": round(res_df["MAE_deg"].mean(), 2)
    }
    res_df = pd.concat([res_df, pd.DataFrame([mean_row])], ignore_index=True)
    res_df.to_csv(output_table, index=False)
    print("\nComparison table:")
    print(res_df.to_string(index=False))
    print(f"\nTable saved → {output_table}")

plt.show()

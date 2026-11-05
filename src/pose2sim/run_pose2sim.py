import os
from pathlib import Path
from Pose2Sim import Pose2Sim
import pandas as pd
import numpy as np

# Project folder (where your video is)
project_dir = Path(r"C:\Users\asd\Desktop\2nd_October_Qualisys_Trials")

# Create videos subfolder and copy/move video if needed
videos_dir = project_dir / "videos"
videos_dir.mkdir(exist_ok=True)

# Make sure the video is in the videos folder
video_src = project_dir / "2_oct_Camera_Qualisys.mp4"
video_dst = videos_dir / "cam1.mp4"
if video_src.exists() and not video_dst.exists():
    import shutil
    shutil.copy(video_src, video_dst)
    print(f"Video copied to {video_dst}")

# Create Config.toml (use forward slashes to avoid TOML escape issues)
config = f'''
[project]
project_dir = "{project_dir.as_posix()}"
multi_person = false
participant_height = 1.75
participant_mass = 70.0
frame_rate = 30

[pose]
pose_model = "Body"
mode = "lightweight"
overwrite_pose = true
tracking_mode = "sports2d"
det_frequency = 10
device = "CPU"
backend = "onnxruntime"
parallel_workers_pose = false
save_video = "none"

[calibration]
calibration_type = "calculate"

[triangulation]
min_cameras_for_triangulation = 1
reconstruction_error_threshold = 100

[filtering]
filter_type = "butterworth"
order = 4
cutoff = 6.0
filter_ik = true

[kinematics]
use_simple_model = true
'''

with open(project_dir / "Config.toml", "w") as f:
    f.write(config)

os.chdir(project_dir)
print("Starting Pose2Sim...")

Pose2Sim.poseEstimation()
Pose2Sim.synchronization()

try:
    Pose2Sim.personAssociation()
except:
    print("Person association skipped (single camera)")

try:
    Pose2Sim.triangulation()
except:
    print("Triangulation skipped (single camera)")

Pose2Sim.filtering()

try:
    Pose2Sim.markerAugmentation()
except:
    print("Marker augmentation skipped")

try:
    Pose2Sim.kinematics()
except:
    print("Kinematics skipped or limited (single camera)")

print("Pose2Sim finished")

# Extract results to CSV
mot_files = list((project_dir / "kinematics").glob("*.mot"))
if mot_files:
    mot_path = mot_files[0]
    with open(mot_path) as f:
        lines = f.readlines()
    header_end = next(i for i, line in enumerate(lines) if "endheader" in line.lower()) + 1
    col_names = lines[header_end].strip().split("\t")
    data = []
    for line in lines[header_end + 1:]:
        vals = line.strip().split("\t")
        if len(vals) == len(col_names):
            try:
                data.append([float(v) for v in vals])
            except:
                continue
    df = pd.DataFrame(data, columns=col_names)
    time = df.iloc[:, 0].values
    hip_col = next((c for c in df.columns if "hip_flexion_r" in c.lower()), None)
    knee_col = next((c for c in df.columns if "knee_angle_r" in c.lower()), None)
    results = pd.DataFrame({"time_s": time})
    if hip_col: results["hip_angle_right_deg"] = df[hip_col]
    if knee_col: results["knee_angle_right_deg"] = df[knee_col]
    if len(time) > 1:
        dt = np.mean(np.diff(time))
        if "hip_angle_right_deg" in results: results["hip_angular_velocity_right_deg_s"] = np.gradient(results["hip_angle_right_deg"], dt)
        if "knee_angle_right_deg" in results: results["knee_angular_velocity_right_deg_s"] = np.gradient(results["knee_angle_right_deg"], dt)
    results.to_csv(project_dir / "hip_knee_angles_velocities_degrees.csv", index=False)
    print("CSV saved: hip_knee_angles_velocities_degrees.csv")
else:
    print("No .mot file created (expected with single camera)")
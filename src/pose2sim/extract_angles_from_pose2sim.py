import json
import numpy as np
import pandas as pd
from pathlib import Path

# Correct path — JSONs are inside cam1_json
base = Path(__file__).parent.parent.parent  # or set manually
pose_dir = base / "data" / "sample" / "pose" / "cam1_json"
output_csv = base / "results" / "sample_outputs" / "pose2sim_2d_angles.csv"

# COCO keypoint indices (Body model)
L_HIP, L_KNEE, L_ANKLE = 11, 13, 15
L_SHOULDER = 5

def angle_between(a, b, c):
    """Angle at point b (in degrees) between points a-b-c."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

results = []

json_files = sorted(pose_dir.glob("*.json"))
if not json_files:
    print("No JSON files found in", pose_dir)
    exit()

print(f"Found {len(json_files)} JSON files")

for jf in json_files:
    with open(jf) as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        persons = data
    elif isinstance(data, dict) and "people" in data:
        persons = data["people"]
    elif isinstance(data, dict):
        persons = [data]
    else:
        continue
    
    if not persons:
        continue
    
    person = persons[0]
    kpts = np.array(person.get("pose_keypoints_2d", person.get("keypoints", [])))
    
    if kpts.size == 0:
        continue
    
    if kpts.ndim == 1:
        kpts = kpts.reshape(-1, 3)
    
    if len(kpts) < 16:
        continue
    
    hip = kpts[L_HIP, :2]
    knee = kpts[L_KNEE, :2]
    ankle = kpts[L_ANKLE, :2]
    shoulder = kpts[L_SHOULDER, :2]
    
    knee_angle = angle_between(hip, knee, ankle)
    hip_angle = angle_between(shoulder, hip, knee)
    
    frame = jf.stem.split("_")[-1] if "_" in jf.stem else jf.stem
    
    results.append({
        "file": jf.name,
        "frame": frame,
        "left_hip_angle_deg": hip_angle,
        "left_knee_angle_deg": knee_angle
    })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} frames → {output_csv}")
print(df.head(10))

# Spinal Mobility Analysis using MediaPipe + IMU

Camera-based quantification of spinal and functional mobility for Axial Spondyloarthritis (AxSpA) research.

This repository contains MediaPipe pose-estimation pipelines for clinical functional tests commonly used in AxSpA assessment (related to SPPB, ASPI, and BASMI/BASFI style metrics).

## Tests Implemented
- Balance / stance analysis (mediolateral sway)
- Stand-and-Reach
- Put-on-Socks
- 4-Meter Walk Test (4MWT)
- Bending / pick-up task
- Cervical rotation
- Thoracic rotation
- Getting up from floor
- 30-second Chair Stand

## Features
- MediaPipe Pose Landmarker (VIDEO mode)
- Joint and spinal angle calculation
- Time-series smoothing and peak detection
- Annotated output videos
- CSV + summary metrics export

## Project Structure
```text
spinal-mobility-imu-cv/
├── notebooks/               # Main analysis notebooks
├── src/vision/              # Shared pose utilities
├── data/sample/             # Sample inputs (small files only)
├── results/sample_outputs/  # Example outputs
├── requirements.txt
└── README.md

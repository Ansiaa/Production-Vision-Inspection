# Production-Vision-Inspection

Production-oriented vision inspection profolio project covering ROI preprocessing, anomaly detection, heatmap/overlay interpretation, operating-threshold tuning, drift monitoring, FastAPI serving, and an MFC demo client.

## 1. Project Overview

### 1-1. Problem

In manufacturing inspection systems, input image quality can vary because of changes in lighting, camera conditions, background, and production line status. When this happens, simple OK/NG classification alone is not enough for stable operation. It also becomes difficult to analyze the causes of false positives (FP) and false negatives (FN) and to determine appropriate follow-up actions.

### 1-2. Goal

The goal of this project is to build the following workflow from an operational manufacturing inspection perspective:

1. stably extract the inspection ROI from raw input images
2. standardize the input range based on the ROI
3. generate anomaly scores and heatmaps with an anomaly detection model
4. perform threshold-based decisions and FP/FN case analysis
5. define a structured JSON response format and practical extension points for inspection operation
6. organize the pipeline so that it can be extended to drift monitoring and retrain-trigger logic
7. expose the final inference flow through a FastAPI endpoint and an MFC demo client

## 2. Dataset

### 2-1. Dataset

- **VisA**
- Categories
  - `pcb4`
  - `cashew`

### 2-2. Why `pcb4` and `cashew`?

- **pcb4**  
  This category is closely aligned with manufacturing inspection scenarios, making it suitable for explaining ROI alignment, input standardization, and inspection-oriented anomaly detection workflow.

- **cashew**  
  This category has different shape and surface texture characteristics from `pcb4`, which makes it useful for comparing how the same preprocessing and anomaly detection pipeline behaves on a different type of object.

### 2-3. Data Split Summary

| Category | Normal | Anomaly |
| -------- | -----: | ------: |
| pcb4     |   1005 |     100 |
| cashew   |    500 |     100 |

## 3. Tech Stack

- Python
- OpenCV
- NumPy / Pandas / scikit-learn
- Lightweight PatchCore-style anomaly detection
- FastAPI
- MFC / C++

## 4. What This Project Shows

This project is intended to demonstrate the following capabilities in a workflow closer to real inspection operations:

- preprocessing design that reduces input variability by organizing the inspection ROI first
- interpretability through anomaly score, heatmap, and overlay outputs
- FP/FN analysis from an operational threshold perspective
- an operational structure that can be extended with drift monitoring, retrain-trigger logic, and future serving/UI integration

---

## 5. Current Baseline: ROI-only Preprocessing

### 5-1. Why ROI First?

Before comparing anomaly detection model performance, the first step is to standardize the input region that contains the inspection target.  
If too much background is included, the score distribution becomes unstable and the heatmap also becomes harder to interpret.

For this reason, this project first established an **ROI-only baseline**.

### 5-2. ROI Strategy

The current ROI is generated with the following procedure:

- estimate the background color from the image border
- calculate color difference from the background in LAB color space
- generate a foreground mask
- select the largest object region
- perform ROI cropping based on the bounding box

In other words, this project uses **foreground-mask-based ROI extraction**, not edge-based ROI extraction.

### 5-3. ROI-only Baseline Definition

The current baseline is defined as follows:

- **ROI / Crop only**
- **without CLAHE**
- used as the baseline input for heatmap and score comparison

---

## 6. ROI Result Examples

### 6-1. pcb4 ROI Example

| Raw | Foreground | ROI Debug |
| --- | --- | --- |
| ![](docs/images/roi/pcb4_raw.jpg) | ![](docs/images/roi/pcb4_foreground.png) | ![](docs/images/roi/pcb4_debug.jpg) |

### 6-2. cashew ROI Example

| Raw | Foreground | ROI Debug |
| --- | --- | --- |
| ![](docs/images/roi/cashew_raw.jpg) | ![](docs/images/roi/cashew_foreground.png) | ![](docs/images/roi/cashew_debug.jpg) |

### 6-3. ROI Baseline Output

The ROI-only output is used as the baseline model input for the later stages.

- `data/processed/pcb4/...`
- `data/processed/cashew/...`

---

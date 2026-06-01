# Walkthrough - H5 Dataset Builder Pipeline

This document details the newly added H5 dataset builder pipeline, local unit test validation results, and execution guides on Google Colab.

---

## Changes Made

### 1. New Dataset Builder CLI Script
- Created [build_wfmeta_h5.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scripts/build_wfmeta_h5.py):
  - Support for scanning raw split directories recursively.
  - Semicolon separator handling via project `trace_reader.py`.
  - Dense sequences calculation (`dir_seq`, `timestamp`, `time_seq` as clamped relative IAT, `diat_raw`, `diat_log`, `dir_ts`, `len_seq`) zero-padded or truncated to 5000.
  - Var-CNN original 7-feature `metadata` extraction on the whole raw trace.
  - ANOVA-ranked 74-feature `wfmeta` extraction mapped using the rank order list from JSON.
  - Memory-efficient HDF5 streaming writes with resize support for skipping files cleanly if errors occur.
  - Full logging of failures in `failed_traces.csv`.
  - Compilation of build schema, label distribution, and ranked order verification report files.

### 2. Unit Testing Suite
- Created [test_h5_builder.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/tests/test_h5_builder.py):
  - Generates dummy closed-world and open-world splits.
  - Traces processing and streaming H5 compilation test.
  - Group and dataset structure check, label alignment verify, report outputs checking.

---

## Validation Results

Running `python tests/test_h5_builder.py` produced:
- Successful compilation with code `0`.
- Outputs generated: `wfmeta_closed_world_v1.h5` and `wfmeta_open_world_v1.h5`.
- Dense and correctly sized datasets verified (Closed shape N=2, Open shape N=3, labels classes 100 and 101 respectively).
- Reports: `h5_build_report.txt`, `h5_schema_report.txt`, `label_distribution_report.txt`, and `wfmeta_order_report.txt` correctly populated.

---

## Kaggle Environment Commands Guide

The full H5 build is migrated to **Kaggle** to execute splits in parallel and bypass timeout/memory limitations.

### Input Dataset
Mounted at `/kaggle/input/wf-raw-splited-v1/` containing:
- `closed_world_split.tar.gz`
- `open_world_split.tar.gz`

---

### Notebook 1: Closed-World Build (`01_build_wfmeta_cw.ipynb`)
This notebook builds the Closed-World dataset `wfmeta_closed_world_v1.h5` (N = 100,000, shape classes = 100).

```bash
# 1. Setup local working directories
mkdir -p /kaggle/working/wf_split/closed
mkdir -p /kaggle/working/h5_build

# 2. Clone Var-CNN repository from GitHub
git clone https://github.com/ShortonKredit/Var-CNN.git /kaggle/working/Var-CNN

# 3. Extract Closed-World splits
tar -xzf /kaggle/input/wf-raw-splited-v1/closed_world_split.tar.gz -C /kaggle/working/wf_split/closed

# 4. Run builder script (Accelerator = None)
python /kaggle/working/Var-CNN/scripts/build_wfmeta_h5.py \
  --closed-dir /kaggle/working/wf_split/closed \
  --ranking-json /kaggle/working/Var-CNN/wfmeta/wfmeta_anova_ranked_features_v1.json \
  --output-dir /kaggle/working/h5_build \
  --seq-length 5000 \
  --build closed_world
```
Output results are published as a new Kaggle Dataset: `wfmeta-cw-h5-v1`.

---

### Notebook 2: Open-World Build (`02_build_wfmeta_ow.ipynb`)
This notebook builds the merged Open-World scenario-ready dataset `wfmeta_open_world_v1.h5` (N = 200,000, shape classes = 101).

```bash
# 1. Setup local working directories
mkdir -p /kaggle/working/wf_split/closed
mkdir -p /kaggle/working/wf_split/open
mkdir -p /kaggle/working/h5_build

# 2. Clone Var-CNN repository from GitHub
git clone https://github.com/ShortonKredit/Var-CNN.git /kaggle/working/Var-CNN

# 3. Extract Closed-World & Open-World splits
tar -xzf /kaggle/input/wf-raw-splited-v1/closed_world_split.tar.gz -C /kaggle/working/wf_split/closed
tar -xzf /kaggle/input/wf-raw-splited-v1/open_world_split.tar.gz -C /kaggle/working/wf_split/open

# 4. Run builder script (Accelerator = None)
python /kaggle/working/Var-CNN/scripts/build_wfmeta_h5.py \
  --closed-dir /kaggle/working/wf_split/closed \
  --open-dir /kaggle/working/wf_split/open \
  --ranking-json /kaggle/working/Var-CNN/wfmeta/wfmeta_anova_ranked_features_v1.json \
  --output-dir /kaggle/working/h5_build \
  --seq-length 5000 \
  --build open_world
```
Output results are published as a new Kaggle Dataset: `wfmeta-ow-h5-v1`.

---

### 5. Verify Compiled H5 Files Structure (Python Inference Test)
```python
import h5py

for name in ["wfmeta_closed_world_v1.h5", "wfmeta_open_world_v1.h5"]:
    h5_path = f"/kaggle/working/h5_build/{name}"
    try:
        with h5py.File(h5_path, "r") as f:
            print(f"\n=== Inspecting {name} ===")
            for split in f.keys():
                print(f"  Split: {split}")
                for ds in f[split].keys():
                    print(f"    Dataset: {ds} | Shape: {f[split][ds].shape} | Dtype: {f[split][ds].dtype}")
    except Exception as e:
        print(f"File {name} check skipped or failed: {str(e)}")
```


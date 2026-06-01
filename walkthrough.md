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

## Colab Commands Guide

### 1. Unpacking Splits to local SSD (SSD is much faster than Drive)
```bash
# Create local split directories
mkdir -p /content/wf_split/closed
mkdir -p /content/wf_split/open

# Extract closed-world split tar
tar -xzf /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz -C /content/wf_split/closed

# Extract open-world split tar
tar -xzf /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz -C /content/wf_split/open
```

### 2. Run H5 Build Pipeline
```bash
python /content/Var-CNN/scripts/build_wfmeta_h5.py \
  --closed-dir /content/wf_split/closed \
  --open-dir /content/wf_split/open \
  --ranking-json /content/Var-CNN/wfmeta/wfmeta_anova_ranked_features_v1.json \
  --output-dir /content/drive/MyDrive/WF/h5 \
  --seq-length 5000 \
  --build closed_world open_world
```

### 3. Verify Compiled H5 Files Structure
```python
import h5py

for name in ["wfmeta_closed_world_v1.h5", "wfmeta_open_world_v1.h5"]:
    h5_path = f"/content/drive/MyDrive/WF/h5/{name}"
    print(f"\n=== Inspecting {name} ===")
    with h5py.File(h5_path, "r") as f:
        for split in f.keys():
            print(f"  Split: {split}")
            for ds in f[split].keys():
                print(f"    Dataset: {ds} | Shape: {f[split][ds].shape} | Dtype: {f[split][ds].dtype}")
```

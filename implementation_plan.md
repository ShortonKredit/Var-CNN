# Implementation Plan - H5 Dataset Builder Pipeline (Kaggle Environment Workflow)

Create a new robust CLI script `scripts/build_wfmeta_h5.py` and parallel Kaggle notebooks to compile the Website Fingerprinting trace datasets into two scenario-ready HDF5 files:
1. `wfmeta_closed_world_v1.h5` (Closed-World: only monitored traces, shape N=100,000, 100 classes)
2. `wfmeta_open_world_v1.h5` (Open-World Scenario: merged monitored + unmonitored traces, total = 200,000 traces, where training_data = 133,000, validation_data = 7,000, and test_data = 60,000; 101 classes)

All H5 builds are migrated from Google Colab to **Kaggle** using parallel notebooks to maximize reliability and avoid notebook timeouts.

---

## User Review Required

> [!IMPORTANT]
> **Kaggle Processing Environment**:
> - Input dataset: `shortonkrediz/wf-raw-splited-v1` mounted at `/kaggle/input/wf-raw-splited-v1/`
> - Working directory: `/kaggle/working/` (repo cloned at `/kaggle/working/Var-CNN`, temp splits at `/kaggle/working/wf_split`, and output at `/kaggle/working/h5_build/`)
> - Accelerator: None (CPU Only)
> - Compiling closed-world and open-world scenario H5 files will be done in parallel using two separate Kaggle notebooks:
>   - **`01_build_wfmeta_cw.ipynb`**: builds closed-world scenario (output: `wfmeta_closed_world_v1.h5` inside Kaggle Dataset `wfmeta-cw-h5-v1`)
>   - **`02_build_wfmeta_ow.ipynb`**: builds open-world scenario (output: `wfmeta_open_world_v1.h5` inside Kaggle Dataset `wfmeta-ow-h5-v1`)
> 
> **Deterministic Ordering and No Shuffling**: The data splits must follow the existing sorted ordering in the directories (e.g. sorted file paths). Shuffling must NOT be performed during the H5 build phase. Random shuffling during training is handled separately by the Keras `data_generator.py` at runtime.
> 
> **ANOVA Ranking Compliance**: The new `wfmeta` dataset (shape `N x 74`) must sort the extracted statistical/WF features according to the exact rank list provided in `wfmeta/wfmeta_anova_ranked_features_v1.json`.

---

## Open Questions

None. The requirements are fully specified and validated by local sanity checks.

---

## Proposed Changes

### Build Pipeline Component

---

#### [NEW] [build_wfmeta_h5.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scripts/build_wfmeta_h5.py)

This file will contain the main dataset builder script. Key features:
- **CLI Options**: Supports parameters for input paths, ranking JSON, output directory, target sequence length, and which scenario to build.
- **Robust Path Processing**:
  - Scanning splits (`training_data`, `validation_data`, `test_data`) in `--closed-dir` and `--open-dir`.
  - Parsers for label and `site_name` / `trace_name`.
- **HDF5 Streaming Write**:
  - Pre-scans file counts to avoid dynamic array resizing.
  - Pre-creates HDF5 groups and datasets with fixed shapes.
  - Processes files in chunks to avoid out-of-memory errors on Google Colab.
- **Sequence Transformation Logic**:
  - Sorts trace df by `packet_index` if available.
  - Translates timestamp sequence to relative time: `timestamp_rel = timestamp - timestamp[0]`.
  - Calculates IAT sequence: `iat[0] = 0.0` and `iat[i] = max(0.0, timestamp_rel[i] - timestamp_rel[i-1])` to clamp inversions.
  - Generates the required streams: `dir_seq`, `timestamp`, `time_seq`, `diat_raw`, `diat_log`, `dir_ts`, `len_seq` (shape: `(N, 5000, 1)`, dtype: `float32`).
  - Pads with zeros up to length 5000, or truncates if longer.
- **Metadata Extraction**:
  - Var-CNN original 7-feature `metadata` dataset (dtype: `float32`).
  - Extract WFMeta using `wfmeta/features.py` and sorting according to the ANOVA ranking JSON (dtype: `float32`).
- **Reports Generation**:
  - Generates `h5_build_report.txt`, `h5_schema_report.txt`, `label_distribution_report.txt`, and `wfmeta_order_report.txt` in the output directory after execution.
- **Error Log**:
  - Outputs `failed_traces.csv` with trace split, world, file path, and error message if failures occur.

---

## Verification Plan

### Local Sanity Check (Executed and Confirmed)
The trace parsing and sequence builder logic has been verified locally on the sample trace file `0_123movies.is_0.csv`.

**Verification Output Logs:**
```text
Reading sample CSV: c:/Users/ADMIN/Desktop/Var-CNN/0_123movies.is_0.csv
CSV columns: ['packet_index', 'timestamp', 'length', 'direction', 'num_cells', 'ack']
First 3 rows:
   packet_index     timestamp  length  direction  num_cells  ack
0             0  1.489000e+09     609          1          1    0
1             1  1.489000e+09      66         -1          0    1
2             2  1.489000e+09     609         -1          1    0
Sorted by packet_index

--- SHAPES ---
dir_seq: (5000, 1)
timestamp: (5000, 1)
time_seq: (5000, 1)
diat_raw: (5000, 1)
diat_log: (5000, 1)
dir_ts: (5000, 1)
len_seq: (5000, 1)
metadata: (7,)
wfmeta: (74,)

--- SAMPLE VALUES (first 5 values for sequences) ---
dir_seq: [ 1. -1. -1.  1.  1.]
timestamp: [0.         0.03675914 0.03725171 0.03726912 0.03773713]
time_seq: [0.0000000e+00 3.6759138e-02 4.9257278e-04 1.7404556e-05 4.6801567e-04]
diat_raw: [ 0.0000000e+00 -3.6759138e-02 -4.9257278e-04  1.7404556e-05 4.6801567e-04]
diat_log: [ 0.0000000e+00 -3.6099635e-02 -4.9245154e-04  1.7404405e-05 4.6790618e-04]
dir_ts: [ 0.         -0.03675914 -0.03725171  0.03726912  0.03773713]
len_seq: [609.  66. 609.  66. 609.]
metadata: [4.8170000e+03 4.1240000e+03 6.9300000e+02 8.5613453e-01 1.4386548e-01 3.3012615e+01 6.8533556e-03]

First 5 ANOVA ranked features:
  Rank #1: total_len_fwd_packets = 385645.0
  Rank #2: total_len_bwd_packets = 5926461.0
  Rank #3: total_bwd_packets = 4124.0
  Rank #4: total_fwd_packets = 693.0
  Rank #5: fwd_burst_count = 626.0
```

### Optional/Nice-To-Have Verification (Not blockers)
1. **Mock H5 Build test**:
   Write a small test script (e.g., `tests/test_h5_builder.py`) that creates a mock directory structure with 2-3 CSV files for CW and OW, runs `build_wfmeta_h5.py` CLI against these mock directories, and asserts the generated H5 file structure, group presence, shapes, metadata alignment, and report generations.
   - Command: `python tests/test_h5_builder.py`
2. **Pyright Linting**:
   Ensure `scripts/build_wfmeta_h5.py` compiles without lint/type checker errors.
   - Command: `npx pyright scripts/build_wfmeta_h5.py`

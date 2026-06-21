# Walkthrough - H5 Dataset Builder Pipeline

This document details the dataset building pipeline, Keras 3 compatibility minimal patch, Deep Fingerprinting baseline integration, and Closed-World low-data evaluation integration.

---

## 1. H5 Dataset Builder Pipeline

The H5 dataset builder pipeline includes scan parsing, IAT calculations, feature extraction, and parallelized compilation.

### Changes Made
- Created [build_wfmeta_h5.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scripts/build_wfmeta_h5.py) supporting directory recursive scans, semicolon separators, relative clamp IAT, ANOVA feature indexing, and resize support.
- Created [test_h5_builder.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/tests/test_h5_builder.py) unit testing suite.

---

## 2. Keras 3 Compatibility Minimal Patch

Minimal changes were made to resolve compilation errors under Keras 3:
- Preserved generator pattern by removing custom wrappers.
- Selective shuffling for training data, validation/test left sequential.
- Weights Isolated to distinct directories under `/kaggle/working/outputs/<model_id>/`.
- Overwrite protection backing up old weights to `.bak`.

---

## 3. Deep Fingerprinting (DF) Baseline Integration

Integrated the DF baseline architecture while keeping Var-CNN intact:
- Created [run_dfmodel.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/run_dfmodel.py) dedicated model runner.
- Created [scratch/create_df_configs.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scratch/create_df_configs.py) baseline configuration generator.
- Added 16 JSON configurations inside `configs/df/`.

---

## 4. Closed-World Low-Data Evaluation Integration ($k \in \{60, 80, 100, 300, 550\}$)

We integrated the Closed-World low-data website fingerprinting evaluation pipeline.

### Created Files
* **[scratch/create_lowdata_indices.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scratch/create_lowdata_indices.py)**: Generates deterministic nested subset indices (`cw60 ⊂ cw80 ⊂ cw100 ⊂ cw300 ⊂ cw550`) by shuffling training indices per class once using a RandomState with seed 42.
* **[scratch/evaluate_lowdata_ensembles.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scratch/evaluate_lowdata_ensembles.py)**: Performs softmax prediction averaging for constituent models in ensembles (Var-CNN, WFMeta-DT, WFMeta-DTL, WFMeta-DIAT-L), verifies output shapes of `(10000, 100)`, and raises a `FileNotFoundError` if files are missing. Supports levels 60, 80, 100, 300, and 550.
* **[scratch/create_lowdata_configs.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scratch/create_lowdata_configs.py)**: Config generator script to write the 30 JSON configurations in `configs/lowdata/`.
* **[scratch/test_lowdata_pipeline.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/scratch/test_lowdata_pipeline.py)**: Test runner script simulating a complete training and evaluation run on a local dummy dataset.

### Modified Files
* **[data_generator.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/data_generator.py)**: Modified to read and handle `train_indices_file` configurations for the training split while keeping validation and test flows unchanged. Implements safe sequential reads by mapping positions in the shuffled order to original indices and querying H5 safely.
* **[run_model.py](file:///c:/Users/ADMIN/Desktop/Var-CNN/run_model.py)**: Modified to calculate epoch lengths (`steps_per_epoch`) based on the size of the active subset indices when `train_indices_file` is specified in the active config.

### Verification & SHA256 Checksums
The dry-run pipeline test was successfully verified. The SHA256 checksums of the generated index files are:
* **`cw60_train_indices.npy`**:  `7fad121d3584c356c1f9a9ee8c4f93e0e11abf09def9811d0c555be843c1e78b`
* **`cw80_train_indices.npy`**:  `055e0da2811573ee404148690512f4baec88c74b67e28a6d459b431455241c6f`
* **`cw100_train_indices.npy`**: `2882b0ee6a507f999b61ae32fe7af1b850e485332b9321d8568837875d32056e`
* **`cw300_train_indices.npy`**: `76903de00c5b5317dfa3ad493379c8222506b4cf1a1545200ef74bc6c974d611`
* **`cw550_train_indices.npy`**: `91230359349d4e099994985b7ce0037173d1712b250f9de8a7075b8bd8a0f72b`

### Max Epoch Rules
The configurations set `var_cnn_max_epochs` based on the model groups:
- **Max Epochs = 50**: Original metadata models (`cw{k}_dir_metadata` and `cw{k}_time_metadata`).
- **Max Epochs = 150**: All 4 WFMeta10 models (`cw{k}_dir_wfmeta10`, `cw{k}_time_wfmeta10`, `cw{k}_len_wfmeta10`, `cw{k}_diat_wfmeta10`).

### Git Synchronization
* **Commit Hash**: `d00f06a` (Support levels 60 and 80 in Closed-World low-data evaluation and update max epoch rules)
* Pushed successfully to `main` branch on GitHub.

import os
import json
import numpy as np
import h5py
import subprocess
import sys

# Paths
scratch_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scratch_dir)
dummy_h5_path = os.path.join(scratch_dir, "dummy.h5")
temp_config_path = os.path.join(scratch_dir, "temp_config.json")
outputs_dir = os.path.join(scratch_dir, "outputs")

print("1. Creating dummy H5 dataset...")
# Create dummy datasets (10 samples total, 100 classes, seq length 5000)
with h5py.File(dummy_h5_path, "w") as f:
    for split, size in [("training_data", 6), ("validation_data", 2), ("test_data", 2)]:
        grp = f.create_group(split)
        
        # Labels: one-hot encoded (shape: N x 100)
        labels = np.zeros((size, 100), dtype=np.float32)
        for i in range(size):
            labels[i, i % 100] = 1.0
        grp.create_dataset("labels", data=labels)
        
        # Sequences (shape: N x 5000 x 1)
        dir_seq = np.random.choice([-1.0, 0.0, 1.0], size=(size, 5000, 1)).astype(np.float32)
        grp.create_dataset("dir_seq", data=dir_seq)
        
        # Metadata (shape: N x 7)
        metadata = np.random.randn(size, 7).astype(np.float32)
        grp.create_dataset("metadata", data=metadata)

print("2. Generating temporary config...")
config = {
    "scenario": "closed_world",
    "processed_h5": dummy_h5_path,
    "num_classes": 100,
    "num_mon_sites": 100,
    "num_mon_inst_train": 1,
    "num_mon_inst_test": 1,
    "num_unmon_sites_train": 0,
    "num_unmon_sites_test": 0,
    "batch_size": 2,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 1,
    "var_cnn_max_epochs": 1,  # Train for just 1 epoch
    "var_cnn_base_patience": 2,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": outputs_dir,
    "model_id": "temp_test",
    "sequence_dataset": "dir_seq",
    "sequence_input_name": "dir_input",
    "sequence_model_suffix": "dir",
    "metadata_dataset": "metadata",
    "metadata_type": "metadata",
    "wfmeta_k": 7
}

with open(temp_config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("3. Executing run_model.py end-to-end...")
cmd = [
    sys.executable,
    os.path.join(project_dir, "run_model.py"),
    "--config", temp_config_path
]

print(f"Running command: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)

print("--- PROCESS STDOUT ---")
print(res.stdout)
print("--- PROCESS STDERR ---")
print(res.stderr)

assert res.returncode == 0, f"Execution failed with return code {res.returncode}"

print("4. Verifying outputs...")
target_dir = os.path.join(outputs_dir, "temp_test")
expected_files = [
    "temp_test.weights.h5",
    "temp_test_model.npy",
    "temp_test_result.json",
    "temp_test.config.json"
]

all_exists = True
for f in expected_files:
    path = os.path.join(target_dir, f)
    if os.path.exists(path):
        print(f"  [PASS] Found output file: {path} (Size: {os.path.getsize(path)} bytes)")
    else:
        print(f"  [FAIL] Missing output file: {path}")
        all_exists = False

if all_exists:
    print("\nEnd-to-end dry-run verification completed successfully!")
else:
    print("\nEnd-to-end verification FAILED due to missing output files.")
    sys.exit(1)

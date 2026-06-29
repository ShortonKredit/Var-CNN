import os
import json
import subprocess
import sys
import numpy as np

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:")
        print(res.stdout)
        print("STDERR:")
        print(res.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    else:
        print("STDOUT:")
        print(res.stdout)

def main():
    print("=== STARTING LOW-DATA PIPELINE DRY RUN ===")
    
    # 1. Modify one config to use our local dummy.h5 and run 1 epoch
    config_path = "configs/lowdata/cw100_dir_metadata.json"
    print(f"Reading original config: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    # Override configuration parameters for local dry run
    cfg["processed_h5"] = "scratch/dummy_closed_world.h5"
    cfg["train_indices_file"] = "lowdata_indices/cw100_train_indices.npy"
    cfg["var_cnn_max_epochs"] = 1
    cfg["output_dir"] = "scratch/outputs"
    
    temp_config_path = "scratch/temp_lowdata_test_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Temporary dry-run config written to: {temp_config_path}")
    
    # 2. Run run_model.py on this config
    # We run it via subprocess to test the command-line interface directly
    pred_path = "scratch/outputs/cw100_dir_metadata/cw100_dir_metadata_model.npy"
    if not os.path.exists(pred_path):
        run_cmd(f"python run_model.py --config {temp_config_path}")
    else:
        print(f"Prediction file already exists at {pred_path}, skipping training step.")
    
    # 3. Verify that predictions files are generated at scratch/outputs/cw100_dir_metadata/cw100_dir_metadata_model.npy
    pred_path = "scratch/outputs/cw100_dir_metadata/cw100_dir_metadata_model.npy"
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file was not generated at: {pred_path}")
    
    # Load and print shape
    pred = np.load(pred_path)
    print(f"Successfully generated prediction file of shape: {pred.shape}")
    assert pred.shape == (10000, 100), f"Expected shape (10000, 100), got {pred.shape}"
    
    # 4. Now, to test evaluate_lowdata_ensembles.py, let's copy this prediction file to make dummy outputs for the other models in the ensemble
    # This lets us test evaluate_lowdata_ensembles.py with a full set of files
    required_models = [
        "cw100_time_metadata",
        "cw100_dir_wfmeta10",
        "cw100_time_wfmeta10",
        "cw100_len_wfmeta10",
        "cw100_diat_wfmeta10"
    ]
    
    for model_id in required_models:
        model_dir = os.path.join("scratch/outputs", model_id)
        os.makedirs(model_dir, exist_ok=True)
        dest_path = os.path.join(model_dir, f"{model_id}_model.npy")
        # Save dummy predictions (using the same prediction array)
        np.save(dest_path, pred)
        print(f"Created dummy predictions for testing: {dest_path}")
        
    # 5. Run evaluate_lowdata_ensembles.py
    run_cmd("python scratch/evaluate_lowdata_ensembles.py --k 100 --outputs_dir scratch/outputs --h5_path scratch/dummy_closed_world.h5")
    
    print("\n=== DRY RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()

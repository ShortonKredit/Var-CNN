import os
import json
import glob

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_dir, "configs", "ow_retrain")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory for retrain configs: {target_dir}")

    # 1. Process Var-CNN configs from configs/full/
    full_configs_dir = os.path.join(project_dir, "configs", "full")
    var_cnn_files = glob.glob(os.path.join(full_configs_dir, "ow_*.json"))
    print(f"Found {len(var_cnn_files)} Var-CNN Open-World configurations in full/.")

    for fpath in var_cnn_files:
        filename = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Determine CW model ID by replacing 'ow_' with 'cw_'
        ow_model_id = config.get("model_id")
        if not ow_model_id or not ow_model_id.startswith("ow_"):
            print(f"Skipping {filename}: model_id is not prefixed with ow_")
            continue
            
        cw_model_id = ow_model_id.replace("ow_", "cw_", 1)
        
        # Add pre-trained CW weights path
        weights_path = f"/kaggle/working/outputs/{cw_model_id}/{cw_model_id}.weights.h5"
        config["pretrained_closed_world_weights"] = weights_path
        
        # Save to targets
        target_path = os.path.join(target_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"  Generated Var-CNN config: {filename} -> weights: {weights_path}")

    # 2. Process Deep Fingerprinting configs from configs/df/
    df_configs_dir = os.path.join(project_dir, "configs", "df")
    # We only process open world files (df_ow_*) that have a corresponding closed-world model
    df_files = glob.glob(os.path.join(df_configs_dir, "df_ow_*.json"))
    print(f"Found {len(df_files)} Deep Fingerprinting Open-World configurations in df/.")

    for fpath in df_files:
        filename = os.path.basename(fpath)
        # Avoid processing small versions if they don't have matching CW configs
        if "_small_" in filename:
            print(f"Skipping {filename}: small version does not have matching CW counterpart.")
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        ow_model_id = config.get("model_id")
        if not ow_model_id or not ow_model_id.startswith("df_ow_"):
            print(f"Skipping {filename}: model_id is not prefixed with df_ow_")
            continue
            
        cw_model_id = ow_model_id.replace("df_ow_", "df_cw_", 1)
        
        # Add pre-trained CW weights path
        weights_path = f"/kaggle/working/outputs/{cw_model_id}/{cw_model_id}.weights.h5"
        config["pretrained_closed_world_weights"] = weights_path
        
        # Save to targets
        target_path = os.path.join(target_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"  Generated DF config: {filename} -> weights: {weights_path}")

    print("\nAll retrain configurations generated successfully under configs/ow_retrain/!")

if __name__ == "__main__":
    main()

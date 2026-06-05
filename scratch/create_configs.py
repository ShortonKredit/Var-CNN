import json
import os
import shutil

# 1. Define paths on Kaggle
SMALL_H5_PATH = "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-small-h5/wfmeta_open_world_small.h5"
WTFPAD_H5_PATH = "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-wtfpad-h5/wfmeta_open_world_wtfpad.h5"

# 2. Get directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
configs_dir = os.path.join(project_dir, "configs")
full_dir = os.path.join(configs_dir, "full")

# Clean existing small and wtfpad directories if they exist, but DO NOT delete configs/ root or configs/full/
small_dir = os.path.join(configs_dir, "small")
wtfpad_dir = os.path.join(configs_dir, "wtfpad")

if os.path.exists(small_dir):
    shutil.rmtree(small_dir)
if os.path.exists(wtfpad_dir):
    shutil.rmtree(wtfpad_dir)

# Clean full_dir if it exists, to avoid duplicates when re-running
# (Keep fullconfigs safe on rerun, only create if missing)
os.makedirs(full_dir, exist_ok=True)

# 3. Move old config files from configs/ root to configs/full/
for item in os.listdir(configs_dir):
    item_path = os.path.join(configs_dir, item)
    # Only move files, not directories (like small, wtfpad)
    if os.path.isfile(item_path) and item.endswith(".json"):
        shutil.move(item_path, os.path.join(full_dir, item))
        print(f"Moved old config: {item} -> configs/full/")

# 4. Define sequence streams mapping (excluding 'len')
sequences = {
    "dir": {
        "sequence_dataset": "dir_seq",
        "sequence_input_name": "dir_input",
        "sequence_model_suffix": "dir"
    },
    "time": {
        "sequence_dataset": "time_seq",
        "sequence_input_name": "time_input",
        "sequence_model_suffix": "time"
    },
    "diat_raw": {
        "sequence_dataset": "diat_raw",
        "sequence_input_name": "diat_raw_input",
        "sequence_model_suffix": "diat_raw"
    },
    "diat_log": {
        "sequence_dataset": "diat_log",
        "sequence_input_name": "diat_log_input",
        "sequence_model_suffix": "diat_log"
    },
    "dir_ts": {
        "sequence_dataset": "dir_ts",
        "sequence_input_name": "dir_ts_input",
        "sequence_model_suffix": "dir_ts"
    }
}

# 5. Define metadata streams mapping
metadatas = {
    "metadata": {
        "metadata_dataset": "metadata",
        "metadata_type": "metadata",
        "wfmeta_k": 7
    },
    "wfmeta10": {
        "metadata_dataset": "wfmeta",
        "metadata_type": "wfmeta10",
        "wfmeta_k": 10
    },
    "wfmeta20": {
        "metadata_dataset": "wfmeta",
        "metadata_type": "wfmeta20",
        "wfmeta_k": 20
    }
}

# 6. Base config templates for Open World (correct instance counts)
base_config_small = {
    "scenario": "open_world",
    "processed_h5": SMALL_H5_PATH,
    "num_classes": 101,
    "num_mon_sites": 100,
    "num_mon_inst_train": 170,       # Corrected: 170 training monitored traces per site
    "num_mon_inst_test": 20,         # Corrected: 20 test monitored traces per site
    "num_unmon_sites_train": 9500,   # Corrected: 9500 training unmonitored traces
    "num_unmon_sites_test": 10000,   # Corrected: 10000 test unmonitored traces
    "batch_size": 256,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 10,
    "var_cnn_max_epochs": 50,  # Default, will be updated dynamically
    "var_cnn_base_patience": 5,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": "/kaggle/working/outputs/"
}

base_config_wtfpad = base_config_small.copy()
base_config_wtfpad["processed_h5"] = WTFPAD_H5_PATH

# 7. Generate configs for Small dataset
for seq_name, seq_info in sequences.items():
    for meta_name, meta_info in metadatas.items():
        model_id = f"ow_small_{seq_name}_{meta_name}"
        config = base_config_small.copy()
        config["model_id"] = model_id
        config.update(seq_info)
        config.update(meta_info)
        
        # Set max epochs dynamically: wfmeta10 and wfmeta20 get 150 epochs, metadata gets 50
        if meta_name in ["wfmeta10", "wfmeta20"]:
            config["var_cnn_max_epochs"] = 150
        else:
            config["var_cnn_max_epochs"] = 50
            
        # Save to configs/small/<metadata_type>/
        target_dir = os.path.join(small_dir, meta_name)
        os.makedirs(target_dir, exist_ok=True)
        
        filepath = os.path.join(target_dir, f"{model_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"Generated Raw Small Config: {filepath}")

# 8. Generate configs for WTF-PAD dataset
for seq_name, seq_info in sequences.items():
    for meta_name, meta_info in metadatas.items():
        model_id = f"ow_wtfpad_{seq_name}_{meta_name}"
        config = base_config_wtfpad.copy()
        config["model_id"] = model_id
        config.update(seq_info)
        config.update(meta_info)
        
        # Set max epochs dynamically: wfmeta10 and wfmeta20 get 150 epochs, metadata gets 50
        if meta_name in ["wfmeta10", "wfmeta20"]:
            config["var_cnn_max_epochs"] = 150
        else:
            config["var_cnn_max_epochs"] = 50
            
        # Determine target folder name for wtfpad
        if meta_name == "wfmeta20":
            if seq_name in ["dir", "time", "diat_raw"]:
                folder_name = "wfmeta20_part1"
            else:
                folder_name = "wfmeta20_part2"
        else:
            folder_name = meta_name
            
        target_dir = os.path.join(wtfpad_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        filepath = os.path.join(target_dir, f"{model_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"Generated WTF-PAD Config: {filepath}")

print("\nConfigs successfully generated and structured for 6+1 Notebooks!")

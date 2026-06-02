import json
import os

configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
os.makedirs(configs_dir, exist_ok=True)

# 1. Define sequence streams mapping
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
    "len": {
        "sequence_dataset": "len_seq",
        "sequence_input_name": "len_input",
        "sequence_model_suffix": "len"
    },
    "diat_log": {
        "sequence_dataset": "diat_log",
        "sequence_input_name": "diat_log_input",
        "sequence_model_suffix": "diat_log"
    },
    "diat_raw": {
        "sequence_dataset": "diat_raw",
        "sequence_input_name": "diat_raw_input",
        "sequence_model_suffix": "diat_raw"
    },
    "dir_ts": {
        "sequence_dataset": "dir_ts",
        "sequence_input_name": "dir_ts_input",
        "sequence_model_suffix": "dir_ts"
    }
}

# 2. Define metadata streams mapping
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
    }
}

# 3. Base config template
base_config = {
    "scenario": "closed_world",
    "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5",
    "num_classes": 100,
    "num_mon_sites": 100,
    "num_mon_inst_train": 855,
    "num_mon_inst_test": 100,
    "num_unmon_sites_train": 0,
    "num_unmon_sites_test": 0,
    "batch_size": 256,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 10,
    "var_cnn_max_epochs": 150,
    "var_cnn_base_patience": 5,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": "/kaggle/working/outputs/"
}

# Base config template for Open World
base_config_ow = {
    "scenario": "open_world",
    "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-h5-v2/wfmeta_open_world_v2.h5",
    "num_classes": 101,
    "num_mon_sites": 100,
    "num_mon_inst_train": 855,
    "num_mon_inst_test": 100,
    "num_unmon_sites_train": 47500,
    "num_unmon_sites_test": 50000,
    "batch_size": 256,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 10,
    "var_cnn_max_epochs": 150,
    "var_cnn_base_patience": 5,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": "/kaggle/working/outputs/"
}

# 4. Generate the 12 config files for CW and 12 for OW
for seq_name, seq_info in sequences.items():
    for meta_name, meta_info in metadatas.items():
        # --- Closed World ---
        cw_model_id = f"cw_{seq_name}_{meta_name}"
        cw_config = base_config.copy()
        cw_config["model_id"] = cw_model_id
        cw_config.update(seq_info)
        cw_config.update(meta_info)
        
        cw_path = os.path.join(configs_dir, f"{cw_model_id}.json")
        with open(cw_path, "w", encoding="utf-8") as f:
            json.dump(cw_config, f, indent=4)
        print(f"Generated CW config: {cw_path}")
        
        # --- Open World ---
        ow_model_id = f"ow_{seq_name}_{meta_name}"
        ow_config = base_config_ow.copy()
        ow_config["model_id"] = ow_model_id
        ow_config.update(seq_info)
        ow_config.update(meta_info)
        
        ow_path = os.path.join(configs_dir, f"{ow_model_id}.json")
        with open(ow_path, "w", encoding="utf-8") as f:
            json.dump(ow_config, f, indent=4)
        print(f"Generated OW config: {ow_path}")

print("\nAll configurations generated successfully in 'configs/' directory.")

import json
import os

configs_dir = "configs/df"
os.makedirs(configs_dir, exist_ok=True)

# Datasets definition
datasets = {
    "cw": {
        "scenario": "closed_world",
        "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5",
        "num_classes": 100,
        "num_mon_inst_train": 855,
        "num_mon_inst_test": 100,
        "num_unmon_sites_train": 0,
        "num_unmon_sites_test": 0
    },
    "ow": {
        "scenario": "open_world",
        "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-h5-v2/wfmeta_open_world_v2.h5",
        "num_classes": 101,
        "num_mon_inst_train": 855,
        "num_mon_inst_test": 100,
        "num_unmon_sites_train": 47500,
        "num_unmon_sites_test": 50000
    },
    "ow_small": {
        "scenario": "open_world",
        "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-small-h5/wfmeta_open_world_small.h5",
        "num_classes": 101,
        "num_mon_inst_train": 170,
        "num_mon_inst_test": 20,
        "num_unmon_sites_train": 9500,
        "num_unmon_sites_test": 10000
    },
    "wtfpad": {
        "scenario": "open_world",
        "processed_h5": "/kaggle/input/datasets/shortonkrediz/wfmeta-open-world-wtfpad-h5/wfmeta_open_world_wtfpad.h5",
        "num_classes": 101,
        "num_mon_inst_train": 170,
        "num_mon_inst_test": 20,
        "num_unmon_sites_train": 9500,
        "num_unmon_sites_test": 10000
    }
}

# Representations definition
reps = {
    "dir": {
        "sequence_dataset": "dir_seq",
        "sequence_input_name": "dir_input"
    },
    "dlog": {
        "sequence_dataset": "diat_log",
        "sequence_input_name": "diat_log_input"
    },
    "draw": {
        "sequence_dataset": "diat_raw",
        "sequence_input_name": "diat_raw_input"
    },
    "dts": {
        "sequence_dataset": "dir_ts",
        "sequence_input_name": "dir_ts_input"
    }
}

# Generate 12 configs
for ds_name, ds_info in datasets.items():
    for rep_name, rep_info in reps.items():
        model_id = f"df_{ds_name}_{rep_name}"
        config = {
            "model_name": "df",
            "model_id": model_id,
            "scenario": ds_info["scenario"],
            "processed_h5": ds_info["processed_h5"],
            "sequence_dataset": rep_info["sequence_dataset"],
            "sequence_input_name": rep_info["sequence_input_name"],
            "num_classes": ds_info["num_classes"],
            "num_mon_sites": 100,
            "num_mon_inst_train": ds_info["num_mon_inst_train"],
            "num_mon_inst_test": ds_info["num_mon_inst_test"],
            "num_unmon_sites_train": ds_info["num_unmon_sites_train"],
            "num_unmon_sites_test": ds_info["num_unmon_sites_test"],
            "batch_size": 256,
            "seq_length": 5000,
            "df_epochs": 30,
            "df_base_patience": 5,
            "output_dir": "/kaggle/working/outputs/"
        }
        
        filename = f"df_{ds_name}_{rep_name}.json"
        config_path = os.path.join(configs_dir, filename)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Generated config file: {config_path}")

print("All 16 configurations generated successfully.")

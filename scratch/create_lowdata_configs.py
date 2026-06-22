import os
import json

def main():
    configs_dir = "configs/lowdata"
    os.makedirs(configs_dir, exist_ok=True)

    k_values = [40, 50, 60, 80, 100, 300, 550]
    
    # 6 models mappings definition
    models_def = [
        {
            "name_suffix": "dir_metadata",
            "sequence_dataset": "dir_seq",
            "sequence_input_name": "dir_input",
            "sequence_model_suffix": "dir",
            "metadata_dataset": "metadata",
            "metadata_type": "metadata",
            "wfmeta_k": 7
        },
        {
            "name_suffix": "time_metadata",
            "sequence_dataset": "time_seq",
            "sequence_input_name": "time_input",
            "sequence_model_suffix": "time",
            "metadata_dataset": "metadata",
            "metadata_type": "metadata",
            "wfmeta_k": 7
        },
        {
            "name_suffix": "dir_wfmeta10",
            "sequence_dataset": "dir_seq",
            "sequence_input_name": "dir_input",
            "sequence_model_suffix": "dir",
            "metadata_dataset": "wfmeta",
            "metadata_type": "wfmeta10",
            "wfmeta_k": 10
        },
        {
            "name_suffix": "time_wfmeta10",
            "sequence_dataset": "time_seq",
            "sequence_input_name": "time_input",
            "sequence_model_suffix": "time",
            "metadata_dataset": "wfmeta",
            "metadata_type": "wfmeta10",
            "wfmeta_k": 10
        },
        {
            "name_suffix": "len_wfmeta10",
            "sequence_dataset": "len_seq",
            "sequence_input_name": "len_input",
            "sequence_model_suffix": "len",
            "metadata_dataset": "wfmeta",
            "metadata_type": "wfmeta10",
            "wfmeta_k": 10
        },
        {
            "name_suffix": "diat_wfmeta10",
            "sequence_dataset": "diat_raw",
            "sequence_input_name": "diat_raw_input",
            "sequence_model_suffix": "diat_raw",
            "metadata_dataset": "wfmeta",
            "metadata_type": "wfmeta10",
            "wfmeta_k": 10
        }
    ]

    for k in k_values:
        for m in models_def:
            model_id = f"cw{k}_{m['name_suffix']}"
            max_epochs = 50 if m["metadata_type"] == "metadata" else 150
            config = {
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
                "var_cnn_max_epochs": max_epochs,
                "var_cnn_base_patience": 5,
                "dir_dilations": True,
                "time_dilations": True,
                "output_dir": "/kaggle/working/outputs/",
                "model_id": model_id,
                "sequence_dataset": m["sequence_dataset"],
                "sequence_input_name": m["sequence_input_name"],
                "sequence_model_suffix": m["sequence_model_suffix"],
                "metadata_dataset": m["metadata_dataset"],
                "metadata_type": m["metadata_type"],
                "wfmeta_k": m["wfmeta_k"],
                "train_traces_per_site": k,
                "train_indices_file": f"lowdata_indices/cw{k}_train_indices.npy"
            }
            
            filename = f"cw{k}_{m['name_suffix']}.json"
            cfg_path = os.path.join(configs_dir, filename)
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Generated lowdata config: {cfg_path}")

    print("All 42 lowdata configurations generated successfully.")

if __name__ == "__main__":
    main()

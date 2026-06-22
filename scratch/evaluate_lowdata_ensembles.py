import os
import argparse
import json
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser(description="Evaluate low-data Closed-World ensembles using softmax averaging.")
    parser.add_argument("--k", type=int, required=True, choices=[40, 50, 60, 80, 100, 300, 550], help="Training traces per site (40, 50, 60, 80, 100, 300, 550).")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list of ensemble methods to evaluate (e.g. 'varcnn' or 'wfmeta_dt,wfmeta_dtl').")
    parser.add_argument("--outputs_dir", type=str, default="/kaggle/working/outputs/", help="Directory where model outputs are stored.")
    parser.add_argument("--h5_path", type=str, default="/kaggle/input/datasets/shortonkrediz/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5", help="Path to closed-world H5 dataset.")
    args = parser.parse_args()

    k = args.k
    outputs_dir = args.outputs_dir
    h5_path = args.h5_path

    # Auto-resolve H5 path
    if not os.path.exists(h5_path):
        alternative_paths = [
            "/kaggle/input/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5",
            "scratch/dummy_closed_world.h5"
        ]
        for p in alternative_paths:
            if os.path.exists(p):
                h5_path = p
                print(f"Resolved alternative H5 path: {h5_path}")
                break

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 dataset not found at: {h5_path}")

    # Load test labels
    with h5py.File(h5_path, "r") as f:
        test_labels = f["test_data/labels"][:]
    
    num_samples = len(test_labels)
    num_classes = test_labels.shape[1]
    print(f"Loaded ground truth test labels from H5. Shape: {test_labels.shape}")

    # Define ensembles
    ensembles = {
        "varcnn": {
            "name": "Var-CNN (DIR + TIME, Metadata goc)",
            "models": [f"cw{k}_dir_metadata", f"cw{k}_time_metadata"]
        },
        "wfmeta_dt": {
            "name": "WFMeta-DT (DIR + TIME, WFMeta10)",
            "models": [f"cw{k}_dir_wfmeta10", f"cw{k}_time_wfmeta10"]
        },
        "wfmeta_dtl": {
            "name": "WFMeta-DTL (DIR + TIME + LEN, WFMeta10)",
            "models": [f"cw{k}_dir_wfmeta10", f"cw{k}_time_wfmeta10", f"cw{k}_len_wfmeta10"]
        },
        "wfmeta_diat_l": {
            "name": "WFMeta-DIAT-L (DIAT + LEN, WFMeta10)",
            "models": [f"cw{k}_diat_wfmeta10", f"cw{k}_len_wfmeta10"]
        }
    }

    # Parse --only argument if provided
    selected_methods = list(ensembles.keys())
    if args.only:
        selected_methods = [m.strip().lower() for m in args.only.split(",")]
        # validate selected methods
        for m in selected_methods:
            if m not in ensembles:
                raise ValueError(f"Unknown ensemble method '{m}'. Available methods: {list(ensembles.keys())}")

    results = {}

    for method in selected_methods:
        meta = ensembles[method]
        print(f"\nEvaluating ensemble: {meta['name']}")
        
        preds_list = []
        missing_files = []

        for model_id in meta["models"]:
            # Correct path structure: outputs/<model_id>/<model_id>_model.npy
            pred_path = os.path.join(outputs_dir, model_id, f"{model_id}_model.npy")
            if not os.path.exists(pred_path):
                missing_files.append(pred_path)
            else:
                pred = np.load(pred_path)
                # Check shape constraint
                expected_shape = (10000, 100)
                if pred.shape != expected_shape:
                    raise ValueError(f"Model prediction shape mismatch for {model_id}. Expected {expected_shape}, got {pred.shape}")
                preds_list.append(pred)

        # Handle missing files
        if len(missing_files) > 0:
            msg = f"Missing prediction files for ensemble '{method}':\n" + "\n".join(f"  - {p}" for p in missing_files)
            raise FileNotFoundError(msg)

        # Softmax averaging
        P_ens = np.mean(preds_list, axis=0)
        y_pred = np.argmax(P_ens, axis=1)
        y_true = np.argmax(test_labels, axis=1)
        
        accuracy = np.mean(y_pred == y_true) * 100.0
        print(f"  Resulting Ensemble Accuracy: {accuracy:.2f}%")
        results[method] = {
            "name": meta["name"],
            "accuracy_percent": float(accuracy),
            "num_constituent_models": len(preds_list)
        }

    # Save summary output
    summary_dir = os.path.join(outputs_dir, "lowdata_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    if args.only:
        filename = f"cw{k}_ensemble_results_{args.only.replace(',', '_')}.json"
    else:
        filename = f"cw{k}_ensemble_results.json"
        
    summary_path = os.path.join(summary_dir, filename)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved ensemble evaluation results summary to: {summary_path}")

if __name__ == "__main__":
    main()

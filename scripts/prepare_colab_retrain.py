import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare paths for Google Colab retraining.")
    parser.add_argument("--h5", type=str, default="/content/drive/MyDrive/WF/h5/wfmeta_open_world_v2.h5",
                        help="Path to wfmeta_open_world_v2.h5 dataset on Colab/Drive")
    parser.add_argument("--weights_dir", type=str, default="/content/drive/MyDrive/WF/OW_weight",
                        help="Directory containing pre-trained 101-class weights on Colab/Drive")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/WF/OW_output/",
                        help="Directory to save output retrained weights and results")
    args = parser.parse_args()

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ow_retrain")
    
    print(f"Updating configs in {config_dir}...")
    print(f"  Dataset H5:  {args.h5}")
    print(f"  Weights Dir: {args.weights_dir}")
    print(f"  Output Dir:  {args.output_dir}")
    
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
        
    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            path = os.path.join(config_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                
            # Update H5 path
            cfg["processed_h5"] = args.h5
            
            # Update output_dir
            cfg["output_dir"] = args.output_dir
            
            # Update pre-trained weights path
            if "pretrained_open_world_weights" in cfg:
                orig_weights = cfg["pretrained_open_world_weights"]
                weights_filename = os.path.basename(orig_weights)
                cfg["pretrained_open_world_weights"] = os.path.join(args.weights_dir, weights_filename).replace("\\", "/")
                
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
                
            print(f"  [UPDATED] {filename}")
            
    print("\nSuccessfully updated all 16 retrain configuration files for Google Colab!")

if __name__ == "__main__":
    main()

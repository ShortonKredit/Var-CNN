import os
import sys
import shutil
import json
import numpy as np
import h5py
import subprocess

# Ensure project root in path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def create_dummy_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("packet_index;timestamp;length;direction;num_cells;ack\n")
        for r in rows:
            f.write(";".join(map(str, r)) + "\n")

def run_test():
    test_temp_dir = os.path.join(project_dir, "tests_temp")
    if os.path.exists(test_temp_dir):
        shutil.rmtree(test_temp_dir)
        
    closed_dir = os.path.join(test_temp_dir, "closed")
    open_dir = os.path.join(test_temp_dir, "open")
    output_dir = os.path.join(test_temp_dir, "output")
    
    # Create dummy traces
    # format: packet_index;timestamp;length;direction;num_cells;ack
    dummy_row_1 = [0, 1488999850.9, 600, 1, 1, 0]
    dummy_row_2 = [1, 1488999851.0, 60, -1, 0, 1]
    dummy_row_3 = [2, 1488999851.2, 600, -1, 1, 0]
    
    dummy_rows = [dummy_row_1, dummy_row_2, dummy_row_3]
    
    # Closed World dummy files
    create_dummy_csv(os.path.join(closed_dir, "training_data", "closed_world", "000_movies", "0.csv"), dummy_rows)
    create_dummy_csv(os.path.join(closed_dir, "training_data", "closed_world", "001_search", "1.csv"), dummy_rows)
    create_dummy_csv(os.path.join(closed_dir, "validation_data", "closed_world", "000_movies", "2.csv"), dummy_rows)
    create_dummy_csv(os.path.join(closed_dir, "test_data", "closed_world", "000_movies", "3.csv"), dummy_rows)
    
    # Open World dummy files
    create_dummy_csv(os.path.join(open_dir, "training_data", "open_world", "unmonitored_0.csv"), dummy_rows)
    create_dummy_csv(os.path.join(open_dir, "validation_data", "open_world", "unmonitored_1.csv"), dummy_rows)
    create_dummy_csv(os.path.join(open_dir, "test_data", "open_world", "unmonitored_2.csv"), dummy_rows)
    
    # Create dummy ANOVA ranking json containing all 74 features
    # Let's import features to get names
    from wfmeta.feature_names import FEATURE_NAMES
    
    ranking_json_path = os.path.join(test_temp_dir, "ranking.json")
    with open(ranking_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": "Mock ranking",
            "feature_order": FEATURE_NAMES
        }, f)
        
    print("Dummy test files created successfully.")
    
    # Command to run build script
    build_script = os.path.join(project_dir, "scripts", "build_wfmeta_h5.py")
    
    cmd = [
        sys.executable,
        build_script,
        "--closed-dir", closed_dir,
        "--open-dir", open_dir,
        "--ranking-json", ranking_json_path,
        "--output-dir", output_dir,
        "--seq-length", "10" # use a small sequence length for fast tests
    ]
    
    # 1. Run main test for closed_world and open_world
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
    
    assert res.returncode == 0, f"Script failed with return code {res.returncode}"
    
    # 2. Run test for open_only scenario
    cmd_open_only = [
        sys.executable,
        build_script,
        "--closed-dir", closed_dir,
        "--open-dir", open_dir,
        "--ranking-json", ranking_json_path,
        "--output-dir", output_dir,
        "--seq-length", "10",
        "--build", "open_only"
    ]
    print(f"Running command: {' '.join(cmd_open_only)}")
    res_oo = subprocess.run(cmd_open_only, capture_output=True, text=True)
    print("STDOUT:")
    print(res_oo.stdout)
    print("STDERR:")
    print(res_oo.stderr)
    
    assert res_oo.returncode == 0, f"open_only build failed with return code {res_oo.returncode}"
    
    # Verify outputs
    closed_h5_file = os.path.join(output_dir, "wfmeta_closed_world_v1.h5")
    open_h5_file = os.path.join(output_dir, "wfmeta_open_world_v1.h5")
    open_only_h5_file = os.path.join(output_dir, "wfmeta_open_only_v1.h5")
    
    assert os.path.exists(closed_h5_file), f"Missing output file: {closed_h5_file}"
    assert os.path.exists(open_h5_file), f"Missing output file: {open_h5_file}"
    assert os.path.exists(open_only_h5_file), f"Missing output file: {open_only_h5_file}"
    
    # Verify reports
    reports = [
        "h5_build_report.txt",
        "h5_schema_report.txt",
        "label_distribution_report.txt",
        "wfmeta_order_report.txt"
    ]
    for r in reports:
        report_path = os.path.join(output_dir, r)
        assert os.path.exists(report_path), f"Missing report: {report_path}"
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"\n--- {r} Content ---")
            print(content)
            
    # Verify H5 Contents
    with h5py.File(closed_h5_file, "r") as f:
        # closed has 100 classes
        assert "training_data" in f
        assert f["training_data/labels"].shape == (2, 100)
        assert f["training_data/dir_seq"].shape == (2, 10, 1)
        assert f["training_data/wfmeta"].shape == (2, 74)
        assert f["training_data/site_name"].shape == (2,)
        # Decode values to assert
        sites = [s.decode('utf-8') for s in f["training_data/site_name"][:]]
        assert "000_movies" in sites
        assert "001_search" in sites
        
        assert "validation_data" in f
        assert f["validation_data/labels"].shape == (1, 100)
        
        assert "test_data" in f
        assert f["test_data/labels"].shape == (1, 100)
        
    with h5py.File(open_h5_file, "r") as f:
        # open has 101 classes, merged closed and open
        # training_data = 2 closed + 1 open = 3 traces
        assert "training_data" in f
        assert f["training_data/labels"].shape == (3, 101)
        assert f["training_data/dir_seq"].shape == (3, 10, 1)
        
        # Verify that CW is first and OW is second
        sites = [s.decode('utf-8') for s in f["training_data/site_name"][:]]
        assert sites == ["000_movies", "001_search", "open_world"]
        
        labels_onehot = f["training_data/labels"][:]
        # Movies: class 0
        assert labels_onehot[0, 0] == 1.0
        # Search: class 1
        assert labels_onehot[1, 1] == 1.0
        # Open world: class 100
        assert labels_onehot[2, 100] == 1.0
        
        assert "validation_data" in f
        assert f["validation_data/labels"].shape == (2, 101) # 1 CW + 1 OW
        
        assert "test_data" in f
        assert f["test_data/labels"].shape == (2, 101) # 1 CW + 1 OW
        
    with h5py.File(open_only_h5_file, "r") as f:
        # open_only has 101 classes, but only contains open traces
        # training_data = 1 open trace
        assert "training_data" in f
        assert f["training_data/labels"].shape == (1, 101)
        assert f["training_data/dir_seq"].shape == (1, 10, 1)
        
        sites = [s.decode('utf-8') for s in f["training_data/site_name"][:]]
        assert sites == ["open_world"]
        
        labels_onehot = f["training_data/labels"][:]
        # Open world: class 100
        assert labels_onehot[0, 100] == 1.0
        
        assert "validation_data" in f
        assert f["validation_data/labels"].shape == (1, 101)
        
        assert "test_data" in f
        assert f["test_data/labels"].shape == (1, 101)
        
    print("\nAll unit tests passed successfully!")
    
    # Cleanup
    shutil.rmtree(test_temp_dir)

if __name__ == "__main__":
    run_test()

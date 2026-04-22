import json
import subprocess
import os

# 1. Back up original config
with open("config.json", "r") as f:
    orig_conf = json.load(f)

# 2. Write small test config
test_conf = orig_conf.copy()
test_conf['num_mon_sites'] = 2
test_conf['num_mon_inst_train'] = 10
test_conf['num_mon_inst_test'] = 2
test_conf['num_unmon_sites_train'] = 10
test_conf['num_unmon_sites_test'] = 5
test_conf['var_cnn_max_epochs'] = 2
test_conf['batch_size'] = 2
test_conf['mixture'] = [["dir", "metadata"]]  # Test only 1 to be fast

with open("config.json", "w") as f:
    json.dump(test_conf, f, indent=4)

try:
    print("=== STARTING PREPARE SCRIPT ===")
    subprocess.run(["python", "prepare_openworld.py"], check=True)
    print("=== STARTING RUN_MODEL SCRIPT ===")
    subprocess.run(["python", "run_model.py"], check=True)
    print("=== PIPELINE TEST PASSED ===")
except subprocess.CalledProcessError as e:
    print("=== PIPELINE TEST FAILED ===", e)
finally:
    # Restore original config
    with open("config.json", "w") as f:
        json.dump(orig_conf, f, indent=4)
        print("Config restored.")

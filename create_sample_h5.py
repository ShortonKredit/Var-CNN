import numpy as np
import h5py
import os
import json
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Configurations for our small sample
NUM_SITES = 5  # We will pick 5 classes (0 to 4)
INSTANCES_PER_SITE = 20
TRAIN_INSTANCES = 15
TEST_INSTANCES = 5

npz_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\OpenWorld\valid.npz'
out_h5_path = f'c:\\Users\\ADMIN\\Desktop\\Var-CNN\\data\\{NUM_SITES}_{INSTANCES_PER_SITE}_0_0.h5'
config_path = r'c:\Users\ADMIN\Desktop\Var-CNN\config_sample.json'

print("Loading data...")
data = np.load(npz_path, allow_pickle=True)
X = data['X']
y = data['y']

print("Filtering data...")
X_filtered = []
y_filtered = []

for site_id in range(NUM_SITES):
    # Find indices for this site
    idx = np.where(y == site_id)[0]
    # Take the required number of instances
    selected_idx = idx[:INSTANCES_PER_SITE]
    if len(selected_idx) < INSTANCES_PER_SITE:
        print(f"Warning: Site {site_id} only has {len(selected_idx)} instances. Expected {INSTANCES_PER_SITE}")
    
    X_filtered.extend(X[selected_idx])
    y_filtered.extend(y[selected_idx])

X_filtered = np.array(X_filtered)
y_filtered = np.array(y_filtered)

print(f"Filtered X shape: {X_filtered.shape}")

# Preprocess features
dir_seqs = []
time_seqs = []
metadatas = []

seq_length = 5000

for row in X_filtered:
    # Remove padding 0s for metadata calculation, keep length 5000 for sequences
    active = row[row != 0]
    
    # Calculate sequences
    row_dir = np.sign(row[:seq_length])
    row_time = np.abs(row[:seq_length]) # absolute time
    
    # Optional: convert to inter-packet time (VarCNN usually expects this, config handles it, but typically raw is absolute time and preprocess_data converts it. Since we bypass preprocess_data, we should do it)
    # The config inter_time=True expects it. So we will just provide absolute time, and optionally convert it here to match preprocess_data behavior.
    # We'll just mimic preprocess_data.py processing.
    
    # Meta data features
    if len(active) == 0:
        meta = np.zeros(7, dtype=np.float32)
    else:
        active_dir = np.sign(active)
        active_time = np.abs(active)
        total_packets = len(active)
        total_incoming = np.sum(active_dir == -1)
        total_outgoing = np.sum(active_dir == 1)
        total_time = active_time[-1]
        
        meta = [
            float(total_packets),
            float(total_incoming),
            float(total_outgoing),
            float(total_incoming / total_packets),
            float(total_outgoing / total_packets),
            float(total_time),
            float(total_time / total_packets) if total_packets > 0 else 0.0
        ]
        
    dir_seqs.append(row_dir)
    time_seqs.append(row_time)
    metadatas.append(meta)

dir_seqs = np.array(dir_seqs, dtype=np.int8)
time_seqs = np.array(time_seqs, dtype=np.float32)
metadatas = np.array(metadatas, dtype=np.float32)

# Inter-time conversion
inter_time = np.zeros_like(time_seqs)
inter_time[:, 1:] = time_seqs[:, 1:] - time_seqs[:, :-1]
time_seqs = inter_time  # Use inter-packet time

# Add 3rd dimension as required by CNN (Batch, Seq_len, 1)
dir_seqs = np.expand_dims(dir_seqs, axis=-1)
time_seqs = np.expand_dims(time_seqs, axis=-1)

# Scale metadata
scaler = StandardScaler()
metadatas = scaler.fit_transform(metadatas)

# One-hot encode labels
num_classes = NUM_SITES
labels_categorical = to_categorical(y_filtered, num_classes=num_classes)

print("Splitting into Train, Val, Test...")
indices = np.arange(len(y_filtered))
np.random.shuffle(indices)

# Simple random split since this is just a quick sample test
# 70% Train, 10% Val, 20% Test
train_end = int(0.7 * len(indices))
val_end = int(0.8 * len(indices))

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

train_dir, train_time, train_meta, train_y = dir_seqs[train_idx], time_seqs[train_idx], metadatas[train_idx], labels_categorical[train_idx]
val_dir, val_time, val_meta, val_y = dir_seqs[val_idx], time_seqs[val_idx], metadatas[val_idx], labels_categorical[val_idx]
test_dir, test_time, test_meta, test_y = dir_seqs[test_idx], time_seqs[test_idx], metadatas[test_idx], labels_categorical[test_idx]


print("Saving to H5 file...")
with h5py.File(out_h5_path, 'w') as f:
    f.create_group('training_data')
    f.create_group('validation_data')
    f.create_group('test_data')
    
    f.create_dataset('training_data/dir_seq', data=train_dir)
    f.create_dataset('training_data/time_seq', data=train_time)
    f.create_dataset('training_data/metadata', data=train_meta)
    f.create_dataset('training_data/labels', data=train_y)

    f.create_dataset('validation_data/dir_seq', data=val_dir)
    f.create_dataset('validation_data/time_seq', data=val_time)
    f.create_dataset('validation_data/metadata', data=val_meta)
    f.create_dataset('validation_data/labels', data=val_y)

    f.create_dataset('test_data/dir_seq', data=test_dir)
    f.create_dataset('test_data/time_seq', data=test_time)
    f.create_dataset('test_data/metadata', data=test_meta)
    f.create_dataset('test_data/labels', data=test_y)

print(f"Sample data successfully extracted to: {out_h5_path}")

# Create config_sample.json
config = {
    "data_dir": "data/",
    "predictions_dir": "predictions/",
    "_comment1": "====== World Size Params ======",
    "num_mon_sites": NUM_SITES,
    "num_mon_inst_train": TRAIN_INSTANCES,
    "num_mon_inst_test": TEST_INSTANCES,
    "num_unmon_sites_train": 0,
    "num_unmon_sites_test": 0,
    "_comment2": "====== Attack Params ======",
    "model_name": "var-cnn",
    "batch_size": 10,
    "_comment3": "====== Var-CNN Architecture Params ======",
    "mixture": [
        ["dir", "metadata"],
        ["time", "metadata"]
    ],
    "_comment4": "====== Other Params =======",
    "seq_length": 5000,
    "df_epochs": 10,
    "var_cnn_max_epochs": 10,
    "var_cnn_base_patience": 2,
    "dir_dilations": True,
    "time_dilations": True,
    "inter_time": True,
    "scale_metadata": True
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f"Created configuration file: {config_path}")

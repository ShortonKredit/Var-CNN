import h5py
import numpy as np
import os

file_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\tiny_stratified_dir_iat.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    with h5py.File(file_path, 'r') as f:
        for group_name in ['training_data', 'validation_data', 'test_data']:
            print(f"\n=== Group: {group_name} ===")
            if group_name in f:
                labels = f[group_name + '/labels'][:]
                if labels.ndim > 1:
                    labels_idx = np.argmax(labels, axis=1)
                else:
                    labels_idx = labels
                unique = np.unique(labels_idx)
                print(f"Total samples: {len(labels_idx)}")
                print(f"Unique classes: {len(unique)}")
                print(f"Classes present: {unique[:10]} ... {unique[-10:]}")
            else:
                print(f"Group {group_name} NOT FOUND!")

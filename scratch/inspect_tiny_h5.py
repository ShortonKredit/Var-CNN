import h5py
import os

file_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\tiny_dir_iat_test.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- File: {os.path.basename(file_path)} ---")
            for group_name in f.keys():
                group = f[group_name]
                print(f"Group: {group_name}")
                if isinstance(group, h5py.Group):
                    for ds_name in group.keys():
                        ds = group[ds_name]
                        if isinstance(ds, h5py.Dataset):
                            print(f"  - Dataset '{ds_name}': shape={ds.shape}, dtype={ds.dtype}")
                elif isinstance(group, h5py.Dataset):
                    print(f"  - Dataset '{group_name}': shape={group.shape}, dtype={group.dtype}")
    except Exception as e:
        print(f"Error reading file: {e}")

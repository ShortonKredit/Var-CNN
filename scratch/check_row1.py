import numpy as np
import os

file_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\OpenWorld\valid.npz'

if os.path.exists(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    
    for i in [1, 2]:
        row = X[i]
        active = row[row != 0]
        abs_vals = np.abs(active)
        diffs = np.diff(abs_vals)
        neg_diffs = diffs[diffs < 0]
        
        print(f"Row {i}:")
        print(f"  Non-zero count: {len(active)}")
        print(f"  Number of negative diffs in abs values: {len(neg_diffs)}")
        if len(neg_diffs) > 0:
            print(f"  Sample negative diffs: {neg_diffs[:5]}")
            print(f"  Sample abs values near neg diff: {abs_vals[np.where(diffs < 0)[0][0] : np.where(diffs < 0)[0][0] + 5]}")

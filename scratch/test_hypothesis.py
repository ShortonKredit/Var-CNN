import numpy as np
import os

file_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\OpenWorld\valid.npz'

if os.path.exists(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    
    row = X[0]
    # Remove zeros (padding)
    active_packets = row[row != 0]
    
    abs_vals = np.abs(active_packets)
    
    print(f"Sample row 0 - Number of non-zero elements: {len(active_packets)}")
    print(f"First 10 abs values: {abs_vals[:10]}")
    
    # Check if absolute values are non-decreasing
    is_increasing = np.all(np.diff(abs_vals) >= -1e-9) # small epsilon for float precision
    print(f"Absolute values are non-decreasing: {is_increasing}")
    
    # If not non-decreasing, check if they are inter-packet times
    if not is_increasing:
        print(f"Abs diffs between consecutive packets: {abs_vals[1:11] - abs_vals[0:10]}")

    # Check another row
    row1 = X[1]
    active_packets1 = row1[row1 != 0]
    abs_vals1 = np.abs(active_packets1)
    is_increasing1 = np.all(np.diff(abs_vals1) >= -1e-9)
    print(f"Sample row 1 - Absolute values are non-decreasing: {is_increasing1}")

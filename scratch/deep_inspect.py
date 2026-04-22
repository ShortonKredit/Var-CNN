import numpy as np
import os

file_path = r'c:\Users\ADMIN\Desktop\Var-CNN\data\OpenWorld\valid.npz'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"X shape: {X.shape}")
    
    # Check for split at 5000
    first_half = X[:, :5000]
    second_half = X[:, 5000:]
    
    print(f"First half - Min: {np.min(first_half)}, Max: {np.max(first_half)}, Mean: {np.mean(first_half)}")
    print(f"Second half - Min: {np.min(second_half)}, Max: {np.max(second_half)}, Mean: {np.mean(second_half)}")
    
    # Check if second half is ever non-zero
    non_zero_second = np.any(second_half != 0, axis=1)
    print(f"Rows with non-zero second half: {np.sum(non_zero_second)} / {len(X)}")
    
    # Check values in first half
    print("Sample row 0, first 20 elements:")
    print(X[0, :20])
    
    # Check if first half values are integer-like (like directions)
    is_int_like = np.all(np.equal(np.mod(first_half, 1), 0))
    print(f"First half is integer-like: {is_int_like}")
    
    # Check unique values in first few rows of first half
    unique_vals = np.unique(first_half[0:5])
    print(f"Unique values in first 5 rows (first half): {unique_vals}")

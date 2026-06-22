import os
import argparse
import numpy as np
import h5py
import hashlib

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Create deterministic nested subset indices for closed-world low-data evaluation.")
    parser.add_argument("--h5_path", type=str, default="/kaggle/input/datasets/shortonkrediz/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5", help="Path to closed-world H5 dataset.")
    args = parser.parse_args()

    h5_path = args.h5_path
    
    # Auto-resolve Kaggle paths if default path doesn't exist
    if not os.path.exists(h5_path):
        alternative_paths = [
            "/kaggle/input/wfmeta-closed-world-h5-v1/wfmeta_closed_world_v1.h5",
            # Local workspace dummy path for testing
            "scratch/dummy_closed_world.h5"
        ]
        for p in alternative_paths:
            if os.path.exists(p):
                h5_path = p
                print(f"Resolved alternative H5 path: {h5_path}")
                break

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Closed-world H5 dataset not found. Checked default and alternatives. Current path resolved to: {h5_path}")

    # 1. Read labels from training_data in H5
    print(f"Reading labels from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        labels = f["training_data/labels"][:]
    
    num_samples = len(labels)
    num_classes = labels.shape[1]
    print(f"Loaded {num_samples} training samples across {num_classes} classes.")
    
    # Convert one-hot to class indices
    y_train = np.argmax(labels, axis=1)
    
    # 2. Group original H5 indices by class
    class_to_indices = {c: [] for c in range(num_classes)}
    for idx, c in enumerate(y_train):
        class_to_indices[c].append(idx)
        
    # Verify that each class has exactly 855 training instances
    for c in range(num_classes):
        assert len(class_to_indices[c]) == 855, f"Expected 855 train instances for class {c}, found {len(class_to_indices[c])}"

    # 3. Deterministically shuffle indices per class using seed 42
    # Then take first 100, 300, 550 to build nested subsets
    cw40_list = []
    cw50_list = []
    cw60_list = []
    cw80_list = []
    cw100_list = []
    cw300_list = []
    cw550_list = []

    rs = np.random.RandomState(42)
    
    for c in range(num_classes):
        class_indices = np.array(class_to_indices[c])
        # Shuffle deterministically
        rs.shuffle(class_indices)
        
        # Slices
        cw40_list.extend(class_indices[:40])
        cw50_list.extend(class_indices[:50])
        cw60_list.extend(class_indices[:60])
        cw80_list.extend(class_indices[:80])
        cw100_list.extend(class_indices[:100])
        cw300_list.extend(class_indices[:300])
        cw550_list.extend(class_indices[:550])

    # Convert to arrays and sort them to keep H5 read indices strictly increasing
    cw40_arr = np.sort(np.array(cw40_list, dtype=np.int32))
    cw50_arr = np.sort(np.array(cw50_list, dtype=np.int32))
    cw60_arr = np.sort(np.array(cw60_list, dtype=np.int32))
    cw80_arr = np.sort(np.array(cw80_list, dtype=np.int32))
    cw100_arr = np.sort(np.array(cw100_list, dtype=np.int32))
    cw300_arr = np.sort(np.array(cw300_list, dtype=np.int32))
    cw550_arr = np.sort(np.array(cw550_list, dtype=np.int32))

    # 4. Assertions & Validation
    # Correct sizes
    assert len(cw40_arr) == num_classes * 40, f"Expected {num_classes * 40} indices, got {len(cw40_arr)}"
    assert len(cw50_arr) == num_classes * 50, f"Expected {num_classes * 50} indices, got {len(cw50_arr)}"
    assert len(cw60_arr) == num_classes * 60, f"Expected {num_classes * 60} indices, got {len(cw60_arr)}"
    assert len(cw80_arr) == num_classes * 80, f"Expected {num_classes * 80} indices, got {len(cw80_arr)}"
    assert len(cw100_arr) == num_classes * 100, f"Expected {num_classes * 100} indices, got {len(cw100_arr)}"
    assert len(cw300_arr) == num_classes * 300, f"Expected {num_classes * 300} indices, got {len(cw300_arr)}"
    assert len(cw550_arr) == num_classes * 550, f"Expected {num_classes * 550} indices, got {len(cw550_arr)}"

    # Check for duplicates in each array
    assert len(np.unique(cw40_arr)) == len(cw40_arr), "Duplicate found in cw40 indices!"
    assert len(np.unique(cw50_arr)) == len(cw50_arr), "Duplicate found in cw50 indices!"
    assert len(np.unique(cw60_arr)) == len(cw60_arr), "Duplicate found in cw60 indices!"
    assert len(np.unique(cw80_arr)) == len(cw80_arr), "Duplicate found in cw80 indices!"
    assert len(np.unique(cw100_arr)) == len(cw100_arr), "Duplicate found in cw100 indices!"
    assert len(np.unique(cw300_arr)) == len(cw300_arr), "Duplicate found in cw300 indices!"
    assert len(np.unique(cw550_arr)) == len(cw550_arr), "Duplicate found in cw550 indices!"

    # Nested inclusion: cw40 ⊂ cw50 ⊂ cw60 ⊂ cw80 ⊂ cw100 ⊂ cw300 ⊂ cw550
    cw40_set = set(cw40_arr)
    cw50_set = set(cw50_arr)
    cw60_set = set(cw60_arr)
    cw80_set = set(cw80_arr)
    cw100_set = set(cw100_arr)
    cw300_set = set(cw300_arr)
    cw550_set = set(cw550_arr)
    
    assert cw40_set.issubset(cw50_set), "Inclusion check failed: cw40 is not a subset of cw50!"
    assert cw50_set.issubset(cw60_set), "Inclusion check failed: cw50 is not a subset of cw60!"
    assert cw60_set.issubset(cw80_set), "Inclusion check failed: cw60 is not a subset of cw80!"
    assert cw80_set.issubset(cw100_set), "Inclusion check failed: cw80 is not a subset of cw100!"
    assert cw100_set.issubset(cw300_set), "Inclusion check failed: cw100 is not a subset of cw300!"
    assert cw300_set.issubset(cw550_set), "Inclusion check failed: cw300 is not a subset of cw550!"
    print("Nested subset checks passed: cw40 subset of cw50 subset of cw60 subset of cw80 subset of cw100 subset of cw300 subset of cw550.")

    # Bounds
    print(f"Index ranges check:")
    print(f"  cw40:  min={np.min(cw40_arr)}, max={np.max(cw40_arr)}")
    print(f"  cw50:  min={np.min(cw50_arr)}, max={np.max(cw50_arr)}")
    print(f"  cw60:  min={np.min(cw60_arr)}, max={np.max(cw60_arr)}")
    print(f"  cw80:  min={np.min(cw80_arr)}, max={np.max(cw80_arr)}")
    print(f"  cw100: min={np.min(cw100_arr)}, max={np.max(cw100_arr)}")
    print(f"  cw300: min={np.min(cw300_arr)}, max={np.max(cw300_arr)}")
    print(f"  cw550: min={np.min(cw550_arr)}, max={np.max(cw550_arr)}")

    # 5. Save indices to directory
    output_dir = "lowdata_indices"
    os.makedirs(output_dir, exist_ok=True)
    
    f40  = os.path.join(output_dir, "cw40_train_indices.npy")
    f50  = os.path.join(output_dir, "cw50_train_indices.npy")
    f60  = os.path.join(output_dir, "cw60_train_indices.npy")
    f80  = os.path.join(output_dir, "cw80_train_indices.npy")
    f100 = os.path.join(output_dir, "cw100_train_indices.npy")
    f300 = os.path.join(output_dir, "cw300_train_indices.npy")
    f550 = os.path.join(output_dir, "cw550_train_indices.npy")

    np.save(f40, cw40_arr)
    np.save(f50, cw50_arr)
    np.save(f60, cw60_arr)
    np.save(f80, cw80_arr)
    np.save(f100, cw100_arr)
    np.save(f300, cw300_arr)
    np.save(f550, cw550_arr)
    print("Indices saved successfully.")

    # 6. Print SHA256 Checksums
    print("\nSHA256 Checksums:")
    print(f"cw40_train_indices.npy sha256:  {calculate_sha256(f40)}")
    print(f"cw50_train_indices.npy sha256:  {calculate_sha256(f50)}")
    print(f"cw60_train_indices.npy sha256:  {calculate_sha256(f60)}")
    print(f"cw80_train_indices.npy sha256:  {calculate_sha256(f80)}")
    print(f"cw100_train_indices.npy sha256: {calculate_sha256(f100)}")
    print(f"cw300_train_indices.npy sha256: {calculate_sha256(f300)}")
    print(f"cw550_train_indices.npy sha256: {calculate_sha256(f550)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/build_wfmeta_h5.py

Builds the official Website Fingerprinting H5 datasets:
1. wfmeta_closed_world_v1.h5
2. wfmeta_open_world_v1.h5

Includes sequence transformations, original Var-CNN metadata extraction,
and ANOVA-ranked WFMeta extraction with streaming HDF5 write support.
"""

import os
import sys
import time
import json
import argparse
import traceback
import numpy as np
import pandas as pd
import h5py

# Ensure project root is in python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from wfmeta.trace_reader import read_trace_csv
from wfmeta.features import extract_features_from_df

def parse_args():
    parser = argparse.ArgumentParser(description="Build Website Fingerprinting H5 datasets.")
    parser.add_argument(
        "--closed-dir",
        type=str,
        required=True,
        help="Path to closed-world split directory (unzipped closed_world_split.tar.gz)"
    )
    parser.add_argument(
        "--open-dir",
        type=str,
        default=None,
        help="Path to open-world split directory (unzipped open_world_split.tar.gz)"
    )
    parser.add_argument(
        "--ranking-json",
        type=str,
        required=True,
        help="Path to wfmeta_anova_ranked_features_v1.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save H5 files and reports"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=5000,
        help="Sequence padding/truncation length (default: 5000)"
    )
    parser.add_argument(
        "--build",
        nargs="+",
        choices=["closed_world", "open_world"],
        default=["closed_world", "open_world"],
        help="Which scenario H5 files to build (default: both)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use gzip compression for H5 datasets (may increase build time but saves space)"
    )
    return parser.parse_args()

def scan_split_directory(base_dir, world_type):
    """
    Scans base_dir recursively and finds CSV files in splits.
    Returns:
        dict: mapping split_name -> list of trace dicts
    """
    splits = {
        "training_data": [],
        "validation_data": [],
        "test_data": []
    }
    
    if not base_dir:
        return splits
        
    if not os.path.exists(base_dir):
        print(f"Warning: Directory does not exist: {base_dir}")
        return splits
        
    print(f"Scanning directory: {base_dir} for {world_type} world CSVs...")
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                # Use Unix slashes for consistency
                full_path = os.path.join(root, f).replace("\\", "/")
                
                # Check split name
                detected_split = None
                for split_cand in ["training_data", "validation_data", "test_data"]:
                    if f"/{split_cand}/" in full_path or full_path.startswith(f"{split_cand}/"):
                        detected_split = split_cand
                        break
                        
                if not detected_split:
                    continue
                
                # Parse labels, site_name, trace_name
                parts = full_path.split("/")
                trace_name = parts[-1]
                
                if world_type == "closed":
                    site_folder = parts[-2]
                    try:
                        label = int(site_folder.split("_")[0])
                    except (ValueError, IndexError):
                        label = -1
                    site_name = site_folder
                else:
                    site_name = "open_world"
                    label = 100
                    
                splits[detected_split].append({
                    "path": full_path,
                    "site_name": site_name,
                    "trace_name": trace_name,
                    "label": label,
                    "world": world_type
                })
                
    # Deterministically sort by path
    for split_name in splits:
        splits[split_name].sort(key=lambda x: x["path"])
        print(f"  Split '{split_name}': {len(splits[split_name])} traces")
        
    return splits

def process_trace(trace_dict, seq_length, ranking_order):
    """
    Processes a single CSV trace and returns sequence arrays and metadata.
    """
    filepath = trace_dict["path"]
    
    # 1. Read CSV using trace_reader
    df = read_trace_csv(filepath)
    
    # 2. Sort by packet_index if present
    if "packet_index" in df.columns:
        df = df.sort_values("packet_index").reset_index(drop=True)
        
    # Extract columns as numpy arrays
    directions = df["direction"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=float)
    lengths = df["length"].to_numpy(dtype=float)
    
    # If trace is empty, raise an error to log in failed traces
    if len(df) == 0:
        raise ValueError("Empty CSV file")
        
    # 3. Relative timestamp
    timestamp_rel = timestamps - timestamps[0]
    
    # 4. Calculate IAT (time_seq) and clamp inversion to 0.0
    iat = np.zeros_like(timestamp_rel)
    if len(timestamp_rel) > 1:
        iat[1:] = timestamp_rel[1:] - timestamp_rel[:-1]
    iat = np.maximum(0.0, iat)
    
    # 5. Length padding/truncation
    n_real = min(len(df), seq_length)
    
    # Sequence arrays shape (5000, 1)
    dir_seq = np.zeros((seq_length, 1), dtype=np.float32)
    timestamp_seq = np.zeros((seq_length, 1), dtype=np.float32)
    time_seq = np.zeros((seq_length, 1), dtype=np.float32)
    len_seq = np.zeros((seq_length, 1), dtype=np.float32)
    
    dir_seq[:n_real, 0] = directions[:n_real]
    timestamp_seq[:n_real, 0] = timestamp_rel[:n_real]
    time_seq[:n_real, 0] = iat[:n_real]
    len_seq[:n_real, 0] = lengths[:n_real]
    
    # Derived streams
    diat_raw = dir_seq * time_seq
    diat_log = dir_seq * np.log1p(time_seq)
    dir_ts = dir_seq * timestamp_seq
    
    # 6. Original Var-CNN Metadata (7 features calculated on entire raw trace)
    total_packets = float(len(df))
    total_incoming = float(np.sum(directions == -1))
    total_outgoing = float(np.sum(directions == 1))
    in_ratio = total_incoming / total_packets if total_packets > 0 else 0.0
    out_ratio = total_outgoing / total_packets if total_packets > 0 else 0.0
    duration = float(timestamp_rel[-1]) if len(timestamp_rel) > 0 else 0.0
    avg_time = duration / total_packets if total_packets > 0 else 0.0
    
    metadata = np.array([
        total_packets,
        total_incoming,
        total_outgoing,
        in_ratio,
        out_ratio,
        duration,
        avg_time
    ], dtype=np.float32)
    
    # 7. Extract WFMeta (74 features) and sort according to ANOVA rank
    feature_dict = extract_features_from_df(df)
    wfmeta = np.zeros(74, dtype=np.float32)
    for idx, feat_name in enumerate(ranking_order):
        wfmeta[idx] = feature_dict[feat_name]
        
    return {
        "dir_seq": dir_seq,
        "timestamp": timestamp_seq,
        "time_seq": time_seq,
        "diat_raw": diat_raw,
        "diat_log": diat_log,
        "dir_ts": dir_ts,
        "len_seq": len_seq,
        "metadata": metadata,
        "wfmeta": wfmeta
    }

def build_hdf5_dataset(h5_filepath, split_traces, seq_length, ranking_order, num_classes, compress):
    """
    Creates and streams trace data into the specified H5 file.
    """
    compression_opts = "gzip" if compress else None
    
    # Prepare HDF5 string dtype (UTF-8)
    str_dtype = h5py.string_dtype(encoding='utf-8')
    
    failed_trace_logs = []
    
    # We will build split by split
    with h5py.File(h5_filepath, "w") as f:
        for split_name, traces in split_traces.items():
            N = len(traces)
            print(f"Building split '{split_name}' in {os.path.basename(h5_filepath)} (pre-allocated count: {N})...")
            
            if N == 0:
                print(f"  No traces in split '{split_name}', skipping group creation.")
                continue
                
            # Create H5 Group
            grp = f.create_group(split_name)
            
            # Pre-allocate datasets with resize capacity (maxshape=(None, ...)) to support skipping failures cleanly
            ds_dir_seq = grp.create_dataset("dir_seq", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_timestamp = grp.create_dataset("timestamp", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_time_seq = grp.create_dataset("time_seq", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_diat_raw = grp.create_dataset("diat_raw", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_diat_log = grp.create_dataset("diat_log", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_dir_ts = grp.create_dataset("dir_ts", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_len_seq = grp.create_dataset("len_seq", shape=(N, seq_length, 1), maxshape=(None, seq_length, 1), dtype=np.float32, chunks=True, compression=compression_opts)
            
            ds_metadata = grp.create_dataset("metadata", shape=(N, 7), maxshape=(None, 7), dtype=np.float32, chunks=True, compression=compression_opts)
            ds_wfmeta = grp.create_dataset("wfmeta", shape=(N, 74), maxshape=(None, 74), dtype=np.float32, chunks=True, compression=compression_opts)
            
            ds_labels = grp.create_dataset("labels", shape=(N, num_classes), maxshape=(None, num_classes), dtype=np.float32, chunks=True, compression=compression_opts)
            
            ds_site_name = grp.create_dataset("site_name", shape=(N,), maxshape=(None,), dtype=str_dtype, chunks=True)
            ds_trace_name = grp.create_dataset("trace_name", shape=(N,), maxshape=(None,), dtype=str_dtype, chunks=True)
            
            write_idx = 0
            start_time = time.time()
            
            for idx, trace_info in enumerate(traces):
                if (idx + 1) % 2000 == 0 or idx + 1 == N:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (N - (idx + 1)) / rate if rate > 0 else 0
                    print(f"    Progress: {idx+1}/{N} files. Rate: {rate:.1f} files/s. ETA: {eta/60:.1f}m")
                    
                try:
                    out = process_trace(trace_info, seq_length, ranking_order)
                    
                    # One-hot labels encoding (float32 array)
                    one_hot_label = np.zeros(num_classes, dtype=np.float32)
                    label_idx = trace_info["label"]
                    if 0 <= label_idx < num_classes:
                        one_hot_label[label_idx] = 1.0
                    else:
                        raise ValueError(f"Label {label_idx} is out of bounds for num_classes={num_classes}")
                    
                    # Streaming write
                    ds_dir_seq[write_idx] = out["dir_seq"]
                    ds_timestamp[write_idx] = out["timestamp"]
                    ds_time_seq[write_idx] = out["time_seq"]
                    ds_diat_raw[write_idx] = out["diat_raw"]
                    ds_diat_log[write_idx] = out["diat_log"]
                    ds_dir_ts[write_idx] = out["dir_ts"]
                    ds_len_seq[write_idx] = out["len_seq"]
                    
                    ds_metadata[write_idx] = out["metadata"]
                    ds_wfmeta[write_idx] = out["wfmeta"]
                    ds_labels[write_idx] = one_hot_label
                    
                    # Store string arrays as byte strings or standard strings (compatible with string_dtype)
                    ds_site_name[write_idx] = trace_info["site_name"]
                    ds_trace_name[write_idx] = trace_info["trace_name"]
                    
                    write_idx += 1
                    
                except Exception as e:
                    print(f"Error processing trace {trace_info['path']}: {str(e)}")
                    failed_trace_logs.append({
                        "split": split_name,
                        "world": trace_info["world"],
                        "path": trace_info["path"],
                        "error": str(e)
                    })
                    
            # If some traces failed, resize datasets to write_idx to keep them dense
            if write_idx < N:
                print(f"  Split '{split_name}': resizing datasets from {N} to {write_idx} due to failures/skips.")
                ds_dir_seq.resize((write_idx, seq_length, 1))
                ds_timestamp.resize((write_idx, seq_length, 1))
                ds_time_seq.resize((write_idx, seq_length, 1))
                ds_diat_raw.resize((write_idx, seq_length, 1))
                ds_diat_log.resize((write_idx, seq_length, 1))
                ds_dir_ts.resize((write_idx, seq_length, 1))
                ds_len_seq.resize((write_idx, seq_length, 1))
                ds_metadata.resize((write_idx, 7))
                ds_wfmeta.resize((write_idx, 74))
                ds_labels.resize((write_idx, num_classes))
                ds_site_name.resize((write_idx,))
                ds_trace_name.resize((write_idx,))
                
            print(f"  Successfully finished split '{split_name}'. Total: {write_idx} written.")
            
    return failed_trace_logs

def generate_verification_reports(output_dir, closed_h5, open_h5, elapsed_time, closed_dir, open_dir, ranking_json, seq_length, failed_traces, ranking_order):
    """
    Generates the four mandatory verification reports in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. h5_build_report.txt
    build_report_path = os.path.join(output_dir, "h5_build_report.txt")
    with open(build_report_path, "w", encoding="utf-8") as f:
        f.write("=== H5 Build Report ===\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - elapsed_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed Time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)\n\n")
        f.write(f"Input Closed-World Directory: {closed_dir}\n")
        f.write(f"Input Open-World Directory: {open_dir}\n")
        f.write(f"Ranking JSON Path: {ranking_json}\n")
        f.write(f"Target Sequence Length: {seq_length}\n\n")
        
        f.write("Output files:\n")
        if closed_h5:
            f.write(f"  - Closed-World H5: {closed_h5} (size: {os.path.getsize(closed_h5)/(1024*1024):.2f} MB)\n")
        if open_h5:
            f.write(f"  - Open-World H5: {open_h5} (size: {os.path.getsize(open_h5)/(1024*1024):.2f} MB)\n")
        f.write("\n")
        
        f.write(f"Success count (total processed): {len(failed_traces) == 0}\n")
        f.write(f"Failed count: {len(failed_traces)}\n")
        
    # Write failed traces CSV
    if failed_traces:
        failed_csv_path = os.path.join(output_dir, "failed_traces.csv")
        pd.DataFrame(failed_traces).to_csv(failed_csv_path, index=False)
        print(f"Warning: {len(failed_traces)} traces failed. Log written to {failed_csv_path}")
        
    # Helper to scan schema
    def get_h5_schema(h5_path):
        schema_lines = []
        if not h5_path or not os.path.exists(h5_path):
            return schema_lines
        with h5py.File(h5_path, "r") as h5:
            for grp_name in h5.keys():
                schema_lines.append(f"Group: {grp_name}\n")
                grp = h5[grp_name]
                for ds_name in grp.keys():
                    ds = grp[ds_name]
                    schema_lines.append(f"  Dataset: {ds_name} | Shape: {ds.shape} | Dtype: {ds.dtype}\n")
        return schema_lines
        
    # 2. h5_schema_report.txt
    schema_report_path = os.path.join(output_dir, "h5_schema_report.txt")
    with open(schema_report_path, "w", encoding="utf-8") as f:
        f.write("=== H5 Schema Report ===\n\n")
        if closed_h5:
            f.write(f"File: {os.path.basename(closed_h5)}\n")
            f.writelines(get_h5_schema(closed_h5))
            f.write("\n" + "="*40 + "\n\n")
        if open_h5:
            f.write(f"File: {os.path.basename(open_h5)}\n")
            f.writelines(get_h5_schema(open_h5))
            
    # Helper to calculate label counts
    def get_label_distribution(h5_path):
        lines = []
        if not h5_path or not os.path.exists(h5_path):
            return lines
        with h5py.File(h5_path, "r") as h5:
            for grp_name in h5.keys():
                labels_ds = h5[grp_name]["labels"][:]
                # Convert back from one-hot encoding
                labels = np.argmax(labels_ds, axis=1)
                unique, counts = np.unique(labels, return_counts=True)
                lines.append(f"Group: {grp_name} (Total samples: {len(labels)})\n")
                for u, c in zip(unique, counts):
                    lines.append(f"  Label {u}: {c} samples\n")
        return lines
        
    # 3. label_distribution_report.txt
    label_report_path = os.path.join(output_dir, "label_distribution_report.txt")
    with open(label_report_path, "w", encoding="utf-8") as f:
        f.write("=== Label Distribution Report ===\n\n")
        if closed_h5:
            f.write(f"File: {os.path.basename(closed_h5)}\n")
            f.writelines(get_label_distribution(closed_h5))
            f.write("\n" + "="*40 + "\n\n")
        if open_h5:
            f.write(f"File: {os.path.basename(open_h5)}\n")
            f.writelines(get_label_distribution(open_h5))
            
    # 4. wfmeta_order_report.txt
    wfmeta_report_path = os.path.join(output_dir, "wfmeta_order_report.txt")
    with open(wfmeta_report_path, "w", encoding="utf-8") as f:
        f.write("=== WFMeta Order Report ===\n")
        f.write(f"Total ranked features: {len(ranking_order)}\n\n")
        f.write("Top 20 Features in Rank Order:\n")
        for i in range(min(20, len(ranking_order))):
            f.write(f"  Rank #{i+1}: {ranking_order[i]}\n")
            
        f.write("\nSpecific Verifications:\n")
        f.write(f"  Rank #1 Feature: {ranking_order[0]}\n")
        f.write(f"  Rank #2 Feature: {ranking_order[1]}\n")
        f.write(f"  Rank #10 Feature: {ranking_order[9]}\n")
        f.write(f"  Rank #74 Feature: {ranking_order[73]}\n")

    print(f"Reports successfully written to: {output_dir}")

def main():
    args = parse_args()
    
    start_time = time.time()
    
    # 1. Load ANOVA feature ranking order
    if not os.path.exists(args.ranking_json):
        print(f"Error: ANOVA ranking json not found at: {args.ranking_json}")
        sys.exit(1)
        
    with open(args.ranking_json, "r", encoding="utf-8") as f:
        ranking_data = json.load(f)
    ranking_order = ranking_data["feature_order"]
    
    if len(ranking_order) != 74:
        print(f"Error: Expected 74 ranked features, but found {len(ranking_order)} in JSON.")
        sys.exit(1)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Scan directories
    print("Scanning split directories...")
    cw_splits = scan_split_directory(args.closed_dir, "closed")
    ow_splits = scan_split_directory(args.open_dir, "open") if args.open_dir else None
    
    # Verify expected counts
    expected_cw = {
        "training_data": 85500,
        "validation_data": 4500,
        "test_data": 10000
    }
    expected_ow = {
        "training_data": 47500,
        "validation_data": 2500,
        "test_data": 50000
    }
    
    cw_total = sum(len(cw_splits[s]) for s in cw_splits)
    if cw_total == 0:
        print("\n" + "!" * 80)
        print("FATAL ERROR: NO CLOSED-WORLD TRACE CSV FILES FOUND IN THE SPECIFIED DIRECTORIES!")
        print("!" * 80 + "\n")
        sys.exit(1)
        
    cw_mismatch = False
    for split_name, expected_count in expected_cw.items():
        actual_count = len(cw_splits[split_name])
        if actual_count != expected_count:
            cw_mismatch = True
            print(f"WARNING: Closed-World split '{split_name}' count mismatch! Expected: {expected_count}, Actual: {actual_count}")
            
    if cw_mismatch:
        print("\n" + "!" * 80)
        print("WARNING: CLOSED-WORLD DATASET COUNTS DO NOT MATCH THE EXPECTED SPECIFICATION!")
        print("Expected counts: Train=85500, Val=4500, Test=10000")
        print("!" * 80 + "\n")
        
    if "open_world" in args.build:
        if not ow_splits:
            print("\n" + "!" * 80)
            print("FATAL ERROR: OPEN-WORLD DIRECTORY MUST BE SPECIFIED FOR open_world BUILD!")
            print("!" * 80 + "\n")
            sys.exit(1)
            
        ow_total = sum(len(ow_splits[s]) for s in ow_splits)
        if ow_total == 0:
            print("\n" + "!" * 80)
            print("FATAL ERROR: NO OPEN-WORLD TRACE CSV FILES FOUND IN THE SPECIFIED DIRECTORIES!")
            print("!" * 80 + "\n")
            sys.exit(1)
            
        ow_mismatch = False
        for split_name, expected_count in expected_ow.items():
            actual_count = len(ow_splits[split_name])
            if actual_count != expected_count:
                ow_mismatch = True
                print(f"WARNING: Open-World split '{split_name}' count mismatch! Expected: {expected_count}, Actual: {actual_count}")
                
        if ow_mismatch:
            print("\n" + "!" * 80)
            print("WARNING: OPEN-WORLD DATASET COUNTS DO NOT MATCH THE EXPECTED SPECIFICATION!")
            print("Expected counts: Train=47500, Val=2500, Test=50000")
            print("!" * 80 + "\n")
            
    failed_traces = []
    closed_h5_path = None
    open_h5_path = None
    
    # 3. Build Closed-World scenarios
    if "closed_world" in args.build:
        closed_h5_path = os.path.join(args.output_dir, "wfmeta_closed_world_v1.h5")
        print("======================================================================")
        print(f"BUILDING CLOSED-WORLD scenario H5 file: {closed_h5_path}")
        print("======================================================================")
        
        # Closed-World scenario dataset contains only closed splits
        # Classes: 100
        cw_failed = build_hdf5_dataset(
            h5_filepath=closed_h5_path,
            split_traces=cw_splits,
            seq_length=args.seq_length,
            ranking_order=ranking_order,
            num_classes=100,
            compress=args.compress
        )
        failed_traces.extend(cw_failed)
        
    # 4. Build Open-World scenarios
    if "open_world" in args.build:
        if not args.open_dir:
            print("Error: --open-dir is required to build the open_world scenario.")
            sys.exit(1)
            
        open_h5_path = os.path.join(args.output_dir, "wfmeta_open_world_v1.h5")
        print("======================================================================")
        print(f"BUILDING OPEN-WORLD scenario H5 file: {open_h5_path}")
        print("======================================================================")
        
        # Merge splits for open-world
        merged_splits = {}
        for split_name in ["training_data", "validation_data", "test_data"]:
            cw_part = cw_splits[split_name]
            ow_part = ow_splits[split_name] if ow_splits else []
            # Order: closed first, open second
            merged_splits[split_name] = cw_part + ow_part
            
        # Classes: 101
        ow_failed = build_hdf5_dataset(
            h5_filepath=open_h5_path,
            split_traces=merged_splits,
            seq_length=args.seq_length,
            ranking_order=ranking_order,
            num_classes=101,
            compress=args.compress
        )
        failed_traces.extend(ow_failed)
        
    # 5. Done, generate reports
    elapsed_time = time.time() - start_time
    print("======================================================================")
    print(f"Completed H5 Build in {elapsed_time/60:.2f}m ({elapsed_time:.1f}s)")
    print("======================================================================")
    
    generate_verification_reports(
        output_dir=args.output_dir,
        closed_h5=closed_h5_path,
        open_h5=open_h5_path,
        elapsed_time=elapsed_time,
        closed_dir=args.closed_dir,
        open_dir=args.open_dir,
        ranking_json=args.ranking_json,
        seq_length=args.seq_length,
        failed_traces=failed_traces,
        ranking_order=ranking_order
    )

if __name__ == "__main__":
    main()

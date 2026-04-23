import h5py
import numpy as np
import json
import argparse
import os
import gc
from tensorflow.keras.utils import to_categorical

def process():
    parser = argparse.ArgumentParser(description="Convert Data into Var-CNN format using Block Loading")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # --- LẤY THAM SỐ TỪ CONFIG ---
    N_MON = config['num_mon_sites']
    TR_MON = config['num_mon_inst_train']
    TE_MON = config['num_mon_inst_test']
    TR_UNM = config['num_unmon_sites_train']
    TE_UNM = config['num_unmon_sites_test']
    SEQ_LEN = config.get('seq_length', 5000)

    INPUT_CW = config.get('raw_cw_h5', 'data/mini_cw_test.h5')
    INPUT_OW = config.get('raw_ow_h5', 'data/mini_ow_test.h5')
    
    num_mon_inst = TR_MON + TE_MON
    OUTPUT_H5 = config.get('processed_h5')
    if not OUTPUT_H5:
        OUTPUT_H5 = os.path.join(config.get('data_dir', 'data/'), f"{N_MON}_{num_mon_inst}_{TR_UNM}_{TE_UNM}.h5")

    print(f"[*] Configuration: {args.config}")
    print(f"[*] Target H5: {OUTPUT_H5}")

    # CHUẨN BỊ KÍCH THƯỚC (Logic 5% Validation)
    mon_train_pool_size = N_MON * TR_MON
    mon_val_size = int(mon_train_pool_size * 0.05)
    mon_train_final_size = mon_train_pool_size - mon_val_size
    mon_test_size = N_MON * TE_MON

    unm_val_size = int(TR_UNM * 0.05)
    unm_train_final_size = TR_UNM - unm_val_size

    total_tr = mon_train_final_size + unm_train_final_size
    total_va = mon_val_size + unm_val_size
    total_te = mon_test_size + TE_UNM

    def create_empty_sets(size):
        return (np.zeros((size, SEQ_LEN, 1), dtype=np.int8), 
                np.zeros((size, SEQ_LEN, 1), dtype=np.float32),
                np.zeros((size, 7), dtype=np.float32),
                np.zeros((size, N_MON + 1), dtype=np.float32))

    tr_d, tr_t, tr_m, tr_l = create_empty_sets(total_tr)
    va_d, va_t, va_m, va_l = create_empty_sets(total_va)
    te_d, te_t, te_m, te_l = create_empty_sets(total_te)

    # BƯỚC 1: XỬ LÝ CLOSED WORLD (SITE-BY-SITE BLOCK LOADING)
    print(f"[*] Processing Monitored Sites from {INPUT_CW}...")
    with h5py.File(INPUT_CW, 'r') as f:
        raw_labels = f['labels'][:]
        unique_labels = np.unique(raw_labels)
        
        # Pool tạm cho Train Monitored trước khi chia 5%
        mon_pool_d = np.zeros((mon_train_pool_size, SEQ_LEN, 1), dtype=np.int8)
        mon_pool_t = np.zeros((mon_train_pool_size, SEQ_LEN, 1), dtype=np.float32)
        mon_pool_m = np.zeros((mon_train_pool_size, 7), dtype=np.float32)
        mon_pool_l = np.zeros((mon_train_pool_size,), dtype=np.int32)

        pool_idx = 0
        te_idx = 0

        for site_id in range(N_MON):
            true_label = unique_labels[site_id]
            # np.where luon tra ve chi so tang dan -> h5py se thoai mai load
            site_indices = np.where(raw_labels == true_label)[0]
            
            # --- BLOCK LOADING: Load nguyen ca site vao RAM ---
            site_dir = np.expand_dims(f['direction'][site_indices], -1)
            site_time = np.expand_dims(f['timing'][site_indices], -1)
            site_meta = f['metadata'][site_indices]
            
            # Pad neu thieu (cho mini)
            curr_size = len(site_indices)
            if curr_size < TR_MON + TE_MON:
                pad_w = (TR_MON + TE_MON) - curr_size
                site_dir = np.pad(site_dir, ((0, pad_w), (0,0), (0,0)), 'wrap')
                site_time = np.pad(site_time, ((0, pad_w), (0,0), (0,0)), 'wrap')
                site_meta = np.pad(site_meta, ((0, pad_w), (0,0)), 'wrap')

            # Tron ngẫu nhiên TRÊN RAM (Khong con phu thuoc I/O cua h5py)
            shuffle_idx = np.random.permutation(len(site_dir))
            train_idx = shuffle_idx[:TR_MON]
            test_idx = shuffle_idx[TR_MON:TR_MON+TE_MON]

            # Ghi vao Test
            te_d[te_idx:te_idx+TE_MON] = site_dir[test_idx]
            te_t[te_idx:te_idx+TE_MON] = site_time[test_idx]
            te_m[te_idx:te_idx+TE_MON] = site_meta[test_idx]
            te_l[te_idx:te_idx+TE_MON] = to_categorical([site_id]*TE_MON, N_MON+1)
            te_idx += TE_MON

            # Ghi vao Train Pool
            mon_pool_d[pool_idx:pool_idx+TR_MON] = site_dir[train_idx]
            mon_pool_t[pool_idx:pool_idx+TR_MON] = site_time[train_idx]
            mon_pool_m[pool_idx:pool_idx+TR_MON] = site_meta[train_idx]
            mon_pool_l[pool_idx:pool_idx+TR_MON] = site_id
            pool_idx += TR_MON

    del raw_labels
    gc.collect()

    print("[*] Splitting 5% Validation for Monitored...")
    shuffled_pool = np.random.permutation(mon_train_pool_size)
    v_idx = shuffled_pool[:mon_val_size]
    t_idx = shuffled_pool[mon_val_size:]

    va_d[:mon_val_size] = mon_pool_d[v_idx]
    va_t[:mon_val_size] = mon_pool_t[v_idx]
    va_m[:mon_val_size] = mon_pool_m[v_idx]
    va_l[:mon_val_size] = to_categorical(mon_pool_l[v_idx], N_MON+1)

    tr_d[:mon_train_final_size] = mon_pool_d[t_idx]
    tr_t[:mon_train_final_size] = mon_pool_t[t_idx]
    tr_m[:mon_train_final_size] = mon_pool_m[t_idx]
    tr_l[:mon_train_final_size] = to_categorical(mon_pool_l[t_idx], N_MON+1)

    del mon_pool_d, mon_pool_t, mon_pool_m, mon_pool_l
    gc.collect()

    # BƯỚC 2: XỬ LÝ OPEN WORLD (BIG BLOCK LOADING)
    print(f"[*] Processing Unmonitored Data from {INPUT_OW}...")
    with h5py.File(INPUT_OW, 'r') as f:
        # Load logic: 150k dau -> Train Pool, 100k sau -> Test
        # Chung ta load tung phan de tranh OOM tren may yeu
        print("    -> Loading Training Unmonitored Block...")
        unm_tr_dir = np.expand_dims(f['direction'][:TR_UNM], -1)
        unm_tr_time = np.expand_dims(f['timing'][:TR_UNM], -1)
        unm_tr_meta = f['metadata'][:TR_UNM]

        unm_idx = np.random.permutation(TR_UNM)
        uv_idx = unm_idx[:unm_val_size]
        ut_idx = unm_idx[unm_val_size:]

        va_d[mon_val_size:] = unm_tr_dir[uv_idx]
        va_t[mon_val_size:] = unm_tr_time[uv_idx]
        va_m[mon_val_size:] = unm_tr_meta[uv_idx]
        va_l[mon_val_size:] = to_categorical([N_MON]*unm_val_size, N_MON+1)

        tr_d[mon_train_final_size:] = unm_tr_dir[ut_idx]
        tr_t[mon_train_final_size:] = unm_tr_time[ut_idx]
        tr_m[mon_train_final_size:] = unm_tr_meta[ut_idx]
        tr_l[mon_train_final_size:] = to_categorical([N_MON]*unm_train_final_size, N_MON+1)

        del unm_tr_dir, unm_tr_time, unm_tr_meta
        gc.collect()

        print("    -> Loading Test Unmonitored Block...")
        te_d[mon_test_size:] = np.expand_dims(f['direction'][TR_UNM:TR_UNM+TE_UNM], -1)
        te_t[mon_test_size:] = np.expand_dims(f['timing'][TR_UNM:TR_UNM+TE_UNM], -1)
        te_m[mon_test_size:] = f['metadata'][TR_UNM:TR_UNM+TE_UNM]
        te_l[mon_test_size:] = to_categorical([N_MON]*TE_UNM, N_MON+1)

    gc.collect()

    print(f"[*] Writing to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as f:
        for group, d, t, m, l in [('training_data', tr_d, tr_t, tr_m, tr_l),
                                  ('validation_data', va_d, va_t, va_m, va_l),
                                  ('test_data', te_d, te_t, te_m, te_l)]:
            g = f.create_group(group)
            g.create_dataset('dir_seq', data=d, compression="gzip")
            g.create_dataset('time_seq', data=t, compression="gzip")
            g.create_dataset('metadata', data=m, compression="gzip")
            g.create_dataset('labels', data=l, compression="gzip")

    print("[*] HOÀN TẤT ALL TASKS! DỮ LIỆU ĐÃ SẴN SÀNG.")

if __name__ == '__main__':
    process()

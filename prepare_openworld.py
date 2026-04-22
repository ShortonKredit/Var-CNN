import numpy as np
import h5py
import os
import argparse
import json
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

def build_features(traces, labels, num_classes):
    seq_len = 5000
    dirs, times, metas = [], [], []
    for row in traces:
        active = row[row != 0]
        dirs.append(np.sign(row[:seq_len]))
        times.append(np.abs(row[:seq_len]))
        if len(active) == 0:
            metas.append(np.zeros(7, dtype=np.float32))
        else:
            a_dir = np.sign(active)
            a_time = np.abs(active)
            metas.append([len(active), np.sum(a_dir == -1), np.sum(a_dir == 1),
                          np.sum(a_dir == -1) / len(active), np.sum(a_dir == 1) / len(active),
                          a_time[-1], a_time[-1] / len(active)])
                          
    dirs = np.expand_dims(np.array(dirs, dtype=np.int8), -1)
    times = np.array(times, dtype=np.float32)
    inter = np.zeros_like(times)
    # Lấy đạo hàm thời gian (Inter-arrival time)
    inter[:, 1:] = times[:, 1:] - times[:, :-1]
    times = np.expand_dims(inter, -1)
    return dirs, times, np.array(metas, dtype=np.float32), to_categorical(labels, num_classes=num_classes)

def write_h5(path, datasets, num_classes):
    with h5py.File(path, 'w') as f:
        for g_name, d_tuple in datasets.items():
            g = f.create_group(g_name)
            g.create_dataset('dir_seq', data=d_tuple[0])
            g.create_dataset('time_seq', data=d_tuple[1])
            g.create_dataset('metadata', data=d_tuple[2])
            g.create_dataset('labels', data=d_tuple[3])

def process_data(config_path='config.json', out_dir='data'):
    # Để giữ nguyên độ trật tự dữ liệu tuyệt đối khi train nhiều lần (Tránh Data Leakage)
    np.random.seed(42)
    
    # Lấy CHUẨN THÔNG SỐ từ file config. Được quản lý hoàn toàn ở config.json
    with open(config_path, 'r') as f:
        conf = json.load(f)
        
    num_mon_sites = conf['num_mon_sites']
    num_mon_inst_train = conf['num_mon_inst_train']
    num_mon_inst_test = conf['num_mon_inst_test']
    num_unmon_sites_train = conf['num_unmon_sites_train']
    num_unmon_sites_test = conf['num_unmon_sites_test']
    
    total_mon_inst = num_mon_inst_train + num_mon_inst_test
    total_unmon_inst = num_unmon_sites_train + num_unmon_sites_test
    
    # Cắt chuẩn 80% từ tổng kích thước làm Data Train thực tế (lõi), phần dư cho Valid.
    mon_train_cut = int(total_mon_inst * 0.8)
    unmon_train_cut = int(total_unmon_inst * 0.8)
    
    # Chặn không chạy kịch bản nếu file H5 đã tốn công tạo sẵn từ đợt chạy trước
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    file_open_base = os.path.join(out_dir, f"{num_mon_sites}_{total_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.h5")
    
    if os.path.exists(file_open_base):
        print(f"[*] Quá tuyệt, file Hệ Sinh Thái Data ({file_open_base}) đã tồn tại từ trước.")
        print("[*] Sẽ bỏ qua quá trình xáo trộn Data để BẢO TOÀN tính vĩnh cửu của bộ đề thi cũ!")
        return True
        
    train_path = os.path.join(out_dir, 'OpenWorld/train.npz')
    valid_path = os.path.join(out_dir, 'OpenWorld/valid.npz')
    
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        print(f"Error: Could not find train.npz and valid.npz in {out_dir}/OpenWorld/")
        print("Please extract OpenWorld.tar.gz first.")
        return False
        
    print(f"Loading Base Dataset (Day 0) from {train_path} and {valid_path}...")
    d_train = np.load(train_path, allow_pickle=True)
    d_valid = np.load(valid_path, allow_pickle=True)
    
    # Hợp nhất tập Train và Valid của tác giả như đề mục 5.1.2 trong báo cáo Var-CNN
    X = np.concatenate([d_train['X'], d_valid['X']])
    y = np.concatenate([d_train['y'], d_valid['y']])
    del d_train, d_valid # Xoá bộ đệm
    
    unique_labels, label_counts = np.unique(y, return_counts=True)
    unmon_idx = np.argmax(label_counts)
    unmon_label = unique_labels[unmon_idx]
    
    # Nhặt ra chính xác các Website được theo dõi (Monitored)
    mon_labels = [l for l in unique_labels if l != unmon_label][:num_mon_sites]
    
    tr_X, va_X, te_X, tr_y, va_y, te_y = [], [], [], [], [], []
    
    # --- 1. Xu ly nhom Web Muc tieu (Monitored Sites) ---
    print(f"[*] Parsing Monitored data ({num_mon_sites} sites) into 80:10:10 ...")
    label_map = {old_l: new_l for new_l, old_l in enumerate(mon_labels)}
    
    for l in mon_labels:
        idx = np.where(y == l)[0]
        np.random.shuffle(idx)
        sel = idx[:total_mon_inst]
        
        tr_X.extend(X[sel[:mon_train_cut]])
        tr_y.extend([label_map[l]] * mon_train_cut)
        va_X.extend(X[sel[mon_train_cut:num_mon_inst_train]])
        va_y.extend([label_map[l]] * (num_mon_inst_train - mon_train_cut))
        te_X.extend(X[sel[num_mon_inst_train:total_mon_inst]])
        te_y.extend([label_map[l]] * num_mon_inst_test)

    # --- 2. Xu ly nhom Rac / Mang dien rong (Unmonitored) ---
    print(f"[*] Parsing Unmonitored data (Label ID: {num_mon_sites}) ...")
    unm_idx = np.where(y == unmon_label)[0]
    np.random.shuffle(unm_idx)
    sel = unm_idx[:total_unmon_inst]
    
    tr_X.extend(X[sel[:unmon_train_cut]])
    tr_y.extend([num_mon_sites] * unmon_train_cut)  # Cho nhãn ID của Rác bằng ID cuối cùng
    va_X.extend(X[sel[unmon_train_cut:num_unmon_sites_train]])
    va_y.extend([num_mon_sites] * (num_unmon_sites_train - unmon_train_cut))
    te_X.extend(X[sel[num_unmon_sites_train:total_unmon_inst]])
    te_y.extend([num_mon_sites] * num_unmon_sites_test)
    
    # Xoá RAM Mảng thô đã dùng xong
    del X, y 
    
    # --- 3. Trich xuat Features (Paper Specs) ---
    print(f"[*] Building Features (Paper Spec). Count => Train: {len(tr_X)} | Valid: {len(va_X)} | Test: {len(te_X)}")
    tr_d, tr_t, tr_m, tr_lbl = build_features(np.array(tr_X), np.array(tr_y), num_mon_sites + 1)
    va_d, va_t, va_m, va_lbl = build_features(np.array(va_X), np.array(va_y), num_mon_sites + 1)
    te_d, te_t, te_m, te_lbl = build_features(np.array(te_X), np.array(te_y), num_mon_sites + 1)
    
    scaler = StandardScaler()
    tr_m = scaler.fit_transform(tr_m)
    va_m = scaler.transform(va_m)
    te_m = scaler.transform(te_m)
    
    # --- 4. Lưu HDF5 file phục vụ Huấn Luyện ---
    write_h5(file_open_base, {
        'training_data': (tr_d, tr_t, tr_m, tr_lbl),
        'validation_data': (va_d, va_t, va_m, va_lbl),
        'test_data': (te_d, te_t, te_m, te_lbl)
    }, num_mon_sites + 1)
    
    print(f"[*] H5 extracted successfully at: {file_open_base}")
    print("\n--- Finished Restoring Var-CNN OpenWorld Dataset (Day 0) ---")
    print("Model is ready to train!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    process_data(args.config)

import numpy as np
import os

def analyze_npz(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} không tồn tại.")
        return

    print(f"\n{'='*50}")
    print(f" PHÂN TÍCH FILE: {os.path.basename(file_path)}")
    print(f"{'='*50}")

    # Load dữ liệu
    data = np.load(file_path, allow_pickle=True)
    
    # Kiểm tra các Keys
    print(f"[*] Các thành phần trong file: {list(data.keys())}")
    
    X = data['X']
    y = data['y']
    
    print(f"[*] Tổng số mẫu: {len(y)}")
    print(f"[*] Kích thước đặc trưng (Features): {X.shape[1]}")

    # Phân tích nhãn
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    # Xác định lớp Unmonitored (thường là lớp có nhiều mẫu nhất hoặc nhãn 0/ -1)
    # Trong bộ OpenWorld của Bhat, lớp Unmonitored thường là lớp có số lượng mẫu vượt trội
    unmon_idx = np.argmax(counts)
    unmon_label = unique[unmon_idx]
    unmon_count = counts[unmon_idx]
    
    mon_counts = [count for label, count in label_counts.items() if label != unmon_label]
    num_mon_sites = len(unique) - 1

    print(f"\n--- THỐNG KÊ CHI TIẾT ---")
    print(f"[-] Số lượng trang web Monitored: {num_mon_sites}")
    print(f"[-] Tổng mẫu Monitored: {sum(mon_counts)}")
    print(f"[-] Tổng mẫu Unmonitored (Nhãn {unmon_label}): {unmon_count}")
    
    if num_mon_sites > 0:
        print(f"[-] Trung bình mẫu mỗi trang Monitored: {np.mean(mon_counts):.2f}")
        print(f"[-] Ít nhất: {np.min(mon_counts)} mẫu/trang")
        print(f"[-] Nhiều nhất: {np.max(mon_counts)} mẫu/trang")

    # Hiển thị Top 10 trang web Monitored có nhiều mẫu nhất
    sorted_labels = sorted([item for item in label_counts.items() if item[0] != unmon_label], 
                          key=lambda x: x[1], reverse=True)
    
    print(f"\n--- TOP 10 WEBSITE MONITORED (Nhiều mẫu nhất) ---")
    for label, count in sorted_labels[:10]:
        print(f"Web ID {label:3}: {count:4} mẫu")

    print(f"\n--- 10 WEBSITE MONITORED (Ít mẫu nhất) ---")
    for label, count in sorted_labels[-10:]:
        print(f"Web ID {label:3}: {count:4} mẫu")

if __name__ == "__main__":
    # Đường dẫn tới các file trong máy bạn
    train_file = "data/OpenWorld/train.npz"
    valid_file = "data/OpenWorld/valid.npz"
    
    if os.path.exists(train_file):
        analyze_npz(train_file)
    if os.path.exists(valid_file):
        analyze_npz(valid_file)
    
    print(f"\n{'='*50}")
    print(" LỜI KHUYÊN: Nếu mỗi trang Monitored có < 200 mẫu,")
    print(" kết quả TPR sẽ cực kỳ thấp. Bạn cần gộp thêm Day1, Day2...")
    print(f"{'='*50}")

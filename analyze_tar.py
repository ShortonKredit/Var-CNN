import tarfile
import os

def analyze_tar_contents(tar_path):
    if not os.path.exists(tar_path):
        print(f"Error: Không tìm thấy file {tar_path}")
        return

    print(f"\n{'='*60}")
    print(f" ĐANG SIÊU ÂM FILE NÉN: {os.path.basename(tar_path)}")
    print(f"{'='*60}")

    total_size = 0
    file_count = 0
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            file_count = len(members)
            
            print(f"{'Tên File':<50} | {'Dung lượng (MB)':>15}")
            print("-" * 70)
            
            for member in members:
                size_mb = member.size / (1024 * 1024)
                total_size += member.size
                
                # Chỉ in ra các file có dung lượng đáng kể hoặc các file .npz / .txt
                if size_mb > 1 or member.name.endswith(('.npz', '.txt', '.json')):
                    print(f"{member.name[:48]:<50} | {size_mb:>15.2f} MB")
                elif member.isdir():
                    print(f"{member.name[:48]:<50} | {'<DIR>':>15}")

            print("-" * 70)
            print(f"[*] Tổng số lượng file/thư mục: {file_count}")
            print(f"[*] Tổng dung lượng sau khi giải nén: {total_size / (1024**3):.2f} GB")
            
            # Kiểm tra xem có dấu hiệu của các Day khác không
            npz_files = [m.name for m in members if m.name.endswith('.npz')]
            print(f"[*] Số lượng file dữ liệu (.npz): {len(npz_files)}")
            
            if len(npz_files) > 2:
                print("\n[!] PHÁT HIỆN: Bộ dữ liệu này có vẻ chứa nhiều phần dữ liệu khác nhau!")
            elif "train.npz" in str(npz_files):
                print("\n[!] CHÚ Ý: Đây có vẻ chỉ là bộ Day 0 (Base) tiêu chuẩn.")
                
    except Exception as e:
        print(f"Lỗi khi đọc file tar: {e}")

if __name__ == "__main__":
    # Đường dẫn file nén trên máy bạn
    target_tar = "data/OpenWorld.tar.gz"
    analyze_tar_contents(target_tar)

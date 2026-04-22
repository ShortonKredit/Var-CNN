import tarfile
import numpy as np
import os
import shutil
import tempfile

def deep_analyze_tar(tar_path):
    if not os.path.exists(tar_path):
        print(f"Error: Không tìm thấy file {tar_path}")
        return

    print(f"\n{'='*70}")
    print(f" PHAN TICH CHUYEN SAU DU LIEU OPENWORLD (12GB)")
    print(f"{'='*70}")

    results = []
    
    # Tạo thư mục tạm an toàn
    temp_dir = tempfile.mkdtemp(dir="data")
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.npz')]
            
            print(f"Tim thay {len(members)} tep .npz. Bat dau phan tich tung tep...\n")
            print(f"{'Ten File':<20} | {'Mau (Samples)':>15} | {'So Lop (Classes)':>15} | {'Ghi chu'}")
            print("-" * 75)
            
            total_samples = 0
            
            for member in members:
                # Giai nen tep hien tai
                tar.extract(member, path=temp_dir)
                extracted_path = os.path.join(temp_dir, member.name)
                
                # Phan tich noi dung npz
                try:
                    with np.load(extracted_path, allow_pickle=True) as data:
                        y = data['y']
                        n_samples = len(y)
                        unique_labels = np.unique(y)
                        n_classes = len(unique_labels)
                        total_samples += n_samples
                    
                        note = ""
                        if "train" in member.name: note = "Tap Huan luyen chinh"
                        elif "valid" in member.name: note = "Tap Kiem dinh chinh"
                        else: note = f"Du lieu bo sung"
                        
                        print(f"{os.path.basename(member.name):<20} | {n_samples:>15,} | {n_classes:>15} | {note}")
                        
                        results.append({
                            'file': member.name,
                            'samples': n_samples,
                            'classes': n_classes
                        })
                except Exception as e:
                    print(f"\n[!] Loi khi doc {member.name}: {e}")
                finally:
                    # XOA NGAY de tiet kiem pin/o cung
                    if os.path.exists(extracted_path):
                        try:
                            os.remove(extracted_path)
                        except Exception:
                            pass
            
            print("-" * 75)
            print(f"[*] TONG CONG SO MAU TOAN BO: {total_samples:,}")
            print(f"[*] Trung binh mau moi tep: {total_samples // len(members):,}")

    finally:
        # Don dep thu muc tam
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    deep_analyze_tar("data/OpenWorld.tar.gz")

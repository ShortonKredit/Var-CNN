# Tổng Quan Dữ Liệu `OpenWorld.tar.gz`

Tài liệu này cung cấp cái nhìn chi tiết về cấu trúc và kích thước phân bổ của kho lưu trữ tập dữ liệu OpenWorld sử dụng trong dự án Var-CNN rà quét Website Fingerprinting.

## 📦 Cấu Trúc Tập Dữ Liệu Tự Nhiên (Concept Drift)

Kho lưu trữ `OpenWorld.tar.gz` có kích thước nén khoảng **944MB**. Bên trong thư mục chứa 7 tệp `.npz`, ghi nhận lại tiến trình thay đổi (decay) của các websites qua thời gian:

| Tên File | Kích Thước (Thô) | Phân Loại | Chức Năng Đề Xuất |
| :--- | :--- | :--- | :--- |
| `train.npz` | **1.89 GB** | Tập Gốc | Huấn luyện Base Model |
| `valid.npz` | **210 MB** | Tập Kiểm tra nội bộ | Đánh giá chéo trong quá trình Train |
| `day14.npz` | **2.05 GB** | Tập Trôi Dạt | Đánh giá độ lệch Concept Drift sau 14 ngày |
| `day30.npz` | **2.07 GB** | Tập Trôi Dạt | Đánh giá sau 30 ngày |
| `day90.npz` | **2.53 GB** | Tập Trôi Dạt | Đánh giá sau 90 ngày |
| `day150.npz` | **2.16 GB** | Tập Trôi Dạt | Đánh giá sau 150 ngày |
| `day270.npz` | **1.83 GB** | Tập Trôi Dạt | Đánh giá sau 270 ngày |

> [!NOTE]
> **Tổng cộng kích thước dữ liệu thô (.npz): ~12.8 GB**. Tổng số tệp dữ liệu này mô phỏng thực tế khi các phần mềm mã nguồn thiết kế web thay đổi theo thời gian, khiến mẫu hành vi lưu lượng (Traffic Fingerprint) bị thay đổi.

---

## 🛑 Thách Thức Lưu Trữ & Xử Lý

> [!WARNING]
> **Lỗi Tràn Bộ Nhớ (Out-Of-Memory/OOM)**
> Dữ liệu được đóng gói vào các mảng `numpy array`. Nếu kịch bản gọi hàm `np.load()` tải tất cả 12.8 GB này vào RAM cục bộ cùng một lúc, thiết bị sẽ phát sinh OOM và tiến trình Python sẽ lập tức tắt đột ngột (Crash).

Việc xử lý toàn bộ cơ sở dữ liệu này bắt buộc phải đi qua các kỹ thuật như:
- **Chunking / Generator Pipeline**: Đọc theo từng khối.
- **Sequential File Loading**: Tải lần lượt từng tệp `.npz` để sinh ra bộ phân tách HDF5 (`.h5`).

---

## 🛠 Phương Án Đề Xuất 

Dựa theo rào cản tài nguyên, chúng ta có các hướng tiếp cận sau:

### Lựa Chọn A: Đánh Giá Lịch Sử Phân Tách (Khuyên Dùng)

> [!TIP]
> **Đây là tiêu chuẩn của bài báo khoa học Var-CNN.**

- **Feature Vectors:** Sử dụng 100% không gian cấu trúc bài báo (Metadata 7 thông số, Hướng Packet và Inter-Arrival Times).
- **Quy trình:** Sử dụng hỗn hợp `train.npz` và `valid.npz` làm bộ dữ liệu đào tạo (Train). Mô hình sau khi huấn luyện sẽ tiếp tục "đi thi" trên các mốc `day14` cho tới `day270`.
- **Mục tiêu:** Kiểm tra độ vững chắc (Robustness) của Neural Network trước việc các trang web tự làm mới mã nguồn.

### Lựa Chọn B: Trộn Tổng Hợp (All-In-One Data)

> [!CAUTION]
> Phức tạp hoá Pipeline - Xoá nhoà yếu tố Concept Drift.

- **Quy trình:** Dùng tập lệnh ghi chép dữ liệu nối tiếp, gom chung toàn bộ 13GB trải qua 270 ngày dồn lại làm một tảng kiến trúc HDF5 khổng lồ.
- **Mục tiêu:** Nhồi ép hàng trăm ngàn lượt truy cập hỗn loạn vào mô hình để đẩy cực đại khả năng phân loại, vứt bỏ toàn bộ lịch sử thời gian.

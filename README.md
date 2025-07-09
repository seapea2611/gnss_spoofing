# GNSS DOA Spoofing Analysis Tool

Công cụ phân tích tín hiệu GNSS và phát hiện spoofing sử dụng phương pháp DOA (Direction of Arrival).

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Windows 10/11

## Cài đặt

1. Clone repository này về máy local của bạn:
```bash
git clone https://github.com/seapea2611/gnss_spoofing.git
cd GNSS_DOA_Spoofing
```

2. Tạo môi trường ảo Python (khuyến nghị):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công các thư viện sau:
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
pip install scikit-learn
```

## Cấu trúc thư mục

```
GNSS_DOA_Spoofing/
├── DoubleDifferentNewDataVersion.py  # File chính chứa code phân tích
├── hai/
│   └── ngoc/
│       └── 2025-04-24_07/           # Thư mục chứa dữ liệu
│           ├── raw_data_1.obs       # File dữ liệu quan sát 1
│           ├── raw_data_2.obs       # File dữ liệu quan sát 2
│           ├── raw_data_1.nav       # File dữ liệu navigation 1
│           └── raw_data_2.nav       # File dữ liệu navigation 2
└── PLOT/                            # Thư mục chứa các biểu đồ kết quả
```

## Cách sử dụng

1. Đảm bảo bạn đã cài đặt đầy đủ các thư viện cần thiết.

2. Chuẩn bị dữ liệu:
   - Đặt các file dữ liệu quan sát (.obs) và navigation (.nav) vào thư mục `hai/ngoc/2025-04-24_07/`
   - Đảm bảo tên file phải khớp với đường dẫn trong code

3. Chạy chương trình:
```bash
python DoubleDifferentNewDataVersion.py
```

4. Kết quả:
   - Các biểu đồ phân tích sẽ được lưu trong thư mục `PLOT/external/D2025_04_21_doplerConvert/`
   - Dữ liệu kết quả sẽ được lưu trong file `obs_data_output.txt`

## Các tham số quan trọng

- `D_rx1_rx2`: Khoảng cách giữa 2 bộ thu (mặc định: 3.2 mét)
- `angleAB`: Góc giữa 2 bộ thu (mặc định: 58 độ)
- `LINE_COUNT`: Số dòng dữ liệu tối đa đọc từ file (mặc định: 20000)

## Lưu ý

- Đảm bảo đường dẫn đến các file dữ liệu trong code khớp với cấu trúc thư mục của bạn
- Có thể cần điều chỉnh các tham số `D_rx1_rx2` và `angleAB` tùy theo setup thực tế
- Chương trình sẽ tự động tạo các thư mục cần thiết nếu chưa tồn tại

## Hỗ trợ

Nếu bạn gặp vấn đề trong quá trình cài đặt hoặc sử dụng, vui lòng tạo issue trong repository. 
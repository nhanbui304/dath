# DATH - Russell 1000 Quantitative Pipeline & In-Sample/Out-Sample Analysis

Dự án thu thập, xử lý dữ liệu và kiểm định pipeline phân tích định lượng dựa trên rổ cổ phiếu Russell 1000 (tracking qua ETF iShares Russell 1000 - IWB).

---

## 📁 Cấu trúc thư mục

```text
DATH_261/
├── data/
│   ├── raw/                       # Dữ liệu thô (holdings CSV, OHLCV thô)
│   ├── processed/                 # Dữ liệu đã xử lý và chia tập train/test
│   └── universe_tickers.json      # Danh sách tickers đã làm sạch
├── notebooks/
│   └── 01_verify_pipeline.ipynb   # Notebook kiểm tra và trực quan hóa pipeline
├── src/
│   ├── build_universe.py          # Lấy danh sách ticker Russell 1000 từ IWB
│   ├── download_data.py           # Tải lịch sử giá OHLCV qua yfinance
│   ├── preprocess_and_split.py    # Xử lý missing data, split In-sample / Out-of-sample
│   └── verify_pipeline.py         # Kiểm tra tính toàn vẹn và survivorship bias
├── Rusell.md                      # Ghi chú phương pháp luận Russell 1000 & Point-in-time
├── .gitignore                     # Bỏ qua các file rác và file dữ liệu kích thước lớn
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Chạy Pipeline

### 1. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.10+:

```bash
pip install -r requirements.txt
# hoặc cài trực tiếp:
pip install pandas numpy yfinance curl_cffi pyarrow
```

### 2. Thứ tự thực thi pipeline

1. **Khởi tạo danh sách cổ phiếu (Universe):**
   ```bash
   python src/build_universe.py
   ```
   *Tải danh sách holdings từ iShares IWB, lọc các cổ phiếu phổ thông (Equity), lưu vào `data/universe_tickers.json`.*

2. **Tải dữ liệu giá OHLCV:**
   ```bash
   python src/download_data.py
   ```
   *Tải dữ liệu lịch sử giá từ Yahoo Finance.*

3. **Tiền xử lý & Chia tập dữ liệu:**
   ```bash
   python src/preprocess_and_split.py
   ```
   *Xử lý missing value, lọc cổ phiếu thanh khoản/đủ dữ liệu, chia thành `in_sample` và `out_sample`.*

4. **Kiểm tra tính toàn vẹn (Verification):**
   ```bash
   python src/verify_pipeline.py
   ```
   *Kiểm tra số lượng cổ phiếu, tránh look-ahead/data leakage, tính ma trận sinh lợi và phân tích survivorship bias.*

5. **Notebook:**
   Mở `notebooks/01_verify_pipeline.ipynb` để trực quan hóa và kiểm tra kết quả chi tiết.

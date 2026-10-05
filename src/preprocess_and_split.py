import os
import sys
import pandas as pd

# Cấu hình bảng mã UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def loadRawOhlcvData(rawParquetPath=None):
    """
    Đọc dữ liệu OHLCV thô đã được tải từ Yahoo Finance.
    """
    if rawParquetPath is None:
        rawParquetPath = os.path.join("data", "raw", "raw_ohlcv.parquet")
        
    if not os.path.exists(rawParquetPath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu thô: {rawParquetPath}. Hãy chạy download_data.py trước.")
        
    print(f"Đang đọc dữ liệu thô từ: {rawParquetPath}...")
    rawOhlcvData = pd.read_parquet(rawParquetPath, engine="pyarrow")
    return rawOhlcvData

def filterAndPreprocessData(rawOhlcvData, missingThreshold=0.20, minRequiredTickers=500):
    """
    Trích xuất bảng giá điều chỉnh (Adj Close) và Khối lượng (Volume).
    Lọc bỏ các mã thiếu quá ngưỡng cho phép (mặc định 20%) trong giai đoạn 10 năm In-sample.
    Áp dụng Forward-fill để điền khuyết thiếu mà không làm rò rỉ dữ liệu tương lai.
    """
    # Trích xuất 2 ma trận cốt lõi phục vụ nghiên cứu định lượng
    adjCloseData = rawOhlcvData["Adj Close"].copy()
    volumeData = rawOhlcvData["Volume"].copy()
    
    # Xác định phạm vi tập In-sample để đánh giá tỷ lệ khuyết thiếu một cách khách quan
    inSampleMask = (adjCloseData.index >= "2014-01-01") & (adjCloseData.index <= "2023-12-31")
    inSampleAdjClose = adjCloseData.loc[inSampleMask]
    
    totalInSampleDays = len(inSampleAdjClose)
    print(f"Số ngày giao dịch trong tập In-sample (2014-2023): {totalInSampleDays} ngày.")
    
    # Tính tỷ lệ thiếu dữ liệu cho từng cổ phiếu trên tập In-sample
    missingRatioSeries = inSampleAdjClose.isna().mean()
    validTickerMask = missingRatioSeries <= missingThreshold
    selectedTickerList = missingRatioSeries[validTickerMask].index.tolist()
    
    totalSelected = len(selectedTickerList)
    print(f"Số mã cổ phiếu đạt chuẩn (tỷ lệ thiếu <= {missingThreshold * 100:.0f}%): {totalSelected} mã.")
    
    if totalSelected < minRequiredTickers:
        raise ValueError(
            f"Số lượng mã đạt chuẩn ({totalSelected}) thấp hơn mức tối thiểu yêu cầu ({minRequiredTickers})."
        )
        
    # Giữ lại tập cổ phiếu hợp lệ và áp dụng Forward-fill theo chuỗi thời gian
    # Lưu ý: ffill chỉ truyền giá trị gần nhất về sau, giữ nguyên các giá trị NaN trước ngày niêm yết (không dùng bfill để tránh lookahead bias)
    filteredAdjClose = adjCloseData[selectedTickerList].ffill()
    filteredVolume = volumeData[selectedTickerList].ffill().fillna(0)
    
    return filteredAdjClose, filteredVolume, selectedTickerList

def splitAndSaveParquet(filteredAdjClose, filteredVolume, processedDir=None):
    """
    Phân chia dữ liệu thành 2 tập In-sample (2014-2023) và Out-sample (2024 đến nay).
    Lưu thành các file Parquet độc lập được nén bằng thuật toán Snappy.
    """
    if processedDir is None:
        processedDir = os.path.join("data", "processed")
        
    os.makedirs(processedDir, exist_ok=True)
    
    # Thiết lập bộ lọc thời gian nghiêm ngặt theo kỷ luật phân chia dữ liệu
    inSampleMask = (filteredAdjClose.index >= "2014-01-01") & (filteredAdjClose.index <= "2023-12-31")
    outSampleMask = filteredAdjClose.index >= "2024-01-01"
    
    # Trích xuất tập In-sample (huấn luyện, khai phá Alpha và backtest)
    inSamplePrices = filteredAdjClose.loc[inSampleMask].copy()
    inSampleVolumes = filteredVolume.loc[inSampleMask].copy()
    
    # Trích xuất tập Out-sample (đóng băng hoàn toàn phục vụ kiểm thử mù)
    outSamplePrices = filteredAdjClose.loc[outSampleMask].copy()
    outSampleVolumes = filteredVolume.loc[outSampleMask].copy()
    
    # Đường dẫn lưu trữ các file Parquet
    inSamplePricePath = os.path.join(processedDir, "in_sample_prices.parquet")
    outSamplePricePath = os.path.join(processedDir, "out_sample_prices.parquet")
    inSampleVolumePath = os.path.join(processedDir, "in_sample_volumes.parquet")
    outSampleVolumePath = os.path.join(processedDir, "out_sample_volumes.parquet")
    
    # Lưu trữ bảng giá đóng cửa điều chỉnh
    inSamplePrices.to_parquet(inSamplePricePath, engine="pyarrow", compression="snappy")
    outSamplePrices.to_parquet(outSamplePricePath, engine="pyarrow", compression="snappy")
    
    # Lưu trữ bảng khối lượng giao dịch đồng bộ
    inSampleVolumes.to_parquet(inSampleVolumePath, engine="pyarrow", compression="snappy")
    outSampleVolumes.to_parquet(outSampleVolumePath, engine="pyarrow", compression="snappy")
    
    # Đồng thời lưu một file tổng hợp MultiIndex gồm cả Price và Volume để tiện tra cứu nếu cần
    inSampleCombined = pd.concat({"Adj Close": inSamplePrices, "Volume": inSampleVolumes}, axis=1)
    outSampleCombined = pd.concat({"Adj Close": outSamplePrices, "Volume": outSampleVolumes}, axis=1)
    
    inSampleAllPath = os.path.join(processedDir, "in_sample_all.parquet")
    outSampleAllPath = os.path.join(processedDir, "out_sample_all.parquet")
    inSampleCombined.to_parquet(inSampleAllPath, engine="pyarrow", compression="snappy")
    outSampleCombined.to_parquet(outSampleAllPath, engine="pyarrow", compression="snappy")
    
    print("\n=== KẾT QUẢ TIỀN XỬ LÝ VÀ PHÂN CHIA DỮ LIỆU ===")
    print(f"Tập In-sample (Prices): {inSamplePrices.shape} | Từ {inSamplePrices.index.min().strftime('%Y-%m-%d')} đến {inSamplePrices.index.max().strftime('%Y-%m-%d')}")
    print(f"Tập Out-sample (Prices): {outSamplePrices.shape} | Từ {outSamplePrices.index.min().strftime('%Y-%m-%d')} đến {outSamplePrices.index.max().strftime('%Y-%m-%d')}")
    print(f"Đã lưu in_sample_prices.parquet: {inSamplePricePath}")
    print(f"Đã lưu out_sample_prices.parquet: {outSamplePricePath}")
    print(f"Đã lưu in_sample_volumes.parquet: {inSampleVolumePath}")
    print(f"Đã lưu out_sample_volumes.parquet: {outSampleVolumePath}")
    
    return inSamplePrices, outSamplePrices

def runPreprocessingPipeline():
    """
    Quy trình điều phối toàn bộ bước tiền xử lý và chia tách dữ liệu.
    """
    rawOhlcvData = loadRawOhlcvData()
    filteredAdjClose, filteredVolume, selectedTickers = filterAndPreprocessData(
        rawOhlcvData=rawOhlcvData,
        missingThreshold=0.20,
        minRequiredTickers=500
    )
    splitAndSaveParquet(filteredAdjClose, filteredVolume)

if __name__ == "__main__":
    runPreprocessingPipeline()

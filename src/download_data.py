import json
import os
import sys
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Thiết lập bảng mã UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def loadUniverseTickers(jsonPath=None):
    """
    Đọc danh sách mã cổ phiếu từ file JSON đã được tạo ở bước build_universe.
    """
    if jsonPath is None:
        jsonPath = os.path.join("data", "universe_tickers.json")
        
    if not os.path.exists(jsonPath):
        raise FileNotFoundError(f"Không tìm thấy file danh sách mã: {jsonPath}. Hãy chạy build_universe.py trước.")
        
    with open(jsonPath, "r", encoding="utf-8") as fileHandler:
        universeData = json.load(fileHandler)
        
    tickerList = universeData.get("tickers", [])
    print(f"Đã tải {len(tickerList)} mã cổ phiếu từ {jsonPath}")
    return tickerList

def downloadOhlcvInBatches(tickerList, startDate="2014-01-01", endDate=None, batchSize=50):
    """
    Tải dữ liệu OHLCV và Adjusted Close từ Yahoo Finance theo từng batch.
    Việc chia batch giúp giảm tải bộ nhớ, tăng tốc độ xử lý song song và hạn chế bị chặn IP.
    """
    if endDate is None:
        endDate = datetime.today().strftime("%Y-%m-%d")
        
    totalTickers = len(tickerList)
    batchDataframeList = []
    failedTickerList = []
    
    print(f"Bắt đầu tải dữ liệu lịch sử từ {startDate} đến {endDate} cho {totalTickers} mã...")
    
    totalBatches = (totalTickers + batchSize - 1) // batchSize
    progressBar = tqdm(range(0, totalTickers, batchSize), total=totalBatches, desc="Tải batch dữ liệu")
    
    for batchStartIndex in progressBar:
        batchEndIndex = min(batchStartIndex + batchSize, totalTickers)
        currentBatchTickers = tickerList[batchStartIndex:batchEndIndex]
        
        try:
            # Tải dữ liệu không tự động điều chỉnh để giữ nguyên cột Close gốc và cột Adj Close riêng biệt
            batchData = yf.download(
                tickers=currentBatchTickers,
                start=startDate,
                end=endDate,
                auto_adjust=False,
                progress=False,
                threads=True
            )
            
            # Xử lý trường hợp chỉ có 1 ticker tải về (yfinance trả về DataFrame cột đơn thay vì MultiIndex)
            if len(currentBatchTickers) == 1 and not isinstance(batchData.columns, pd.MultiIndex):
                singleTicker = currentBatchTickers[0]
                batchData.columns = pd.MultiIndex.from_product([batchData.columns, [singleTicker]], names=["Price", "Ticker"])
                
            if not batchData.empty:
                batchDataframeList.append(batchData)
            else:
                failedTickerList.extend(currentBatchTickers)
                
        except Exception as errorDetails:
            print(f"\nLỗi khi tải batch [{batchStartIndex}:{batchEndIndex}]: {errorDetails}. Thử tải lại từng mã đơn lẻ...")
            for singleTicker in currentBatchTickers:
                try:
                    singleData = yf.download(
                        tickers=singleTicker,
                        start=startDate,
                        end=endDate,
                        auto_adjust=False,
                        progress=False
                    )
                    if not singleData.empty:
                        singleData.columns = pd.MultiIndex.from_product([singleData.columns, [singleTicker]], names=["Price", "Ticker"])
                        batchDataframeList.append(singleData)
                    else:
                        failedTickerList.append(singleTicker)
                except Exception:
                    failedTickerList.append(singleTicker)
                    
        # Nghỉ ngắn giữa các batch để giữ kết nối ổn định
        time.sleep(0.3)
        
    if not batchDataframeList:
        raise RuntimeError("Không tải được dữ liệu cho bất kỳ mã cổ phiếu nào.")
        
    print("\nĐang hợp nhất dữ liệu từ các batch...")
    # Ghép toàn bộ các batch theo trục ngang (cột) và loại bỏ cột trùng lặp nếu có
    combinedDataframe = pd.concat(batchDataframeList, axis=1)
    combinedDataframe = combinedDataframe.loc[:, ~combinedDataframe.columns.duplicated()]
    
    # Sắp xếp lại thứ tự index theo thời gian
    combinedDataframe.sort_index(inplace=True)
    
    if failedTickerList:
        print(f"Tổng số mã không thể tải hoặc không có dữ liệu: {len(failedTickerList)}")
        
    return combinedDataframe, failedTickerList

def runDownloadPipeline():
    """
    Quy trình chính điều phối tải toàn bộ dữ liệu OHLCV và lưu trữ dưới dạng Parquet thô.
    """
    rawParquetPath = os.path.join("data", "raw", "raw_ohlcv.parquet")
    os.makedirs(os.path.dirname(rawParquetPath), exist_ok=True)
    
    tickerList = loadUniverseTickers()
    ohlcvDataframe, failedTickers = downloadOhlcvInBatches(
        tickerList=tickerList,
        startDate="2014-01-01",
        batchSize=50
    )
    
    print(f"Kích thước ma trận dữ liệu thô: {ohlcvDataframe.shape}")
    print(f"Phạm vi thời gian: {ohlcvDataframe.index.min()} đến {ohlcvDataframe.index.max()}")
    
    # Lưu file Parquet nén Snappy để tối ưu dung lượng và tốc độ I/O
    ohlcvDataframe.to_parquet(rawParquetPath, engine="pyarrow", compression="snappy")
    print(f"Đã lưu thành công dữ liệu thô vào: {rawParquetPath}")
    
    return ohlcvDataframe

if __name__ == "__main__":
    runDownloadPipeline()

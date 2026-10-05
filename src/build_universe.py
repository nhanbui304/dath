import io
import json
import os
import re
import sys
import pandas as pd
from curl_cffi import requests

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi bảng mã cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def fetchIwbHoldingsCsv(saveRawPath=None):
    """
    Tải danh sách tài sản nắm giữ (holdings) mới nhất của quỹ iShares Russell 1000 ETF (IWB).
    Sử dụng curl_cffi để tránh bị chặn bởi cơ chế chống bot của trang BlackRock.
    """
    holdingsUrl = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    response = requests.get(holdingsUrl, headers=headers, impersonate="chrome", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Không thể tải file holdings từ iShares. Mã lỗi HTTP: {response.status_code}")
    
    csvRawText = response.text
    if saveRawPath:
        os.makedirs(os.path.dirname(saveRawPath), exist_ok=True)
        with open(saveRawPath, "w", encoding="utf-8") as fileHandler:
            fileHandler.write(csvRawText)
        print(f"Đã lưu file holdings thô thành công vào: {saveRawPath}")
        
    return csvRawText

def parseAndCleanTickers(csvRawText):
    """
    Trích xuất bảng dữ liệu holdings, lọc chỉ giữ lại cổ phiếu phổ thông (Equity),
    loại bỏ tiền mặt, công cụ phái sinh và chuẩn hóa ký tự ticker phù hợp với Yahoo Finance.
    """
    textLines = csvRawText.splitlines()
    headerRowIndex = -1
    
    # Tìm dòng bắt đầu chứa tiêu đề cột của bảng holdings
    for rowIndex, lineContent in enumerate(textLines[:30]):
        if lineContent.strip().startswith("Ticker,"):
            headerRowIndex = rowIndex
            break
            
    if headerRowIndex == -1:
        raise ValueError("Không tìm thấy dòng tiêu đề bảng hợp lệ trong dữ liệu CSV từ iShares.")
        
    cleanCsvString = "\n".join(textLines[headerRowIndex:])
    holdingsDataframe = pd.read_csv(io.StringIO(cleanCsvString))
    
    # Lọc chỉ lấy các tài sản thuộc loại cổ phiếu (Equity)
    equityMask = holdingsDataframe["Asset Class"].str.strip().str.lower() == "equity"
    filteredEquities = holdingsDataframe[equityMask].copy()
    
    cleanedTickerList = []
    excludedList = []
    
    for rawTicker in filteredEquities["Ticker"].dropna():
        cleanedTicker = str(rawTicker).strip().upper()
        
        # Bỏ qua các mã rỗng, mã tiền mặt hoặc ký hiệu giữ chỗ
        if not cleanedTicker or cleanedTicker in ["-", "USD", "CASH"]:
            continue
            
        # Chuẩn hóa các mã có khoảng trắng hoặc dấu chấm sang dấu gạch ngang theo quy ước Yahoo Finance (ví dụ: BRK B -> BRK-B)
        standardizedTicker = re.sub(r"[\s\.]+", "-", cleanedTicker)
        
        # Chỉ giữ lại các mã cổ phiếu hợp lệ chứa ký tự chữ cái và dấu gạch ngang phân tách lớp cổ phiếu
        if re.match(r"^[A-Z0-9\-]+$", standardizedTicker):
            cleanedTickerList.append(standardizedTicker)
        else:
            excludedList.append(rawTicker)
            
    # Loại bỏ các mã trùng lặp nhưng vẫn bảo toàn thứ tự ưu tiên theo vốn hóa ban đầu
    seenTickers = set()
    finalUniqueTickers = []
    for tickerItem in cleanedTickerList:
        if tickerItem not in seenTickers:
            seenTickers.add(tickerItem)
            finalUniqueTickers.append(tickerItem)
            
    if excludedList:
        print(f"Các mã không hợp lệ đã bị loại bỏ: {excludedList}")
        
    return finalUniqueTickers

def buildUniverse():
    """
    Hàm thực thi chính: Tải holdings, trích xuất danh sách mã đạt chuẩn Russell 1000
    và lưu thành file JSON phục vụ cho các bước tải dữ liệu tiếp theo.
    """
    rawHoldingsCsvPath = os.path.join("data", "raw", "iwb_holdings.csv")
    outputJsonPath = os.path.join("data", "universe_tickers.json")
    
    print("Đang bắt đầu quá trình trích xuất danh sách vũ trụ cổ phiếu Russell 1000...")
    
    # Kiểm tra xem file holdings thô đã tồn tại cục bộ chưa nhằm tiết kiệm băng thông
    if os.path.exists(rawHoldingsCsvPath):
        print(f"Đọc dữ liệu holdings từ file cục bộ: {rawHoldingsCsvPath}")
        with open(rawHoldingsCsvPath, "r", encoding="utf-8") as fileHandler:
            csvContent = fileHandler.read()
    else:
        csvContent = fetchIwbHoldingsCsv(saveRawPath=rawHoldingsCsvPath)
        
    tickerUniverse = parseAndCleanTickers(csvContent)
    print(f"Tổng số mã cổ phiếu hợp lệ trích xuất được: {len(tickerUniverse)} mã.")
    
    universeMetadata = {
        "universeName": "Russell 1000 Proxy (iShares IWB Holdings)",
        "source": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf",
        "totalTickers": len(tickerUniverse),
        "tickers": tickerUniverse
    }
    
    with open(outputJsonPath, "w", encoding="utf-8") as jsonFileHandler:
        json.dump(universeMetadata, jsonFileHandler, indent=2)
        
    print(f"Đã lưu danh sách vũ trụ cổ phiếu vào: {outputJsonPath}")
    return tickerUniverse

if __name__ == "__main__":
    buildUniverse()

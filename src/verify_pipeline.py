import os
import sys
import numpy as np
import pandas as pd

# Thiết lập bảng mã UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def verifyDataIntegrity(parquetFilePath=None):
    """
    Kiểm tra tính toàn vẹn của tập dữ liệu In-sample:
    Xác minh số lượng mã cổ phiếu tối thiểu đạt chuẩn (>= 500)
    và mốc thời gian không được vượt quá ngày 2023-12-31 để tránh rò rỉ dữ liệu.
    """
    if parquetFilePath is None:
        parquetFilePath = os.path.join("data", "processed", "in_sample_prices.parquet")
        
    if not os.path.exists(parquetFilePath):
        raise FileNotFoundError(f"Không tìm thấy file In-sample: {parquetFilePath}. Hãy chạy preprocess_and_split.py trước.")
        
    print(f"Đang kiểm tra tính toàn vẹn của file: {parquetFilePath}...")
    priceData = pd.read_parquet(parquetFilePath, engine="pyarrow")
    
    # Hỗ trợ tự động trích xuất nếu DataFrame lưu dưới dạng MultiIndex
    if isinstance(priceData.columns, pd.MultiIndex):
        if "Adj Close" in priceData.columns.levels[0]:
            priceMatrix = priceData["Adj Close"]
        else:
            priceMatrix = priceData
    else:
        priceMatrix = priceData
        
    tickerCount = priceMatrix.shape[1]
    tradingDayCount = priceMatrix.shape[0]
    minDate = priceMatrix.index.min()
    maxDate = priceMatrix.index.max()
    
    print(f"Số lượng cổ phiếu hợp lệ: {tickerCount} mã.")
    print(f"Số ngày giao dịch In-sample: {tradingDayCount} ngày.")
    print(f"Khoảng thời gian: {minDate.strftime('%Y-%m-%d')} đến {maxDate.strftime('%Y-%m-%d')}")
    
    # Kiểm tra ràng buộc tối thiểu 500 mã
    if tickerCount < 500:
        raise AssertionError(f"Số lượng mã ({tickerCount}) không đạt chuẩn tối thiểu 500 mã theo yêu cầu.")
        
    # Kiểm tra kỷ luật đóng băng dữ liệu (không vượt quá 2023-12-31)
    cutoffDate = pd.Timestamp("2023-12-31")
    if maxDate > cutoffDate:
        raise AssertionError(f"Phát hiện rò rỉ dữ liệu: Ngày lớn nhất {maxDate} vượt quá mốc đóng băng {cutoffDate}!")
        
    print("Xác minh tính toàn vẹn thành công: Dữ liệu In-sample hợp lệ và cách ly an toàn.")
    return priceMatrix

def computeBaselineFiveDayReversal(priceMatrix):
    """
    Tính toán tín hiệu Alpha của chiến lược Baseline 5-day Reversal:
    Công thức hướng dẫn: rawAlpha = (price.shift(5) - price.shift(1)) / 5
    """
    # Tính hiệu số giá giữa 5 phiên trước và phiên liền kề trước đó
    rawAlpha = (priceMatrix.shift(5) - priceMatrix.shift(1)) / 5.0
    return rawAlpha

def constructDollarNeutralPortfolio(rawAlpha):
    """
    Khử trung bình lát cắt chéo (Demean) và chuẩn hóa vốn theo chuẩn L1 (L1 Normalization).
    Đảm bảo tổng vị thế mua (Long = +0.5) và bán khống (Short = -0.5) triệt tiêu nhau,
    tổng vốn ròng bằng 0 (Dollar-Neutral).
    """
    # Khử trung bình theo từng ngày (cross-sectional demean)
    crossSectionalMean = rawAlpha.mean(axis=1)
    demeanedAlpha = rawAlpha.sub(crossSectionalMean, axis=0)
    
    # Tính chuẩn L1 (tổng trị tuyệt đối trọng số lát cắt) cho từng ngày
    l1Norm = demeanedAlpha.abs().sum(axis=1)
    
    # Chuẩn hóa tỷ trọng vốn: tổng |w_i| = 1.0 (50% Long và 50% Short)
    capitalWeights = demeanedAlpha.div(l1Norm, axis=0)
    
    return capitalWeights

def evaluateStrategyPerformance(priceMatrix, capitalWeights, rawAlpha):
    """
    Tính toán PnL, Annualized Return, Annualized Volatility, Sharpe Ratio
    và Rank Information Coefficient (Spearman Rank IC).
    """
    # Lợi nhuận kỳ vọng phiên kế tiếp: r_{i, t+1} = (P_{t+1} - P_t) / P_t
    forwardReturns = priceMatrix.pct_change().shift(-1)
    
    # Lợi nhuận danh mục hàng ngày: PnL_t = sum(w_{i, t} * r_{i, t+1})
    portfolioDailyReturns = (capitalWeights * forwardReturns).sum(axis=1)
    
    # Loại bỏ các ngày đầu thiếu dữ liệu do shift(5) và ngày cuối cùng do shift(-1)
    validPnlSeries = portfolioDailyReturns.dropna().iloc[5:-1]
    
    # Chuẩn hóa theo năm với 252 ngày giao dịch của thị trường Mỹ
    annualizedReturn = validPnlSeries.mean() * 252.0
    annualizedVolatility = validPnlSeries.std() * np.sqrt(252.0)
    sharpeRatio = annualizedReturn / annualizedVolatility if annualizedVolatility != 0 else 0.0
    
    # Tính tỷ suất sinh lời tích lũy và độ sụt giảm tối đa (Max Drawdown)
    cumulativeReturns = (1.0 + validPnlSeries).cumprod()
    peakReturns = cumulativeReturns.cummax()
    drawdownSeries = (cumulativeReturns - peakReturns) / peakReturns
    maxDrawdown = drawdownSeries.min()
    
    # Tính Rank Information Coefficient (Rank IC) tốc độ cao bằng tương quan Pearson trên thứ hạng chuẩn hóa
    rankAlpha = rawAlpha.rank(axis=1, pct=True)
    rankReturns = forwardReturns.rank(axis=1, pct=True)
    
    centeredRankAlpha = rankAlpha.sub(rankAlpha.mean(axis=1), axis=0)
    centeredRankReturns = rankReturns.sub(rankReturns.mean(axis=1), axis=0)
    
    covarianceSeries = (centeredRankAlpha * centeredRankReturns).mean(axis=1)
    varianceAlphaSeries = (centeredRankAlpha ** 2).mean(axis=1)
    varianceReturnsSeries = (centeredRankReturns ** 2).mean(axis=1)
    
    dailyRankIc = (covarianceSeries / np.sqrt(varianceAlphaSeries * varianceReturnsSeries)).dropna().iloc[5:-1]
    meanRankIc = dailyRankIc.mean()
    stdRankIc = dailyRankIc.std()
    icirValue = (meanRankIc / stdRankIc) * np.sqrt(252.0) if stdRankIc != 0 else 0.0
    
    # Kiểm tra điều kiện Dollar-Neutral
    dailyNetHolding = capitalWeights.sum(axis=1).dropna().iloc[5:-1]
    maxAbsoluteNetHolding = dailyNetHolding.abs().max()
    dailyLongExposure = capitalWeights.clip(lower=0).sum(axis=1).dropna().iloc[5:-1].mean()
    dailyShortExposure = capitalWeights.clip(upper=0).sum(axis=1).dropna().iloc[5:-1].mean()
    
    performanceMetrics = {
        "tickerCount": priceMatrix.shape[1],
        "tradingDays": len(validPnlSeries),
        "annualizedReturn": annualizedReturn,
        "annualizedVolatility": annualizedVolatility,
        "sharpeRatio": sharpeRatio,
        "maxDrawdown": maxDrawdown,
        "meanRankIc": meanRankIc,
        "icir": icirValue,
        "maxAbsoluteNetHolding": maxAbsoluteNetHolding,
        "meanLongExposure": dailyLongExposure,
        "meanShortExposure": dailyShortExposure
    }
    
    return performanceMetrics, validPnlSeries, dailyRankIc

def printSummaryTable(metrics):
    """
    In bảng tóm tắt kết quả kiểm định theo định dạng chuyên nghiệp ra terminal.
    """
    borderLine = "=" * 65
    dividerLine = "-" * 65
    
    print("\n" + borderLine)
    print("   BẢNG TỔNG HỢP KIỂM ĐỊNH PIPELINE & BASELINE 5-DAY REVERSAL")
    print(borderLine)
    print(f"Số lượng cổ phiếu hợp lệ (In-sample)  : {metrics['tickerCount']:>12d} mã")
    print(f"Tổng số phiên giao dịch backtest      : {metrics['tradingDays']:>12d} phiên")
    print(dividerLine)
    print("1. KIỂM ĐỊNH VỊ THẾ CÂN BẰNG DOLLAR-NEUTRAL:")
    print(f"   - Tỷ trọng vị thế Long trung bình  : {metrics['meanLongExposure']:>12.4f} (+50%)")
    print(f"   - Tỷ trọng vị thế Short trung bình : {metrics['meanShortExposure']:>12.4f} (-50%)")
    print(f"   - Trị tuyệt đối vị thế ròng tối đa : {metrics['maxAbsoluteNetHolding']:>12.2e} (~ 0.0)")
    print(dividerLine)
    print("2. HIỆU SUẤT ĐẦU TƯ (PORTFOLIO PERFORMANCE):")
    print(f"   - Lợi nhuận chuẩn hóa năm (Ann. Ret): {metrics['annualizedReturn'] * 100:>11.2f}%")
    print(f"   - Độ biến động năm (Ann. Volatility): {metrics['annualizedVolatility'] * 100:>11.2f}%")
    print(f"   - Tỷ số Sharpe (Sharpe Ratio)       : {metrics['sharpeRatio']:>12.4f}")
    print(f"   - Mức sụt giảm tối đa (Max Drawdown): {metrics['maxDrawdown'] * 100:>11.2f}%")
    print(dividerLine)
    print("3. CHỈ SỐ THÔNG TIN DỰ BÁO (INFORMATION METRICS):")
    print(f"   - Hệ số tương quan Rank IC trung bình: {metrics['meanRankIc']:>12.4f}")
    print(f"   - Tỷ số thông tin Rank ICIR (năm)   : {metrics['icir']:>12.4f}")
    print(borderLine + "\n")

def runVerification():
    """
    Quy trình điều phối kiểm tra toàn bộ pipeline từ đọc dữ liệu đến in báo cáo.
    """
    priceMatrix = verifyDataIntegrity()
    rawAlpha = computeBaselineFiveDayReversal(priceMatrix)
    capitalWeights = constructDollarNeutralPortfolio(rawAlpha)
    metrics, pnlSeries, rankIcSeries = evaluateStrategyPerformance(priceMatrix, capitalWeights, rawAlpha)
    printSummaryTable(metrics)
    return metrics

if __name__ == "__main__":
    runVerification()

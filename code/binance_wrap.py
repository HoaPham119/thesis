import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.stats import skew, kurtosis

import warnings
warnings.filterwarnings('ignore')


class BinanceWrap:
    def __init__(self, file_path, symbol="BNBUSDT", window=50, h=50, ma_short_window=20, ma_long_window=50):
        self.symbol = symbol
        self.file_path = file_path
        self.window = window
        self.h = h
        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')
        cols = [
            "tradeId",
            "price",
            "quantity",
            "firstTradeId",
            "lastTradeId",
            "timestamp",
            "buyerMaker",
            "bestPriceMatch",
        ]
        df.columns = cols
        df = df[["timestamp", "price", "quantity", "buyerMaker"]]
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[["datetime", "price", "quantity", "buyerMaker"]]
        df['datetime'] = df['datetime'].dt.floor('S')
        return df

    def get_agg(self, df):
        df['side'] = np.where(df['buyerMaker'], 'sell', 'buy')
        agg = (df.groupby(['datetime', 'side'])
               .agg(price_mean=('price', 'mean'),
                    qty_sum=('quantity', 'sum'))
               .reset_index()
               )
        price_wide = agg.pivot(index='datetime', columns='side', values='price_mean').add_prefix('price_')
        qty_wide   = agg.pivot(index='datetime', columns='side', values='qty_sum').add_prefix('qty_')
        out_df = pd.concat([price_wide, qty_wide], axis=1).fillna(0.0)
        out_df = out_df[['price_buy', 'price_sell', 'qty_buy', 'qty_sell']].sort_index()
        return out_df

    def calc_V(self, out_df):
        # Tính V
        out_df["total_qty"] = out_df["qty_buy"] + out_df["qty_sell"]
        # resample theo ngày
        daily_vol = out_df["total_qty"].resample("D").sum()
        # bỏ ngày đầu và ngày cuối vì không đủ dữ liệu
        daily_vol = daily_vol.iloc[1:-1]
        # Tính Volume Bucket
        V = int(daily_vol.mean() / 50)
        out_df = out_df.drop(columns=["total_qty"])
        return V

    def get_buckets(self, df, bucketSize: float) -> pd.DataFrame:
        d = df.copy()
        buckets = []
        BV = SV = filled = 0.0  # Buy Vol, Sell Vol, đã lấp đầy trong bucket hiện tại

        # tích lũy cho trung bình giá mua/bán
        bid_price_num = 0.0   # sum(alloc_buy * price_buy)
        ask_price_num = 0.0   # sum(alloc_sell * price_sell)
        # sum((alloc_buy*price_buy + alloc_sell*price_sell))
        total_price_num = 0.0

        for ts, row in d.iterrows():
            buy_remain = float(row['qty_buy'])
            sell_remain = float(row['qty_sell'])
            total_remain = buy_remain + sell_remain

            while total_remain > 0:
                space = bucketSize - filled
                take = min(space, total_remain)

                # phân bổ theo tỷ lệ buy/sell còn lại
                buy_share = (
                    buy_remain / total_remain) if total_remain > 0 else 0.0
                alloc_buy = take * buy_share
                alloc_sell = take - alloc_buy

                # cộng dồn volume
                BV += alloc_buy
                SV += alloc_sell

                # cộng dồn cho từng loại giá
                if alloc_buy > 0:
                    bid_price_num += alloc_buy * float(row['price_buy'])
                    total_price_num += alloc_buy * float(row['price_buy'])
                if alloc_sell > 0:
                    ask_price_num += alloc_sell * float(row['price_sell'])
                    total_price_num += alloc_sell * float(row['price_sell'])

                # cập nhật trạng thái
                filled += take
                buy_remain -= alloc_buy
                sell_remain -= alloc_sell
                total_remain = buy_remain + sell_remain

                # đủ bucket → ghi lại
                if filled >= bucketSize - 1e-12:
                    total_vol = BV + SV
                    bid_mean = (bid_price_num / BV) if BV > 0 else np.nan
                    ask_mean = (ask_price_num / SV) if SV > 0 else np.nan
                    avg_price = (total_price_num /
                                 total_vol) if total_vol > 0 else np.nan

                    buckets.append({
                        'Time': ts,
                        'Buy': BV,
                        'Sell': SV,
                        'Price': avg_price,    # giá chung (VWAP toàn bucket)
                        'BidPrice': bid_mean,  # giá mua trung bình
                        'AskPrice': ask_mean   # giá bán trung bình
                    })

                    # reset cho bucket mới
                    BV = SV = filled = 0.0
                    bid_price_num = ask_price_num = total_price_num = 0.0

        return pd.DataFrame(buckets)

    def calc_input_var(self, df, bucketSize):
        data = self.get_buckets(df, bucketSize)
        data["Volume"] = data["Buy"] + data["Sell"]
        # VPIN: rolling mean của |Buy - Sell| / V
        data['VPIN'] = abs(data['Buy'] - data['Sell']
                           ).rolling(self.window).mean() / bucketSize
        # Tính DeltaPrice:
        data["DeltaPrice"] = data['Price'].diff()
        data['DeltaPrice_lag'] = data['DeltaPrice'].shift(1)
        data["Roll"] = 2 * np.sqrt(abs(data["DeltaPrice"].rolling(
            window=self.window).cov(data["DeltaPrice_lag"])))
        data["RollImpact"] = data["Roll"] / (data["Volume"]*data["Price"])
        data.drop(["DeltaPrice", "DeltaPrice_lag"], axis=1, inplace=True)
        data["KyleLambda"] = (data["Price"].shift(self.window) - data["Price"]) / \
            ((data["Volume"] * np.sign(data['Price'].diff())
              ).rolling(window=self.window).sum())
        data['Returns'] = data['Price'].pct_change()
        data["AmihudLambda"] = (1/self.window)/((abs(data['Returns']) /
                                                 (data['Volume'] * data['Price'])).rolling(window=self.window).sum())

        return data

    def calc_output_var(self, data):
        # 1. Tính Dấu của sự thay đổi Bid-Ask Spread
        data['BidAskSpread'] = data['AskPrice'] - data['BidPrice']
        data['SpreadChangeSign'] = np.sign(data['BidAskSpread'].diff(self.h))
        # Tính độ lệch chuẩn cuộn (Realized Volatility) và dấu hiệu thay đổi
        data['RealizedVolatility'] = data['Returns'].rolling(
            window=self.window).std()
        data['RealizedVolatilitySign'] = np.sign(
            data['RealizedVolatility'].diff(self.h))
        data["ReturnsSign"] = np.sign(data['Returns'].diff(self.h))
        # Tính Skewness và Kurtosis
        data['Skewness'] = data['Returns'].rolling(
            window=self.window).apply(skew)
        data['Kurtosis'] = data['Returns'].rolling(
            window=self.window).apply(kurtosis)

        def jarque_bera_statistic(skewness, kurtosis, window):
            # Tính Jarque-Bera Statistic theo công thức
            return (window / 6) * (skewness**2 + ((kurtosis - 3)**2) / 4)
        data['JB_rolling'] = data.apply(lambda row: jarque_bera_statistic(
            row['Skewness'],
            row['Kurtosis'],
            self.window
        )
            if not np.isnan(row['Skewness']) else np.nan, axis=1)
        # Tính dấu của sự thay đổi trong thống kê Jarque-Bera
        data['JB_ChangeSign'] = np.sign(data['JB_rolling'].diff(self.h))
        data['AR'] = data['Returns'].rolling(window=self.window).apply(
            lambda x: x.autocorr(lag=1), raw=False)
        data['AR_ChangeSign'] = np.sign(data['AR'].diff(self.h))
        data['SkewnessSign'] = np.sign(data['Skewness'].diff(self.h))
        data['KurtosisSign'] = np.sign(data['Kurtosis'].diff(self.h))

        data['MA_20'] = data['Price'].rolling(
            window=self.ma_short_window).mean()
        data['MA_50'] = data['Price'].rolling(
            window=self.ma_long_window).mean()
        # Dấu hiệu cắt nhau của MA ngắn hạn và dài hạn
        data['MASign'] = np.where(data['MA_20'] > data['MA_50'], 1, -1)
        return data


if __name__ == "__main__":
    symbol = "BNBUSDT"
    if not Path("Users").exists():  # Windows
        input_path = r"C:\Users\phamhoa\Downloads\thesis\data\Binance\agg\500"
        file_path = rf"{input_path}\{symbol}.csv"
    else:  # Macbook
        input_path = "/Users/hoapham/Documents/Learning/thesis/data/Binance/agg/500"
        file_path = f"{input_path}/{symbol}.csv"
    binance = BinanceWrap(file_path, symbol)
    df = binance.load_data()
    df = binance.get_agg(df)
    V = binance.calc_V(df)
    data = binance.calc_input_var(df, V)
    data = binance.calc_output_var(data)
    data

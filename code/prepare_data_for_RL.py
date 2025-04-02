import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import os
from data_load import load_data

def get_buckets(df, bucketSize):
    volumeBuckets = []
    count = 0
    BV = 0
    SV = 0
    prices = []
    for index, row in df.iterrows():
        newVolume = row['volume']
        z = row['z']
        price = row['price']
        if bucketSize < count + newVolume:
            portion = bucketSize - count
            BV += portion * z
            SV += portion * (1 - z)
            prices.append(price)
            volumeBuckets.append({'Buy': BV, 'Sell': SV, 'Time': index, 'Price': np.mean(prices)})

            count = newVolume - portion
            prices = [price]

            while count >= bucketSize:
                BV = bucketSize * z
                SV = bucketSize * (1 - z)
                volumeBuckets.append({'Buy': BV, 'Sell': SV, 'Time': index, 'Price': price})
                count -= bucketSize

            BV = count * z
            SV = count * (1 - z)
        else:
            BV += newVolume * z
            SV += newVolume * (1 - z)
            count += newVolume
            prices.append(price)

    volumeBuckets = pd.DataFrame(volumeBuckets).set_index('Time')
    return volumeBuckets


def calc_vpin(data, bucketSize, window):
    volume = data['SIZE']
    trades = data['PRICE']

    trades_1min = trades.diff(1).resample('1min').sum().dropna()
    volume_1min = volume.resample('1min').sum().dropna()
    sigma = trades_1min.std()
    z = trades_1min.apply(lambda x: norm.cdf(x / sigma))
    price_1min = trades.resample('1min').last().dropna()

    df = pd.DataFrame({'z': z, 'volume': volume_1min, 'price': price_1min}).dropna()

    volumeBuckets = get_buckets(df, bucketSize)
    volumeBuckets['VPIN'] = abs(volumeBuckets['Buy'] - volumeBuckets['Sell']).rolling(window).mean() / bucketSize
    volumeBuckets['CDF'] = volumeBuckets['VPIN'].rank(pct=True)

    return volumeBuckets

if __name__ == "__main__":
    # Đường dẫn tới thư mục cần kiểm tra
    folder_path = "RL_data"
    # Kiểm tra nếu thư mục chưa tồn tại thì tạo mới
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Đã tạo thư mục: {folder_path}")
    else:
        print(f"Thư mục '{folder_path}' đã tồn tại.")
        
    df={}; sec_trades = {}
    sym = ['STB', 'SAB','MWG', 'VCB','TCB'] 
    
    ## Load data
    data_tick = load_data(folder="tick")
    data_orderbook = load_data(folder="orderbook")
    
    # Transform data
    for s in sym:
        data = data_tick[s].copy()
        data.rename(columns = {"Gia KL": "PRICE", "KL": "SIZE"}, inplace = True)
        data.set_index("Date", inplace = True)
        data = data.resample("T").agg({
                'SIZE': 'sum',  # Cột volume tính tổng
                'PRICE': 'mean'    # Cột price tính trung bình
            })
        data.dropna(how = "any", inplace = True)
        data.to_csv(f"{folder_path}/{s}price.csv")
        sec_trades[s] = data
    
        # Set volume
    volume = {}
    for key, val in sec_trades.items():
        volume[key] = int(val['SIZE'].resample("D").sum().mean()/50) # Sum của từng ngày, rồi lấy mean, rồi chia 50
    volume

    # Cal vpin
    for s in sym:
        print('Calculating VPIN')
        df[s] = calc_vpin(sec_trades[s],volume[s],50)
        df[s].to_csv(f"{folder_path}/{s}VPIN.csv",index = True)
        print(s+' '+str(df[s].shape))
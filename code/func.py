import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm

def get_buckets(df,bucketSize):
    volumeBuckets = pd.DataFrame(columns=['Buy','Sell','Time'])
    count = 0
    BV = 0
    SV = 0
    for index,row in df.iterrows():
        newVolume = row['volume']
        z = row['z']
        if bucketSize < count + newVolume:
            BV = BV + (bucketSize-count)*z
            SV = SV + (bucketSize-count)*(1-z)
            # volumeBuckets = volumeBuckets.append({'Buy':BV, 'Sell':SV, 'Time':index},ignore_index=True)
            volumeBuckets = pd.concat([volumeBuckets, pd.DataFrame([{'Buy':BV, 'Sell':SV, 'Time':index}])],ignore_index=True)
            count = newVolume-(bucketSize-count)
            if int(count/bucketSize) > 0:
                for i in range(0,int(count/bucketSize)):
                    BV = (bucketSize)*z
                    SV = (bucketSize)*(1-z)
                    # volumeBuckets = volumeBuckets.append({'Buy':BV, 'Sell':SV, 'Time':index},ignore_index=True)
                    volumeBuckets = pd.concat([volumeBuckets, pd.DataFrame([{'Buy':BV, 'Sell':SV, 'Time':index}])],ignore_index=True)

            count = count%bucketSize
            BV = (count)*z
            SV = (count)*(1-z)
        else:
            BV = BV + (newVolume)*z
            SV = SV + (newVolume)*(1-z)
            count = count + newVolume

    volumeBuckets = volumeBuckets.set_index('Time')
    return volumeBuckets

def calc_vpin(data, bucketSize,window):
    volume = (data['SIZE'])
    trades = (data['PRICE'])
    
    trades_1min = trades.diff(1).resample('1min').sum().dropna()
    volume_1min = volume.resample('1min').sum().dropna()
    sigma = trades_1min.std()
    z = trades_1min.apply(lambda x: norm.cdf(x/sigma))
    df = pd.DataFrame({'z': z, 'volume': volume_1min}).dropna()
    
    volumeBuckets=get_buckets(df,bucketSize)
    volumeBuckets['VPIN'] = abs(volumeBuckets['Buy']-volumeBuckets['Sell']).rolling(window).mean()/bucketSize
    volumeBuckets['CDF'] = volumeBuckets['VPIN'].rank(pct=True)
    
    return volumeBuckets

if __name__ == "__main__":
    # Khai báo biến
    sym = ['C','BAC','USB','JPM','WFC']
    start="2025-02-21"
    end="2025-02-28"
    df={}; sec_trades = {}

    # Lấy dữ liệu từ yfnance
    for s in sym:
        print('Reading '+s)
        data = yf.download(s, start=start, end=end, interval="1m")
        data.columns = data.columns.droplevel(1)
        data = data[["Close", "Volume"]]
        data = data.rename(columns = {"Close": "PRICE", "Volume": "SIZE"})
        sec_trades[s] = data.copy()

    volume = {'C':300000,'USB':100000,'JPM':200000,'BAC':1250000,'WFC':300000}
    for s in sym:
        print('Calculating VPIN')
        df[s] = calc_vpin(sec_trades[s],volume[s],50)
        print(s+' '+str(df[s].shape))
        
    avg = pd.DataFrame()
    print(avg.shape)
    metric = 'CDF'
    avg[metric] = np.nan
    for stock,frame in df.items():
        frame = frame[[metric]].reset_index().drop_duplicates(subset='Time', keep='last').set_index('Time')
        avg = avg.merge(frame[[metric]],left_index=True,right_index=True,how='outer',suffixes=('',stock))
        print(avg.shape)
    avg = avg.dropna(axis=0,how='all').fillna(method='ffill')

    avg.to_csv('CDF.csv')

    


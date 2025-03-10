import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm

def get_buckets(df,bucketSize):
    volumeBuckets = pd.DataFrame(columns=['Buy','Sell','Time'])
    count = 0
    BV = 0
    SV = 0
    # Tính theo công thức số 7 trong bài báo
    for index,row in df.iterrows():
        newVolume = row['volume']
        z = row['z']
        if bucketSize < count + newVolume:
            BV = BV + (bucketSize-count)*z
            SV = SV + (bucketSize-count)*(1-z)
            volumeBuckets = pd.concat([volumeBuckets, pd.DataFrame([{'Buy':BV, 'Sell':SV, 'Time':index}])],ignore_index=True)
            count = newVolume-(bucketSize-count)
            if int(count/bucketSize) > 0:
                for i in range(0,int(count/bucketSize)):
                    BV = (bucketSize)*z
                    SV = (bucketSize)*(1-z)
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
    # Không thể fillnan = 0 ở đây, vì sẽ làm tính sai diff() về sau
    volume = (data['SIZE'])
    trades = (data['PRICE'])
    
    trades_1min = trades.diff(1).resample('1min').sum().dropna()
    volume_1min = volume.resample('1min').sum().dropna()
    sigma = trades_1min.std()
    z = trades_1min.apply(lambda x: norm.cdf(x/sigma)) # norm.cdf(...): Trả về xác suất tích luỹ của pp chuẩn tắc (mean = 0, var = 1)
    df = pd.DataFrame({'z': z, 'volume': volume_1min}).dropna()
    
    volumeBuckets=get_buckets(df,bucketSize)
    volumeBuckets['VPIN'] = abs(volumeBuckets['Buy']-volumeBuckets['Sell']).rolling(window).mean()/bucketSize
    volumeBuckets['CDF'] = volumeBuckets['VPIN'].rank(pct=True) #Trả về một Series chứa phân vị (giá trị từ 0 đến 1) của các giá trị trong cột VPIN
    
    return volumeBuckets


import pandas as pd

def imbalance(sec_quotes):
    # Lọc các dòng có giá và khối lượng hợp lệ
    sec_quotes = sec_quotes[(sec_quotes['BID'] > 0) & (sec_quotes['BIDSIZ'] > 0) &
                            (sec_quotes['ASK'] > 0) & (sec_quotes['ASKSIZ'] > 0)]
    
    # Resample theo từng phút, lấy giá trị cuối cùng
    df_resampled = sec_quotes.resample('1min').last().ffill().fillna(0)
    
    # Lấy giá và khối lượng đặt mua/bán sau resample
    bid_vol = df_resampled['BIDSIZ']
    ask_vol = df_resampled['ASKSIZ']

    # Tính toán imbalance trực tiếp từ volume
    df_resampled['quote_imb'] = bid_vol - ask_vol

    # Chỉ trả về cột imbalance dưới dạng DataFrame
    return df_resampled[['quote_imb']]

# def imbalance(sec_quotes):
#     sec_bids=sec_quotes[sec_quotes['BID']>0]
#     sec_bids=sec_bids[sec_bids['BIDSIZ']>0]
#     sec_asks=sec_quotes[sec_quotes['ASK']>0]
#     sec_asks=sec_asks[sec_quotes['ASKSIZ']>0]
#     bids=(sec_bids['BID'])
#     bids_vol=(sec_bids['BIDSIZ'])

#     asks=(sec_asks['ASK'])
#     asks_vol=(sec_asks['ASKSIZ'])
    
#     df_comb=pd.DataFrame()
#     df=pd.DataFrame()
#     df1=pd.DataFrame()
#     bidquote_1min = bids.resample('1min').last().ffill().fillna(0)
#     bidvol_1min = bids_vol.resample('1min').last().ffill().fillna(0)
#     askquote_1min = asks.resample('1min').last().ffill().fillna(0)
#     askvol_1min = asks_vol.resample('1min').last().ffill().fillna(0)
#     df1=pd.concat([bidquote_1min, bidvol_1min,askquote_1min,askvol_1min], join='outer', axis=1)
#     df1=df1.ffill().fillna(0)
#     df['bprice']=df1['BID']
#     df['bvol']=df1['BIDSIZ']
#     df['aprice']=df1['ASK']
#     df['avol']=df1['ASKSIZ']

#     df_comb=df.copy()
#     df_comb['avg_bid']=df['bprice']*df['bvol']
#     df_comb['price_bid']=df['bprice']
#     df_comb['avg_ask']=df['aprice']*df['avol']
#     df_comb['price_ask']=df['aprice']
#     n1=df_comb['bprice'].apply(lambda x: int(x!=0)) # Check True-False nhưng trả về số
#     n2=df_comb['aprice'].apply(lambda x: int(x!=0))

#     df_comb['avg_bid']=df_comb['avg_bid']*n1/df_comb['price_bid']
#     df_comb['avg_ask']=df_comb['avg_ask']*n2/df_comb['price_ask']
#     imbalance=df_comb['avg_bid']-df_comb['avg_ask']
#     imbalance = pd.DataFrame(imbalance).rename(columns = {0: "quote_imb"})
#     return imbalance



if __name__ == "__main__":
    # Khai báo biến
    sym = ['C','BAC','USB','JPM','WFC']
    start="2025-02-21"
    end="2025-02-28"
    df={}; sec_trades = {}

    # Lấy dữ liệu từ yfinance
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
    
    fields = ['Time','CDFC','CDFBAC','CDFUSB','CDFJPM','CDFWFC']
    df = pd.read_csv('CDF.csv',parse_dates=['Time'],index_col=[0],usecols = fields)

    rolling_pariwise_corr = df.rolling(window=50).corr()

    thres = pd.DataFrame()
    thres['AvgCorrAssets'] = rolling_pariwise_corr.groupby(by=['Time']).sum().sum(axis=1)/((len(fields)-1)**2)
    thres.to_csv('AvgCorrAssets.csv')



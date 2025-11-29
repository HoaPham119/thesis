import requests
import json, os
import time
from datetime import datetime
import pandas as pd
output_path = "/Users/hoapham/Documents/Learning/thesis/data/Binance/agg/500"
# output_path = r"C:\Users\phamhoa\Downloads\thesis\data\Binance\agg\500"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def call_candlestick_api(symbol: str = "BTCUSDT", interval = "1m", limit = 1000, endTime = None):
    # Gọi API Binance
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
        }   
    if endTime is not None:
       params["endTime"] =  endTime

    response = requests.get(url, params=params)
    klines = response.json()
    klines = pd.DataFrame(klines)
    endTime = klines[0].to_list()[0]-1
    return  klines, endTime

def get_symbol_list():
    rsp = requests.get("https://api.binance.com/api/v3/exchangeInfo?permissions=SPOT")
    rsp = rsp.json()
    symbols = rsp["symbols"]
    symbols = pd.DataFrame(symbols)
    symbols = symbols[symbols["status"] == "TRADING"]
    symbols = symbols["symbol"].to_list()[:50]
    return symbols

def get_candledstick_data():
    symbols_df = pd.read_csv("symbol.csv")
    symbol_list = symbols_df["symbol"].to_list()[:25]
    for symbol in symbol_list:
        endTime = None
        i = 0
        # df = pd.DataFrame()
        df = pd.read_csv(f"{output_path}/{symbol}.csv")
        while i < 50:
            klines, endTime = call_candlestick_api(symbol = symbol,
                                       endTime = endTime)
            df = pd.concat([df, klines]).reset_index(drop = True)
            i+=1
        df.to_csv(f"{output_path}/{symbol}.csv")
    print()
    
def call_orderbookticker_api(symbol: str = "BTCUSDT", interval = "1m", limit = 1000, endTime = None, startTime = None):
    # Gọi API Binance
    url = 'https://api.binance.com/api/v3/aggTrades'
    params = {
        'symbol': symbol,
        'limit': limit
        }   
    if endTime is not None:
       params["endTime"] =  endTime
    if startTime is not None:
       params["startTime"] =  startTime
    response = requests.get(url, params=params)
    klines = response.json()
    klines = pd.DataFrame(klines)
    endTime = klines["T"].to_list()[0]-1
    startTime = klines["T"].to_list()[-1]+1
    return  klines, startTime, endTime


current_ts_ms = int(time.time() * 1000)
def get_orderbookticker_data():
    # symbols_df = pd.read_csv("symbol.csv")
    # symbol_list = symbols_df["symbol"].to_list()[:25]
    symbol_list = [
        # "BTCUSDT",
        "ETHUSDT",
        # "BNBUSDT",
    ]
    for symbol in symbol_list:
        endTime = 0
        i = 0
        try:
            df = pd.read_csv(f"{output_path}/{symbol}.csv")
            startTime = df["T"].max() + 1
        except:
            # startTime = None
            startTime = 1759881600000
            df = pd.DataFrame()
        while endTime < current_ts_ms:
            try:

                klines, startTime, endTime = call_orderbookticker_api(symbol = symbol,
                                        startTime = startTime)

                df = pd.concat([df, klines]).reset_index(drop = True)
                if i%100 == 0:
                    df.to_csv(f"{output_path}/{symbol}.csv", index = False)
                i+=1
            except:
                time.sleep(1)
        df.to_csv(f"{output_path}/{symbol}.csv", index = False)

    print()
if __name__ == "__main__":
    get_orderbookticker_data()

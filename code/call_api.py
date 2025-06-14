import requests
import json
from datetime import datetime
import pandas as pd
output_path = "/Users/hoapham/Documents/Learning/thesis/data/Binance"

def call_api(symbol: str = "BTCUSDT", interval = "1m", limit = 1000, endTime = None):
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

if __name__ == "__main__":
    symbols_df = pd.read_csv("symbol.csv")
    symbol_list = symbols_df["symbol"].to_list()[:25]
    for symbol in symbol_list:
        endTime = None
        i = 0
        df = pd.DataFrame()
        while i < 50:
            klines, endTime = call_api(symbol = symbol,
                                       endTime = endTime)
            df = pd.concat([df, klines]).reset_index(drop = True)
            i+=1
        df.to_csv(f"{output_path}/{symbol}.csv")
    print()
    

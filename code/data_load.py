import pickle
import datetime
import pandas as pd
import os
import yfinance as yf
from pathlib import Path

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")


def load_data(folder="tick", type = "pkl"):
    if folder == "":
        file_path = os.path.join(BASE_DIR, f"data")
        folder_path = Path(file_path)
        files = ["vn30_index.pkl"]
    else:
        file_path = os.path.join(BASE_DIR, f"data/{folder}")
        folder_path = Path(file_path)
        files = [f.name for f in folder_path.iterdir() if f.is_file()]
    if type == "pkl":
        data_dict = {}
        for filename in files:
            name, ext = os.path.splitext(filename)
            with open(f"{folder_path}/{filename}", "rb") as f:
                data_dict[name] = pickle.load(f)
        return data_dict
    elif type == "csv":
        data_dict = {}
        for filename in files:
            name, ext = os.path.splitext(filename)
            df = pd.read_csv(f"{folder_path}/{filename}")
            df = df[["0", "4", "5"]]
            # convert df["0"] to datetime
            df["0"] = df["0"]/1000
            df["0"] = df["0"].apply(lambda x: datetime.datetime.fromtimestamp(x))
            df.rename(columns = {"0": "Date", "4": "Gia KL", "5": "KL"}, inplace = True)
            data_dict[name] = df
        return data_dict
        

def get_data_from_yfinance():

    # Lấy dữ liệu của S&P 500
    sp500 = yf.Ticker("^GSPC")

    # Xem thông tin chung
    print(sp500.info)

    # Lấy dữ liệu lịch sử (ví dụ: 1 năm)
    data = sp500.history(period="1y")
    print(data.head())

    # Lấy giá đóng cửa gần nhất
    latest_close = data["Close"].iloc[-1]
    print("Giá đóng cửa gần nhất:", latest_close)


if __name__ == "__main__":
    # get_data_from_yfinance()
    data = load_data(folder="Binance", type = "csv")

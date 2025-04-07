import pickle
import os
import yfinance as yf
from pathlib import Path

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")


def load_data(folder="tick"):
    if folder == "":
        file_path = os.path.join(BASE_DIR, f"data")
        folder_path = Path(file_path)
        files = ["vn30_index.pkl"]
    else:
        file_path = os.path.join(BASE_DIR, f"data/{folder}")
        folder_path = Path(file_path)
        files = [f.name for f in folder_path.iterdir() if f.is_file()]
    data_dict = {}
    for filename in files:
        name, ext = os.path.splitext(filename)
        with open(f"{folder_path}/{filename}", "rb") as f:
            data_dict[name] = pickle.load(f)
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
    get_data_from_yfinance()
    load_data(folder="")

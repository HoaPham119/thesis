import pandas as pd
from func import calc_vpin
import numpy as np


def transform_avg_volume_hourly(data_dict, key1: str = "STB", key2: str = "SAB"):
    # Lấy data STB và SAB
    STB = data_dict[key1]
    SAB = data_dict[key2]
    # Tính trung bình khối lượng giao dịch theo giờ
    STB["Hour"] = STB["Date"].dt.hour
    SAB["Hour"] = SAB["Date"].dt.hour
    STB_avg_hourly = STB.groupby("Hour")["KL"].mean().reset_index()
    SAB_avg_hourly = SAB.groupby("Hour")["KL"].mean().reset_index()
    return STB_avg_hourly, SAB_avg_hourly


def transform_buy_sell_volume(
    data_dict,
    key1: str = "STB",
):
    # Lấy dữ liệu từ dictionary, sử dụng keys
    STB = data_dict[key1]

    # Sắp xếp data từ cũ nhất đến mới nhất
    STB = STB.sort_values(by="Date", ascending=True)  # Sắp xếp từ cũ -> mới

    # Làm tròn cột Date theo phút
    STB["Date"] = STB["Date"].dt.floor("T")

    # Phân loại dữ liệu theo mua và bán - Drop nan
    STB_sell = STB[
        [
            "Date",
            "Ban Gia 1",
            "Ban Gia 2",
            "Ban Gia 3",
            "Ban KL 1",
            "Ban KL 2",
            "Ban KL 3",
        ]
    ].dropna()
    STB_buy = STB[
        [
            "Date",
            "Mua Gia 1",
            "Mua Gia 2",
            "Mua Gia 3",
            "Mua KL 1",
            "Mua KL 2",
            "Mua KL 3",
        ]
    ].dropna()

    # Tạo cột KL để Tính tổng giá trị mua và bán trong từng dòng
    STB_sell["KL_ban"] = (
        STB_sell["Ban KL 1"] + STB_sell["Ban KL 2"] + STB_sell["Ban KL 3"]
    )
    STB_buy["KL_mua"] = STB_buy["Mua KL 1"] + STB_buy["Mua KL 2"] + STB_buy["Mua KL 3"]

    # Groupby và tính sum theo thời gian bằng phút trước khi merge
    STB_sell = STB_sell.groupby("Date", as_index=False).agg({
        "KL_ban": "sum",
        "Ban Gia 1": "mean",
        "Ban Gia 2": "mean",
        "Ban Gia 3": "mean",
    })
    STB_buy = STB_buy.groupby("Date", as_index=False).agg({
        "KL_mua": "sum",
        "Mua Gia 1": "mean",
        "Mua Gia 2": "mean",
        "Mua Gia 3": "mean",
    })

    # Merge lại data theo ngày làm tròn đến phút (cột Date) - Điền những giá trị không có = 0:
    STB = pd.merge(STB_sell, STB_buy, on="Date", how="outer").fillna(0)

    # Tính giá bán trung bình
    STB["Gia_Ban"] = (STB["Ban Gia 1"] + STB["Ban Gia 2"] + STB["Ban Gia 3"])/3
    STB["Gia_Mua"] = (STB["Mua Gia 1"] + STB["Mua Gia 2"] + STB["Mua Gia 3"])/3

    # Chỉ giữ lại các cột: Date, KL_mua, KL_ban
    STB = STB[["Date", "Gia_Ban", "Gia_Mua", "KL_ban", "KL_mua"]]
    STB["KL"] = STB["KL_mua"] + STB["KL_ban"]
    
    # ## Dùng để 
    # STB_sample = STB[["Date", "Gia_Ban", "KL"]].copy()
    # STB_sample.rename(columns = {"Date":"Datetime", "Gia_Ban": "PRICE", "KL":"SIZE"}, inplace = True)
    # STB_sample.set_index("Datetime", inplace = True)
    # # Chỉ lấy giá trị đến phút
    return STB#, STB_sample



def calculate_gap_vpin_time(
    df: pd.DataFrame,  # Đây là dataframe đã chia bucket_number
    start_bucket: int,
):
    """
    Vì thời gian giao dịch giữa các phiên có nghỉ (nghỉ trưa và nghỉ tối)
    Nên chúng ta phải xử lý vấn đề này
    """
    # Lấy dữ liệu
    # Lấy dòng đầu tiên của start_bucket và dòng đầu tiên của bucket tiếp theo
    df_first_rows = (
        df[df["bucket_number"].isin([start_bucket, start_bucket + 1])]
        .groupby("bucket_number")
        .first()
        .reset_index()
    )
    start_bucket_time = df_first_rows["Date"][0]
    end_bucket_time = df_first_rows["Date"][1]
    gap_time = (end_bucket_time - start_bucket_time).total_seconds()

    # # TODO: Cần xem lại phần logic này, logic đoạn này chưa được chặt chẽ
    # # Sug: Cần xét cùng ngày hoặc khác ngày...
    # # Giờ nghỉ cuối tuần, nghỉ lễ, nghỉ tết...

    # if start_bucket_time.day == end_bucket_time.day:
    if start_bucket_time.hour < 11 or (
        start_bucket_time == 11 and start_bucket_time.minute < 30
    ):
        if end_bucket_time.hour >= 13:  # Trừ đi 1.5 tiếng nghỉ trưa
            gap_time = gap_time - (1.5 * 60 * 60)

    elif end_bucket_time.hour < start_bucket_time.hour:
        # if end_bucket_time.hour < 14:  # Đã qua ngày hôm sau hoặc từ thứ 6 đến thứ 2 nên trừ đi 18 tiếng hoặc là 66 tiếng
            if (end_bucket_time - start_bucket_time).days==2: 
                gap_time = gap_time - (66 * 60 * 60)
            else:
                gap_time = gap_time - (18 * 60 * 60)
    return gap_time, start_bucket_time, end_bucket_time


def calcualate_gap_time_faction_of_the_day(
    df: pd.DataFrame,
    total_time_of_the_day: int = 16200,
):
    """
    df là df sau khi tính vpin và gap time
    total_time_of_the_day được tính theo giây là thời gian giao dịch của sàn chứng khoán ở Việt Nam:
    - Sáng: Từ 9h00 đến 11h30,
    - Chiều từ 1300 đến 15h00
    Tổng là 270 phút = 16200 giây
    """
    df["gap_time_faction_of_the_day"] = df["gap_time"] / total_time_of_the_day
    return df


if __name__ == "__main__":
    from data_load import load_data

    # Khai báo biến
    df={}; sec_trades = {}
    sym = ['STB', 'SAB','MWG', 'VCB','TCB']
    # # # Load data
    # data_orderbook = load_data(folder="orderbook")

    ## Load data
    data_tick = load_data(folder="tick")

    # Transform data
    for s in sym:
        data = data_tick[s].copy()
        data.rename(columns = {"Gia KL": "PRICE", "KL": "SIZE"}, inplace = True)
        data.set_index("Date", inplace = True)
        data = data.resample("T").agg({
                'SIZE': 'sum',  # Cột volume tính tổng
                'PRICE': 'mean'    # Cột price tính trung bình
            })
        sec_trades[s] = data
    
    # Cal vpin
    volume = {}
    for key, val in sec_trades.items():
        volume[key] = int(val['SIZE'].resample("D").sum().mean())
        print()
        
    for s in sym:
        print('Calculating VPIN')
        df[s] = calc_vpin(sec_trades[s],volume[s],50)
        print(s+' '+str(df[s].shape))

    ## 
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


    
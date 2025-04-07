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


def transform_buy_sell_volume(data_dict, key1: str = "STB"):
    STB = data_dict[key1].sort_values(by="Date").copy()
    STB["Date"] = STB["Date"].dt.floor("T")

    def process_side(STB, price_cols, volume_cols, volume_col_name):
        return (STB[["Date"] + price_cols + volume_cols]
                .dropna()
                .assign(**{volume_col_name: STB[volume_cols].sum(axis=1)})
                .groupby("Date", as_index=False)
                .agg({volume_col_name: "sum", **{col: "mean" for col in price_cols}}))

    STB_sell = process_side(STB, ["Ban Gia 1", "Ban Gia 2", "Ban Gia 3"], ["Ban KL 1", "Ban KL 2", "Ban KL 3"], "KL_ban")
    STB_buy = process_side(STB, ["Mua Gia 1", "Mua Gia 2", "Mua Gia 3"], ["Mua KL 1", "Mua KL 2", "Mua KL 3"], "KL_mua")

    STB = (pd.merge(STB_sell, STB_buy, on="Date", how="outer")
                .fillna(0)
                .assign(Gia_Ban=lambda x: x[["Ban Gia 1", "Ban Gia 2", "Ban Gia 3"]].mean(axis=1),
                        Gia_Mua=lambda x: x[["Mua Gia 1", "Mua Gia 2", "Mua Gia 3"]].mean(axis=1),
                        KL=lambda x: x["KL_mua"] + x["KL_ban"]))
    
    return STB[["Date", "Gia_Ban", "Gia_Mua", "KL_ban", "KL_mua", "KL"]]




def calculate_gap_vpin_time(
    df: pd.DataFrame,  # DataFrame chứa bucket_number
    start_bucket: int,
):
    """
    Tính khoảng cách thời gian giữa hai bucket, có xét đến nghỉ trưa, nghỉ đêm, cuối tuần, và nghỉ lễ.
    """
    # Lọc dữ liệu bucket hiện tại và bucket tiếp theo
    df_first_rows = (
        df[df["bucket_number"].isin([start_bucket, start_bucket + 1])]
        .groupby("bucket_number")
        .first()
        .reset_index()
    )

    # Đảm bảo có đủ dữ liệu để tính toán
    if len(df_first_rows) < 2:
        return None, None, None  # Trả về None nếu thiếu dữ liệu

    start_bucket_time = df_first_rows["Date"].iloc[0]
    end_bucket_time = df_first_rows["Date"].iloc[1]
    gap_time = (end_bucket_time - start_bucket_time).total_seconds()

    # Xác định ngày trong tuần
    start_weekday = start_bucket_time.weekday()  # Thứ 2 -> 0, Chủ Nhật -> 6
    end_weekday = end_bucket_time.weekday()

    # Kiểm tra xem có qua nghỉ trưa không (11:30 - 13:00)
    if start_bucket_time.hour < 11 or (start_bucket_time.hour == 11 and start_bucket_time.minute < 30):
        if end_bucket_time.hour >= 13:  
            gap_time -= (1.5 * 60 * 60)  # Trừ đi 1.5 tiếng nghỉ trưa

    # Xử lý khoảng nghỉ qua đêm hoặc cuối tuần
    if end_bucket_time.date() > start_bucket_time.date():
        # Trường hợp qua ngày thường (thứ 2 - thứ 6)
        if start_weekday < 4 or (start_weekday == 4 and end_weekday == 5):
            gap_time -= (18 * 60 * 60)  # Trừ đi 18 tiếng nghỉ qua đêm
        # Trường hợp qua cuối tuần (thứ 6 -> thứ 2)
        elif start_weekday == 4 and end_weekday == 0:
            gap_time -= (66 * 60 * 60)  # Trừ đi 66 tiếng nghỉ cuối tuần

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


    
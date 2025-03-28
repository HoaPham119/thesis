import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler


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

    # Làm tròn cột Date theo giây
    STB["Date"] = STB["Date"].dt.floor("s")

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

    # Groupby và tính sum theo thời gian bằng giây trước khi merge
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

    # Merge lại data theo ngày làm tròn đến giây (cột Date) - Điền những giá trị không có = 0:
    STB = pd.merge(STB_sell, STB_buy, on="Date", how="outer").fillna(0)

    # Tính giá bán trung bình
    STB["Gia_Ban"] = (STB["Ban Gia 1"] + STB["Ban Gia 2"] + STB["Ban Gia 3"])/3
    STB["Gia_Mua"] = (STB["Mua Gia 1"] + STB["Mua Gia 2"] + STB["Mua Gia 3"])/3

    # Chỉ giữ lại các cột: Date, KL_mua, KL_ban
    STB = STB[["Date", "Gia_Ban", "Gia_Mua", "KL_ban", "KL_mua"]]
    STB["KL"] = STB["KL_mua"] + STB["KL_ban"]
    return STB


def calculate_V(
    df: pd.DataFrame,
    n: int = 50,
):
    STB = df.copy()
    # Tạo cột Date_only theo ngày
    STB["Date_only"] = STB["Date"].dt.strftime("%Y-%m-%d")
    """
    # trong một số trường hợp, việc thu thập dữ liệu của ngày đầu tiên không được bắt đầu lúc 0h,
    # trong trường hợp này là không được bắt đầu từ 9h00 sáng,
    # Do đó để đảm bảo tính đầy đủ của dữ liệu ta sẽ tính V từ giá trị của ngày thứ 2 của dữ liệu
    Tuy nhiên, có thể cập nhật cách tính V từ trung bình volume của toàn bộ dữ liệu hiện có: V = avg(KL)/n
    - Lưu ý Nên loại bỏ ngày đầu tiên và ngày cuối cũng
    """

    # Lấy Total của ngày thứ 2
    total_day2 = STB.groupby("Date_only", as_index=False).sum("KL").loc[1, "KL"]

    # Lấy ngày của dòng đầu tiên
    first_date = STB["Date_only"].iloc[0]
    # Lọc bỏ những dòng có ngày trùng với dòng đầu tiên
    STB = STB[STB["Date_only"] != first_date].reset_index()
    # Tính V:
    V = total_day2 / n
    return V, STB


def calculate_bucket_number(
    df: pd.DataFrame,
    V: float,
):
    df["bucket_number"] = 0
    current_sum = 0
    bucket_index = 1
    for i in range(len(df)):
        current_sum += df.loc[i, "KL"]
        df.loc[i, "bucket_number"] = bucket_index  # Gán bucket hiện tại

        if current_sum >= V:  # Khi đạt ngưỡng V, chuyển sang bucket mới
            bucket_index += 1
            current_sum = 0  # Reset tổng cho bucket tiếp theo
    return df


def calculate_vpin(df: pd.DataFrame, n: int, V: float):
    pre_vpin_df = df.groupby("bucket_number", as_index=False).sum(
        ["KL_mua", "KL_ban", "KL"]
    )
    # Kiểm tra nếu số bucket nhỏ hơn n (để tránh lỗi truy cập ngoài phạm vi)
    if len(pre_vpin_df) < n:
        raise ValueError("Số lượng bucket nhỏ hơn n, không thể tính VPIN.")
    start_bucket = 1
    end_bucket = start_bucket + n - 1
    vpin_value = []
    max_bucket = max(pre_vpin_df["bucket_number"])
    while end_bucket <= max_bucket:
        window = pre_vpin_df[
            pre_vpin_df["bucket_number"].between(start_bucket, end_bucket)
        ]
        abs_diff = (window["KL_ban"] - window["KL_mua"]).abs().sum()
        vpin = abs_diff / (n * V)  # Tính VPIN

        # Gap time cho vpin từ lúc bắt đầu vpin này đến khi bắt đầu vpin tiếp theo
        gap_time, start_bucket_time, end_bucket_time = calculate_gap_vpin_time(df, start_bucket)
        vpin_value.append(
            {
                "start_bucket_time": start_bucket_time,
                "end_bucket_time": end_bucket_time,
                "vpin": vpin,
                "gap_time": gap_time,
                "start_bucket": start_bucket,
                "end_bucket": end_bucket,
            }
        )
        start_bucket += 1
        end_bucket += 1

    # Chấp nhận sai số 5% - Nếu bucket cuối cùng chưa cập nhật đủ thì loại bỏ:
    if pre_vpin_df["KL"].to_list()[-1] < 0.95 * V:
        vpin_value = vpin_value[0:-2]
    vpin_df = pd.DataFrame(vpin_value)
    return vpin_df


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

def transform_price_data(data_dict: dict,
                         key1: str="ACB"):
    data = transform_buy_sell_volume(data_dict, key1)
    V, STB = calculate_V(data, 50)
    STB_day = calculate_bucket_number(STB, V)
    # Thay thế giá trị 0 trong cột Gia_Ban bằng số khác 0 gần nhất
    STB_day["Gia_Ban"] = STB_day["Gia_Ban"].replace(0, pd.NA).ffill().bfill()
    # Nếu giờ lớn hơn hoặc bằng 13 thì ship về 1.5 tiếng (Để chia đều theo thời gian)
    STB_day["Date"] = STB_day["Date"].apply(lambda _time: _time if _time.hour < 13 else (_time - timedelta(minutes = 90)))
    # Xác định những ngày có trong df
    day_list = STB_day["Date_only"].unique()

    # Tính 50 mốc thời gian chia đều
    # Mỗi ngày chia 50 khoảng
    num_points = 50
    # Luư thời gian trade khớp với thời gian chia đều
    time_trades = []
    for day in day_list:
        # Ngày đang xét
        df_tam = STB[STB["Date_only"] == day].reset_index()
        nums_hours = df_tam["Date"].dt.hour.nunique()
        # Nếu dữ liệu trong ngày đó không đủ thì có thể bỏ qua
        if nums_hours < 4:
            continue
        # Tính bắt đầu từ 9h30 đến 1h00 (sau khi đã ship về 1.5 tiếng)
        start_time = df_tam["Date"][0].replace(hour = 9, minute=30, second=0, microsecond=0)
        end_time = df_tam["Date"][0].replace(hour = 13, minute=0, second=0, microsecond=0)
        # Giờ thời gian chia đều trong ngày
        time_step = [start_time + timedelta(seconds=i * (end_time - start_time).total_seconds() / (num_points - 1)) for i in range(num_points)]
        time_trade = []
        for _tg_deu in time_step:
            tam = 10000000
            for _tg_trade in df_tam["Date"]:
                gap = abs((_tg_trade - _tg_deu).total_seconds())
                if gap <= tam:
                    min_time = _tg_trade
                    tam = gap
            time_trade.append(min_time)
        time_trades.extend(time_trade)
    # CHọn các dòng có dữ liệu theo thời gian đã được chia đều
    STB_time = STB_day[STB_day["Date"].isin(time_trades)].reset_index(drop = True)
    STB_time_price = STB_time["Gia_Ban"]
    # Chọn các dòng bắt đầu của mỗi bucket
    STB_bucket = STB_day.groupby("bucket_number").first().reset_index()
    STB_bucket_price = STB_bucket["Gia_Ban"]
    # Tạo df mới để scale dữ liệu về cùng range
    price_df = pd.DataFrame()
    price_df["STB_time_price"] = STB_time_price
    price_df["STB_bucket_price"] = STB_bucket_price
    # Scale dữ liệu
    scaler = MinMaxScaler()
    price_df_scaled = pd.DataFrame(scaler.fit_transform(price_df), columns = ["time", "bucket"])
    # price_df_scaled.head()
    return price_df_scaled

if __name__ == "__main__":

    from data_load import load_data
    # Load data
    data_orderbook = load_data(folder="orderbook")
    transform_price_data("ACB")

    
    # Chọn ra 1 loại cổ phiêú
    STB = transform_buy_sell_volume(data_orderbook)
    # Tính V và loại bỏ ngày đầu tiên trong dữ liệu
    V, STB = calculate_V(STB)
    # Chia bucket
    df = calculate_bucket_number(STB, V)
    # Chuẩn bị để tính Vpin
    vpin_df = calculate_vpin(df, n=50, V=V)
    # Tính Time gap faction of the day
    vpin_df = calcualate_gap_time_faction_of_the_day(vpin_df)

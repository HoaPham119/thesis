import pandas as pd

def transform_avg_volume_hourly(data_dict,
                                key1: str = "STB",
                                key2: str = "SAB"
                                ):
    # Lấy data STB và SAB
    STB = data_dict[key1]
    SAB = data_dict[key2]
    # Tính trung bình khối lượng giao dịch theo giờ
    STB["Hour"] = STB["Date"].dt.hour
    SAB["Hour"] = SAB["Date"].dt.hour
    STB_avg_hourly = STB.groupby("Hour")["KL"].mean().reset_index()
    SAB_avg_hourly = SAB.groupby("Hour")["KL"].mean().reset_index()
    return STB_avg_hourly, SAB_avg_hourly

def transform_buy_sell_volume(data_dict,
                            key1: str = "STB",
                            ):
    # Lấy dữ liệu từ dictionary, sử dụng keys
    STB = data_dict[key1]

    # Sắp xếp data từ cũ nhất đến mới nhất
    STB = STB.sort_values(by='Date', ascending=True)  # Sắp xếp từ cũ -> mới

    # Làm tròn cột Date theo giây
    STB['Date'] = STB['Date'].dt.floor('s')

    # Phân loại dữ liệu theo mua và bán - Drop nan
    STB_sell = STB[['Date', 'Ban Gia 1', 'Ban Gia 2', 'Ban Gia 3', 'Ban KL 1',
    'Ban KL 2', 'Ban KL 3']].dropna()
    STB_buy = STB[['Date', 'Mua Gia 1', 'Mua Gia 2', 'Mua Gia 3',
        'Mua KL 1', 'Mua KL 2', 'Mua KL 3']].dropna()
    
    # Tạo cột KL để Tính tổng giá trị mua và bán trong từng dòng
    STB_sell["KL_ban"] = STB_sell["Ban KL 1"]+STB_sell["Ban KL 2"]+STB_sell["Ban KL 3"]
    STB_buy["KL_mua"] = STB_buy["Mua KL 1"]+STB_buy["Mua KL 2"]+STB_buy["Mua KL 3"]

    # Groupby và tính sum theo thời gian bằng giây trước khi merge
    STB_sell = STB_sell.groupby('Date', as_index=False).sum("KL_ban")
    STB_buy = STB_buy.groupby('Date', as_index=False).sum("KL_mua")

    # Merge lại data theo ngày làm tròn đến giây (cột Date) - Điền những giá trị không có = 0:
    STB = pd.merge(STB_sell, STB_buy, on='Date', how='outer').fillna(0)

    # Chỉ giữ lại các cột: Date, KL_mua, KL_ban
    STB = STB[["Date", "KL_mua", "KL_ban"]]
    STB["KL"] = STB["KL_mua"] + STB["KL_ban"]
    return STB

def calculate_V(df: pd.DataFrame,
                n: int = 50,
                ):
    STB = df.copy()
    # Tạo cột Date_only theo ngày
    STB['Date_only'] = STB['Date'].dt.strftime('%Y-%m-%d')

    # trong một số trường hợp, việc thu thập dữ liệu của ngày đầu tiên không được bắt đầu lúc 0h,
    # trong trường hợp này là không được bắt đầu từ 9h00 sáng,
    # Do đó để đảm bảo tính đầy đủ của dữ liệu ta sẽ tính V từ giá trị của ngày thứ 2 của dữ liệu
    
    # Lấy Total của ngày thứ 2
    total_day2 = STB.groupby("Date_only", as_index=False).sum("KL").loc[1,"KL"]

    # Lấy ngày của dòng đầu tiên
    first_date = STB['Date_only'].iloc[0]
    # Lọc bỏ những dòng có ngày trùng với dòng đầu tiên
    STB = STB[STB['Date_only'] != first_date]
    # Tính V:
    V = total_day2/n
    return V

def calculate_bucket_number(df: pd.DataFrame,
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
    
def calculate_vpin(df: pd.DataFrame,
                   n: int,
                   V: float
                   ):
    pre_vpin_df = df.groupby("bucket_number", as_index=False).sum(["KL_mua", "KL_ban", "KL"])
    # Kiểm tra nếu số bucket nhỏ hơn n (để tránh lỗi truy cập ngoài phạm vi)
    if len(pre_vpin_df) < n:
        raise ValueError("Số lượng bucket nhỏ hơn n, không thể tính VPIN.")
    start_bucket = 1
    end_bucket = 50
    vpin_value = []
    max_bucket = max(pre_vpin_df["bucket_number"])
    while end_bucket <= max_bucket:
        window = pre_vpin_df[pre_vpin_df["bucket_number"].between(start_bucket, end_bucket)]
        abs_diff = (window["KL_ban"] - window["KL_mua"]).abs().sum()
        VPIN = abs_diff / (n * V)  # Tính VPIN
        vpin_value.append({
            "VPIN": VPIN,
            "start_bucket": start_bucket,
            "end_bucket": end_bucket
        })
        start_bucket += 1
        end_bucket += 1
    # Chấp nhận sai số 5% - Nếu bucket cuối cùng chưa cập nhật đủ thì loại bỏ:
    if pre_vpin_df["KL"].to_list()[-1] <0.95*V:
        vpin_value = vpin_value[0:-2]
    vpin_df = pd.DataFrame(vpin_value)
    return vpin_df

if __name__ == "__main__":
    from data_load import load_data
    # Load data
    data_orderbook = load_data(folder= "orderbook")
    # Chọn ra 1 loại cổ phiêú
    STB = transform_buy_sell_volume(data_orderbook)
    # Tính V
    V = calculate_V(STB)
    # Chia bucket
    df = calculate_bucket_number(STB, V)
    # Chuẩn bị để tính Vpin
    df = calculate_vpin(df,
                        n = 50,
                        V = V)

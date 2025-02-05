

def transform_avg_volume_hourly(data_dict,
                                key1: str = "STB",
                                key2: str = "SAB"):
    # Lấy data STB và SAB
    STB = data_dict[key1]
    SAB = data_dict[key2]
    # Tính trung bình khối lượng giao dịch theo giờ
    STB["Hour"] = STB["Date"].dt.hour
    SAB["Hour"] = SAB["Date"].dt.hour
    STB_avg_hourly = STB.groupby("Hour")["KL"].mean().reset_index()
    SAB_avg_hourly = SAB.groupby("Hour")["KL"].mean().reset_index()
    return STB_avg_hourly, SAB_avg_hourly
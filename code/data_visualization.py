import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Trong bài báo, phần này là hình minh hoạ cho E-mini S&P 500 futures  và Euro/U.S. dollar (EC1) futures
def visualize_volume_avg_hour(data1: pd.DataFrame, data2: pd.DataFrame):
    STB_avg_hourly = data1.copy()
    SAB_avg_hourly = data2.copy()
    # Tạo figure và trục chính
    fig, ax1 = plt.subplots()
    # Trục Y trái (ES1 Index)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Volume (STB)", color="blue")
    ax1.plot(
        STB_avg_hourly["Hour"],
        STB_avg_hourly["KL"],
        color="blue",
        linewidth=2,
        label="STB",
    )
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(STB_avg_hourly["Hour"])

    ax1.grid(True, linestyle="--")

    # Trục Y phải (EC1 Currency)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volume (SAB)", color="red")
    ax2.plot(
        SAB_avg_hourly["Hour"],
        SAB_avg_hourly["KL"],
        color="red",
        linewidth=2,
        label="SAB",
    )
    ax2.tick_params(axis="y", labelcolor="red")

    # Thêm chú thích
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Hiển thị biểu đồ
    plt.title("Volume of STB and SAB")
    plt.show()


# def visualize_vpin_and_gap_time(df: pd.DataFrame):
#     # Plot
#     fig, ax1 = plt.subplots(figsize=(10, 5))

#     # Left y-axis (Time gap faction of the day)
#     ax1.plot(
#         df["start_bucket_time"],
#         df["gap_time_faction_of_the_day"],
#         color="navy",
#         label="Time gap used by VPIN metric",
#     )
#     ax1.set_xlabel("Time")
#     ax1.set_ylabel("Time gap (in faction of a day)", color="navy")
#     ax1.tick_params(axis="y", labelcolor="navy")

#     # Right y-axis (VPIN)
#     ax2 = ax1.twinx()
#     ax2.plot(
#         df["start_bucket_time"],
#         df["vpin"],
#         color="red",
#         label="VPIN metric on E-mini S&P 500 futures",
#     )
#     ax2.set_ylabel("VPIN metric", color="red")
#     ax2.tick_params(axis="y", labelcolor="red")

#     # Title and legend
#     fig.suptitle("Time Gap and VPIN Metric Over Time")
#     ax1.legend(loc="upper left")
#     ax2.legend(loc="upper right")

#     plt.show()




def visualize_vpin_and_gap_time(df: pd.DataFrame):
    # Tạo figure với kích thước lớn hơn
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Định dạng trục x nếu chứa dữ liệu thời gian
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))

    # Biểu đồ đường của Time gap trên trục y1 (màu xanh navy)
    ax1.plot(
        df["start_bucket_time"],
        df["gap_time_faction_of_the_day"],
        color="navy",
        linewidth=2,
        label="Time gap used by VPIN metric",
    )
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Time gap (fraction of a day)", color="navy", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="navy")
    ax1.grid(True, linestyle="--", alpha=0.6)  # Thêm lưới nền nhẹ

    # Biểu đồ VPIN trên trục y2 (màu đỏ)
    ax2 = ax1.twinx()
    ax2.plot(
        df["start_bucket_time"],
        df["vpin"],
        color="crimson",
        linewidth=2,
        linestyle="dashed",
        label="VPIN metric on E-mini S&P 500 futures",
    )
    ax2.set_ylabel("VPIN metric", color="crimson", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Tiêu đề
    fig.suptitle("Time Gap and VPIN Metric Over Time", fontsize=14, fontweight="bold")

    # Điều chỉnh vị trí chú thích
    ax1.legend(loc="upper left", fontsize=10, frameon=True, bbox_to_anchor=(0, 1))
    ax2.legend(loc="upper right", fontsize=10, frameon=True, bbox_to_anchor=(1, 1))

    # Điều chỉnh layout
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_price(price_df_scaled: pd.DataFrame):
    data = np.random.normal(loc=0, scale=1, size=1000)
    price_time_data = price_df_scaled["time"].dropna()
    price_bucket_data = price_df_scaled["bucket"].dropna()
    plt.figure(figsize = (8,6))
    sns.kdeplot(price_time_data, color='red', fill=False, legend="time")
    sns.kdeplot(price_bucket_data, color='blue', fill=False, legend="volume")
    sns.kdeplot(data, color='black', fill=False, legend="normal dist")
    plt.show()
    

def visualize_position_wealth(pos):
    fig = go.Figure()
    # Vẽ Position (Trục trái)
    fig.add_trace(go.Scatter(
        x=pos.index, y=pos['position'],
        mode='lines+markers', name='Position',
        line=dict(color='blue'),
        connectgaps=False  # Không nối khoảng trống
    ))

    # Vẽ Wealth (Trục phải)
    fig.add_trace(go.Scatter(
        x=pos.index, y=pos['wealth'],
        mode='lines+markers', name='Wealth',
        line=dict(color='orange'),
        yaxis='y2', connectgaps=False  # Không nối khoảng trống
    ))

    # Cấu hình layout
    fig.update_layout(
        title="Position & Wealth Over Time",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Position",
            tickfont=dict(color="blue")  # Chỉnh màu chữ của tick trên trục y
        ),
        yaxis2=dict(
            title="Wealth",
            tickfont=dict(color="orange"),
            overlaying='y', side='right'
        ),
        legend=dict(x=1.1, y=1),
        template="plotly_white"
    )

    fig.show()
    



def visualize_price_wealth(pos, name):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Thêm đường giá (Price) - Trục y trái
    fig.add_trace(
        go.Scatter(x=pos.index, y=pos['price'], 
                   name="Price", line=dict(color="blue", width=2)),
        secondary_y=False,
    )

    # Thêm đường tài sản (Wealth) - Trục y phải
    fig.add_trace(
        go.Scatter(x=pos.index, y=pos[name], 
                   name="Wealth", line=dict(color="orange", width=2)),
        secondary_y=True,
    )

    # Cấu hình trục
    fig.update_layout(
        title=dict(text=name, font=dict(size=16, family="Arial", color="black")),
        xaxis=dict(title="Time"),
        yaxis=dict(title=dict(text="Price", font=dict(color="blue")), tickfont=dict(color="blue")),
        yaxis2=dict(title=dict(text="Wealth", font=dict(color="orange")), tickfont=dict(color="orange"), overlaying='y', side='right'),
        legend=dict(x=1.1, y=1),
        template="plotly_white"
    )

    # Hiển thị biểu đồ
    fig.show()


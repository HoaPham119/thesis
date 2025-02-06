import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


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

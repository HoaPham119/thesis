import matplotlib.pyplot as plt

# Tạo figure và trục chính
def visualize_volume_avg_hour(data1, data2):
    STB_avg_hourly = data1.copy()
    SAB_avg_hourly = data2.copy()
    fig, ax1 = plt.subplots()
    # Trục Y trái (ES1 Index)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Volume (STB)", color='blue')
    ax1.plot(STB_avg_hourly["Hour"], STB_avg_hourly["KL"], color='blue', linewidth=2, label="STB")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00", "00:00"])
    ax1.grid(True, linestyle='--')

    # Trục Y phải (EC1 Currency)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volume (SAB)", color='red')
    ax2.plot(SAB_avg_hourly["Hour"], SAB_avg_hourly["KL"], color='red', linewidth=2, label="SAB")
    ax2.tick_params(axis='y', labelcolor='red')

    # Thêm chú thích
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))

    # Hiển thị biểu đồ
    plt.title("Cumulative Volume of STB and SAB")
    plt.show()

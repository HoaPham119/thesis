#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter(action="ignore")


# In[2]:


from data_load import load_data
from data_visualization import visualize_volume_avg_hour, visualize_vpin_and_gap_time
from data_processing import (
    transform_avg_volume_hourly,
    transform_buy_sell_volume,
    calculate_V,
    calculate_bucket_number,
    calculate_vpin,
    calcualate_gap_time_faction_of_the_day,
)


# In[3]:


# Load data
data_tick = load_data()
data_orderbook = load_data(folder="orderbook")
data_vn30_index = load_data(folder="")


# ## PHẦN 1: VOLUME THEO THỜI GIAN

# In[4]:


key1 = "STB"
key2 = "SAB"
STB_avg_hourly, SAB_avg_hourly = transform_avg_volume_hourly(data_tick)
SAB_avg_hourly, STB_avg_hourly


# - Vì sàn chứng khoáng Việt Nam chỉ giao dịch từ:
#     - 9h sáng 11h30 trưa
#     - Nghỉ trưa từ 11h30 đến 13h chiều
#     - Hoặt động lại từ 13h chiều đến 15h00 chiều
# - Do đó dữ liệu theo giờ chỉ có ở các mốc 9, 10, 11, 13, 14 (làm tròn lùi theo giờ)

# In[5]:


visualize_volume_avg_hour(STB_avg_hourly, SAB_avg_hourly)


# ## PHẦN 2: VPIN

# In[6]:


# Chọn ra 1 loại cổ phiêú
key1 = "STB"
STB = transform_buy_sell_volume(data_orderbook, key1=key1)
STB.head()


# In[7]:


# # Tính V
V, STB = calculate_V(STB)
V


# In[8]:


# # Chia bucket
bucket_df = calculate_bucket_number(STB, V)
bucket_df.head()


# In[9]:


# Tính vpin
vpin_df = calculate_vpin(bucket_df, n=50, V=V)
vpin_df.head()


# In[10]:


# Tính Time gap faction of the day
vpin_df = calcualate_gap_time_faction_of_the_day(vpin_df)
vpin_df.head()


# In[11]:


df = vpin_df.head(60)


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left y-axis (Time gap faction of the day)
ax1.plot(
    df["start_bucket_time"],
    df["gap_time_faction_of_the_day"],
    color="navy",
    label="Time gap used by VPIN metric",
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Time gap (in faction of a day)", color="navy")
ax1.tick_params(axis="y", labelcolor="navy")

# Right y-axis (VPIN)
ax2 = ax1.twinx()
ax2.plot(
    df["start_bucket_time"],
    df["vpin"],
    color="red",
    label="VPIN metric on E-mini S&P 500 futures",
)
ax2.set_ylabel("VPIN metric", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Title and legend
fig.suptitle("Time Gap and VPIN Metric Over Time")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()


#

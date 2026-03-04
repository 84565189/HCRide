import pandas as pd
import pickle
import os
import numpy as np

# ---------------------- 配置 (必须与 run.py 一致) ----------------------
PARQUET_PATH = os.path.join("data", "yellow_tripdata_2021-01.parquet")
OUTPUT_ORDER_PATH = os.path.join("data", "order.pkl")
OUTPUT_DRIVER_LOC_PATH = os.path.join("data", "driver_location.pkl")
OUTPUT_DRIVER_PREF_PATH = os.path.join("data", "driver_preference.pkl")

DRIVER_NUM = 150  # 对应 run.py 中的 DN
MAX_DAY = 50  # 对应 run.py 中的 maxDay
MAX_TIME = 180  # 对应 run.py 中的 maxTime
M, N = 10, 10
REGION_NUM = M * N

# 仿真环境定义的经纬度边界
MIN_LON, MAX_LON = 113.90, 114.05
MIN_LAT, MAX_LAT = 22.530, 22.670


def generate_random_loc():
    """在仿真范围内生成随机经纬度坐标"""
    lon = np.random.uniform(MIN_LON, MAX_LON)
    lat = np.random.uniform(MIN_LAT, MAX_LAT)
    return lon, lat


# ---------------------- 1. 读取并清洗数据 ----------------------
print("正在读取 Parquet 数据...")
# 采样一部分数据进行转换
df = pd.read_parquet(PARQUET_PATH).sample(n=100000, random_state=42)
df = df.dropna(subset=["PULocationID", "DOLocationID"])

# ---------------------- 2. 生成 order.pkl ----------------------
# 预期结构: {day_index: [[batch_orders]]}
# 每个 order 列表: [id, day, time, region, oriLon, oriLat, dummy, destLon, destLat, price]
print("正在生成订单数据 (order.pkl)...")
order_info = {day: [[]] for day in range(MAX_DAY)}

for idx, row in df.iterrows():
    day = idx % MAX_DAY
    time = idx % MAX_TIME
    # 将纽约的 LocationID 映射到 0-99 范围内
    region = (int(row["PULocationID"]) - 1) % REGION_NUM

    ori_lon, ori_lat = generate_random_loc()
    dest_lon, dest_lat = generate_random_loc()

    order_item = [
        idx, day, time, region,
        ori_lon, ori_lat, 0,  # index 6 是占位符
        dest_lon, dest_lat,
        float(row["total_amount"])
    ]
    order_info[day][0].append(order_item)

with open(OUTPUT_ORDER_PATH, "wb") as f:
    pickle.dump(order_info, f)

# ---------------------- 3. 生成司机数据 ----------------------
print(f"正在生成 {DRIVER_NUM} 个司机的数据...")
driver_loc_dict = {}
driver_pref_dict = {}

for i in range(DRIVER_NUM):
    # 司机初始位置: 随机分配一个区域 ID (0-99)
    driver_loc_dict[i] = np.random.randint(0, REGION_NUM)

    # 司机偏好: 必须是区域 ID 的列表
    # 这样在 envs.py 中 (x+1) 才会是整数加法，不会报错
    driver_pref_dict[i] = [np.random.randint(0, REGION_NUM) for _ in range(5)]

with open(OUTPUT_DRIVER_LOC_PATH, "wb") as f:
    pickle.dump(driver_loc_dict, f)

with open(OUTPUT_DRIVER_PREF_PATH, "wb") as f:
    pickle.dump(driver_pref_dict, f)

print("\n✅ 数据转换完成！")
print(f"1. 已修复 TypeError: driver_preference 现在存储的是整数列表。")
print(f"2. 已确保司机 ID 从 0 到 {DRIVER_NUM - 1}，修复 KeyError。")
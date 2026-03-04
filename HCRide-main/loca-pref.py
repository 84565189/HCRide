import pandas as pd
import pickle
import os
import random
import numpy as np
from datetime import datetime

# ---------------------- 1. 配置路径 ----------------------
PARQUET_PATH = os.path.join("data", "yellow_tripdata_2021-01.parquet")
OUTPUT_ORDER_PATH = os.path.join("data", "order.pkl")
OUTPUT_DRIVER_LOC_PATH = os.path.join("data", "driver_location.pkl")
OUTPUT_DRIVER_PREF_PATH = os.path.join("data", "driver_preference.pkl")

# 关键配置：生成的司机数量（可根据需求调整，比如500/1000）
DRIVER_NUM = 500
# 采样行程数（避免内存溢出）
TRIP_SAMPLE_NUM = 100000

# ---------------------- 2. 读取并清洗Parquet数据 ----------------------
print("开始读取Parquet数据...")
df = pd.read_parquet(PARQUET_PATH, engine="pyarrow").sample(n=TRIP_SAMPLE_NUM, random_state=42)

# 清洗无效数据
df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"])
df = df[df["passenger_count"] > 0]
df = df[df["trip_distance"] > 0]
# 重置索引（避免行索引混乱）
df = df.reset_index(drop=True)

print(f"清洗后有效行程数：{len(df)}")

# ---------------------- 3. 生成订单数据（order.pkl） ----------------------
print("生成订单数据...")
orders = []
for idx, row in df.iterrows():
    order = {
        "order_id": idx,  # 订单ID=行程索引
        "create_time": row["tpep_pickup_datetime"],
        "pickup_location": int(row["PULocationID"]),
        "dropoff_location": int(row["DOLocationID"]),
        "passenger_count": int(row["passenger_count"]),
        "estimated_fare": float(row["fare_amount"]),
        "trip_distance": float(row["trip_distance"])
    }
    orders.append(order)

# 保存订单pkl
with open(OUTPUT_ORDER_PATH, "wb") as f:
    pickle.dump(orders, f)
print(f"订单数据保存完成，共{len(orders)}条")

# ---------------------- 4. 生成司机池+分配行程给司机 ----------------------
print(f"生成{DRIVER_NUM}个司机的基础信息...")
# 生成司机ID列表（1~DRIVER_NUM）
driver_ids = list(range(1, DRIVER_NUM + 1))
# 为每个行程随机分配一个司机ID（核心：解决司机ID缺失问题）
df["driver_id"] = np.random.choice(driver_ids, size=len(df), replace=True)

# ---------------------- 5. 生成司机位置数据（driver_location.pkl） ----------------------
print("生成司机位置数据...")
driver_locations = []
# 为每个司机生成「最新位置」（基于该司机最后一次接单的上车位置）
for driver_id in driver_ids:
    # 筛选该司机的所有行程
    driver_trips = df[df["driver_id"] == driver_id]
    if len(driver_trips) == 0:
        # 若无行程，随机分配一个纽约区域ID（1~265）
        loc_id = random.randint(1, 265)
        timestamp = datetime.now()
    else:
        # 取最后一次接单的位置和时间
        last_trip = driver_trips.sort_values("tpep_pickup_datetime").iloc[-1]
        loc_id = int(last_trip["PULocationID"])
        timestamp = last_trip["tpep_pickup_datetime"]

    driver_location = {
        "driver_id": driver_id,
        "location_id": loc_id,
        "timestamp": timestamp
    }
    driver_locations.append(driver_location)

# 保存司机位置pkl
with open(OUTPUT_DRIVER_LOC_PATH, "wb") as f:
    pickle.dump(driver_locations, f)
print(f"司机位置数据保存完成，共{len(driver_locations)}条")

# ---------------------- 6. 生成司机偏好数据（driver_preference.pkl） ----------------------
print("生成司机偏好数据...")
driver_prefs = []
for driver_id in driver_ids:
    driver_trips = df[df["driver_id"] == driver_id]
    if len(driver_trips) == 0:
        # 若无行程，设置默认偏好
        preferred_area = random.randint(1, 265)
        preferred_distance = 5.0  # 默认偏好5公里
        min_fare = 10.0  # 默认最低接受10美元
    else:
        # 偏好区域：接单最多的区域（众数）
        preferred_area_series = driver_trips["PULocationID"].mode()
        preferred_area = int(preferred_area_series.iloc[0]) if not preferred_area_series.empty else random.randint(1,
                                                                                                                   265)
        # 偏好距离：该司机平均行程距离
        preferred_distance = float(driver_trips["trip_distance"].mean())
        # 最低接受费用：该司机接单的最低费用
        min_fare = float(driver_trips["fare_amount"].min())

    driver_pref = {
        "driver_id": driver_id,
        "preferred_area": preferred_area,
        "preferred_distance": preferred_distance,
        "min_fare": min_fare,
        # 额外补充：偏好费用（平均费用），增强仿真真实性
        "preferred_fare": float(driver_trips["fare_amount"].mean()) if len(driver_trips) > 0 else 15.0
    }
    driver_prefs.append(driver_pref)

# 保存司机偏好pkl
with open(OUTPUT_DRIVER_PREF_PATH, "wb") as f:
    pickle.dump(driver_prefs, f)
print(f"司机偏好数据保存完成，共{len(driver_prefs)}条")

print("\n✅ 所有数据转换完成！")
print(f"- order.pkl：{len(orders)}条订单")
print(f"- driver_location.pkl：{len(driver_locations)}条司机位置")
print(f"- driver_preference.pkl：{len(driver_prefs)}条司机偏好")
import pandas as pd
import pickle
import os
import random
import numpy as np
from datetime import datetime

# ---------------------- 配置 ----------------------
PARQUET_PATH = os.path.join("data", "yellow_tripdata_2021-01.parquet")
OUTPUT_ORDER_PATH = os.path.join("data", "order.pkl")
OUTPUT_DRIVER_LOC_PATH = os.path.join("data", "driver_location.pkl")
OUTPUT_DRIVER_PREF_PATH = os.path.join("data", "driver_preference.pkl")
DRIVER_NUM = 500  # 可按需调整：测试用100~500，正式用1000~2000
TRIP_SAMPLE_NUM = 100000

# ---------------------- 读取清洗数据 ----------------------
print("读取Parquet数据...")
df = pd.read_parquet(PARQUET_PATH, engine="pyarrow").sample(n=TRIP_SAMPLE_NUM, random_state=42)
df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"])
df = df[df["passenger_count"] > 0]
df = df[df["trip_distance"] > 0]
df = df.reset_index(drop=True)

# ---------------------- 生成订单数据（列表套字典，和之前一致） ----------------------
orders = []
for idx, row in df.iterrows():
    orders.append({
        "order_id": idx,
        "create_time": row["tpep_pickup_datetime"],
        "pickup_location": int(row["PULocationID"]),
        "dropoff_location": int(row["DOLocationID"]),
        "passenger_count": int(row["passenger_count"]),
        "estimated_fare": float(row["fare_amount"]),
        "trip_distance": float(row["trip_distance"])
    })
with open(OUTPUT_ORDER_PATH, "wb") as f:
    pickle.dump(orders, f)
print(f"订单数据：{len(orders)}条")

# ---------------------- 生成司机池+分配行程 ----------------------
driver_ids = list(range(1, DRIVER_NUM + 1))
df["driver_id"] = np.random.choice(driver_ids, size=len(df), replace=True)

# ---------------------- 生成司机位置（字典：{driver_id: location_id}） ----------------------
driver_loc_dict = {}  # 原有代码预期的格式：driver_id → 整数区域ID
for driver_id in driver_ids:
    driver_trips = df[df["driver_id"] == driver_id]
    if len(driver_trips) == 0:
        loc_id = random.randint(1, 265)
    else:
        last_trip = driver_trips.sort_values("tpep_pickup_datetime").iloc[-1]
        loc_id = int(last_trip["PULocationID"])
    driver_loc_dict[driver_id] = loc_id  # 仅存整数区域ID，不是字典

with open(OUTPUT_DRIVER_LOC_PATH, "wb") as f:
    pickle.dump(driver_loc_dict, f)
print(f"司机位置：{len(driver_loc_dict)}条（格式：driver_id→整数区域ID）")

# ---------------------- 生成司机偏好（字典：{driver_id: 偏好字典}） ----------------------
driver_pref_dict = {}  # 原有代码预期的格式：driver_id → 偏好字典
for driver_id in driver_ids:
    driver_trips = df[df["driver_id"] == driver_id]
    if len(driver_trips) == 0:
        pref = {
            "preferred_area": random.randint(1, 265),
            "preferred_distance": 5.0,
            "min_fare": 10.0,
            "preferred_fare": 15.0
        }
    else:
        pref = {
            "preferred_area": int(driver_trips["PULocationID"].mode().iloc[0]),
            "preferred_distance": float(driver_trips["trip_distance"].mean()),
            "min_fare": float(driver_trips["fare_amount"].min()),
            "preferred_fare": float(driver_trips["fare_amount"].mean())
        }
    driver_pref_dict[driver_id] = pref

with open(OUTPUT_DRIVER_PREF_PATH, "wb") as f:
    pickle.dump(driver_pref_dict, f)
print(f"司机偏好：{len(driver_pref_dict)}条（格式：driver_id→偏好字典）")

print("\n✅ 数据生成完成！格式完全适配HCRide原有代码")
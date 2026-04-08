import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
import math
import statistics
from simulator.envs import *
from algorithm.Habic import *
from algorithm.AC import *
import pickle
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


seed = 100
set_seed(seed)

dayIndex = 50
maxTime = 180
maxDay = 81

minlon = 113.90
maxlon = 114.05
minlat = 22.530
maxlat = 22.670

locRange = [minlon, maxlon, minlat, maxlat]

M = 10
N = 10

stateDim = 9
actionDim = 15
actorLr = 0.001
criticLr = 0.01
lagLr = 1e-3
limit = 5
lagrange = 1
epochs = 5
eps = 0.2
gamma = 0.95
memorySize = 10000
batchSize = 1000

testNum = 1

result_dir = f'result/Test{testNum}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

orderInfo = pd.read_pickle('data/order.pkl')
driverPreInit = pd.read_pickle('data/driver_preference.pkl')
driverLocInit = pd.read_pickle('data/driver_location.pkl')

driverNum = 150

env = Env(driverPreInit, driverLocInit, orderInfo, locRange, driverNum, M, N, maxDay, maxTime)
agent = torch.load(f'result/Test1/agent.pth')
replayBuffer = ReplayBuffer(memorySize, batchSize)

env.set_driver_info(driverPreInit, driverLocInit)
env.set_region_info()

updateRound = 0

wholewtList = []
wholeRewardList = []
wholeInterFairnessList = []
wholeIntraFairnessList = []
wholeIndividualwtList = []
wholePVRList = []

while dayIndex < maxDay:
    print(f"Day: {dayIndex}")
    env.set_day_info(dayIndex)
    env.reset_clean()

    T = 0
    dDict = {}

    daily_violated_orders = 0
    daily_total_orders = 0

    while T < maxTime:
        for order in env.dayOrder[T]:
            driverList = env.driver_collect(order)
            if driverList == 0:
                continue
            candidateList = env.generate_candidate_set(order, driverList)
            driverStateArray = env.driver_state_calculate(candidateList)
            actionStateArray = env.action_state_calculate(candidateList, order)

            stateArray = driverStateArray
            matchingStateArray = np.hstack((stateArray, actionStateArray))

            action = agent.take_action_test(matchingStateArray, dayIndex)

            rightDriver = candidateList[action]
            rightRegion = env.regionList[order.oriRegion]
            state = stateArray[action]
            matchingState = matchingStateArray
            wt = actionStateArray[action, 9]
            dt = actionStateArray[action, 8]
            trs = int(math.ceil(wt + dt))
            env.add_global_wt(wt)
            meanwt = env.regionList[order.oriRegion].meanwt
            globalmeanwt = env.cal_global_mean_wt()
            minwt = env.cal_global_max_wt()
            cost = env.cal_cost(order, candidateList[action])

            # 修正：传入 region_id 和当前时间 T
            reward = env.cal_reward(wt, meanwt, trs, cost, order.oriRegion, T)

            if cost > 0:
                daily_violated_orders += 1
            daily_total_orders += 1

            d = DispatchSolution()
            d.add_driver_ID(rightDriver.driverID)
            d.add_state(state)
            d.add_matchingState(matchingState)
            d.add_trs(trs)
            d.add_action(action)
            d.add_reward(reward)
            d.add_cost(cost)
            dDict[rightDriver] = d

            rightDriver.accept_order(trs, order.destLoc, reward, wt, cost)
            rightRegion.accept_order(reward, wt)

            # 将当前订单的等待时间记录到时间片列表中，供后续方差计算
            env.region_time_slot_wt[order.oriRegion][T].append(wt)

            wholeIndividualwtList.append(wt)

        env.step(dDict, replayBuffer)
        T += 1

    regionDaywt = []
    regionDayVarwt = []
    regionDayOrder = []
    regionDayReward = []
    for region in env.regionList:
        regionDaywt.append(region.dayAccwt)
        regionDayOrder.append(region.dayAccOrder)
        regionDayReward.append(region.dayAccReward)
        if len(region.daywtList) > 1:
            regionDayVarwt.append(statistics.variance(region.daywtList))
        else:
            regionDayVarwt.append(0)
    meanDaywt = round(sum(regionDaywt) / sum(regionDayOrder), 3)
    regionMeanDaywt = [x / y if y != 0 else -1 for x, y in zip(regionDaywt, regionDayOrder)]
    regionMeanDaywt = [x for x in regionMeanDaywt if x != -1]
    regionMeanDayVarwt = round(float(np.mean(np.array(regionDayVarwt))), 3)
    regionVarDayMeanwt = round(float(np.var(np.array(regionMeanDaywt))), 3)
    meanDayReward = round(sum(regionDayReward) / sum(regionDayOrder), 3)

    dayPVR = daily_violated_orders / daily_total_orders if daily_total_orders > 0 else 0

    print(f"Day {env.cityDay} mean reward: {meanDayReward}")
    print(f"Day {env.cityDay} mean waiting time: {meanDaywt}")
    print(f"Day {env.cityDay} mean inter-region fairness {regionVarDayMeanwt}")
    print(f"Day {env.cityDay} mean intra-region fairness {regionMeanDayVarwt}")
    print(f"Day {env.cityDay} mean PVR: {round(dayPVR, 4)}")
    print(" ")

    wholeRewardList.append(meanDayReward)
    wholewtList.append(meanDaywt)
    wholeInterFairnessList.append(regionVarDayMeanwt)
    wholeIntraFairnessList.append(regionMeanDayVarwt)
    wholePVRList.append(dayPVR)

    dayIndex += 1

metrics_result = {
    "mean_reward": round(statistics.mean(wholeRewardList), 3),
    "var_reward": round(statistics.variance(wholeRewardList), 3) if len(wholeRewardList) > 1 else 0.0,
    "mean_waiting_time": round(statistics.mean(wholewtList), 3),
    "var_waiting_time": round(statistics.variance(wholewtList), 3) if len(wholewtList) > 1 else 0.0,
    "mean_inter_fairness": round(statistics.mean(wholeInterFairnessList), 3),
    "var_inter_fairness": round(statistics.variance(wholeInterFairnessList), 3) if len(
        wholeInterFairnessList) > 1 else 0.0,
    "mean_intra_fairness": round(statistics.mean(wholeIntraFairnessList), 3),
    "var_intra_fairness": round(statistics.variance(wholeIntraFairnessList), 3) if len(
        wholeIntraFairnessList) > 1 else 0.0,
    "mean_pvr": round(statistics.mean(wholePVRList), 4),
    "var_pvr": round(statistics.variance(wholePVRList), 4) if len(wholePVRList) > 1 else 0.0
}

print("=" * 50)
print("全周期（Day50-Day81）指标汇总")
print("=" * 50)
print(f"平均奖励: {metrics_result['mean_reward']}, 奖励方差: {metrics_result['var_reward']}")
print(f"平均等待时间: {metrics_result['mean_waiting_time']}, 等待时间方差: {metrics_result['var_waiting_time']}")
print(
    f"平均区域间公平性: {metrics_result['mean_inter_fairness']}, 区域间公平性方差: {metrics_result['var_inter_fairness']}")
print(
    f"平均区域内公平性: {metrics_result['mean_intra_fairness']}, 区域内公平性方差: {metrics_result['var_intra_fairness']}")
print(f"平均偏好违背率(PVR): {metrics_result['mean_pvr']}, 偏好违背率方差: {metrics_result['var_pvr']}")
print("=" * 50)

txt_path = os.path.join(result_dir, "full_cycle_metrics.txt")
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("HCRide 全周期指标汇总（Day50-Day81）\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. 奖励指标\n")
    f.write(f"   平均值: {metrics_result['mean_reward']}\n")
    f.write(f"   方差: {metrics_result['var_reward']}\n\n")
    f.write(f"2. 等待时间指标\n")
    f.write(f"   平均值: {metrics_result['mean_waiting_time']}\n")
    f.write(f"   方差: {metrics_result['var_waiting_time']}\n\n")
    f.write(f"3. 区域间公平性指标\n")
    f.write(f"   平均值: {metrics_result['mean_inter_fairness']}\n")
    f.write(f"   方差: {metrics_result['var_inter_fairness']}\n\n")
    f.write(f"4. 区域内公平性指标\n")
    f.write(f"   平均值: {metrics_result['mean_intra_fairness']}\n")
    f.write(f"   方差: {metrics_result['var_intra_fairness']}\n\n")
    f.write(f"5. 偏好违背率(PVR)指标\n")
    f.write(f"   平均值: {metrics_result['mean_pvr']}\n")
    f.write(f"   方差: {metrics_result['var_pvr']}\n")
print(f"\n全周期指标已保存至: {txt_path}")

regionwt = []
regionOrder = []
regionIntraFairness = []
for region in env.regionList:
    regionwt.append(region.accwt)
    regionOrder.append(region.accOrder)
    regionIntraFairness.append(np.var(np.array(region.accwtList)))
regionMeanwt = [x / y if y != 0 else 0 for x, y in zip(regionwt, regionOrder)]

wtLoc = os.path.join(result_dir, 'global_region_wt.pkl')
fairnessLoc = os.path.join(result_dir, 'region_Intra_Fairness.pkl')
individualwtLoc = os.path.join(result_dir, 'individual_wt_list.pkl')

with open(wtLoc, 'wb') as f:
    pickle.dump(regionMeanwt, f)
with open(fairnessLoc, 'wb') as f:
    pickle.dump(regionIntraFairness, f)
with open(individualwtLoc, 'wb') as f:
    pickle.dump(wholeIndividualwtList, f)

print("所有指标文件保存完成！")
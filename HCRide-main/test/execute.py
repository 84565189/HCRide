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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


seed = 100
set_seed(seed)

dayIndex = 50  # the current day
maxTime = 180  # Max dispatch round
maxDay = 81

minlon = 113.90
maxlon = 114.05
minlat = 22.530
maxlat = 22.670

locRange = [minlon, maxlon, minlat, maxlat]

M = 10
N = 10

stateDim = 6
actionDim = 10
actorLr = 0.001
criticLr = 0.01
lagLr = 1e-3
limit = 0.15
lagrange = 1
epochs = 5
eps = 0.2
gamma = 0.95
memorySize = 10000
batchSize = 1000

#
testNum = 1

# optimization

# ① 启始位置随机


orderInfo = pd.read_pickle('data/order.pkl')

driverPreInit = pd.read_pickle('data/driver_preference.pkl')
driverLocInit = pd.read_pickle('data/driver_location.pkl')
# regionWT = pd.read_pickle('../data/regionMeanWT.pkl')

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


while dayIndex < maxDay:

    print(f"Day: {dayIndex}")

    env.set_day_info(dayIndex)
    env.reset_clean()

    T = 0
    dDict = {}
    while T < maxTime:
        for order in env.dayOrder[T]:
            driverList = env.driver_collect(order)
            if driverList == 0:
                continue
            candidateList = env.generate_candidate_set(order, driverList)
            driverStateArray = env.driver_state_calculate(candidateList)  #
            actionStateArray = env.action_state_calculate(candidateList, order)  #

            stateArray = driverStateArray
            matchingStateArray = np.hstack((stateArray, actionStateArray))  #

            action = agent.take_action_test(matchingStateArray, dayIndex)  #

            rightDriver = candidateList[action]
            rightRegion = env.regionList[order.oriRegion]
            state = stateArray[action]  #
            matchingState = matchingStateArray  #
            wt = actionStateArray[action, 9]
            dt = actionStateArray[action, 8]
            trs = int(math.ceil(wt + dt))
            env.add_global_wt(wt)
            meanwt = env.regionList[order.oriRegion].meanwt
            globalmeanwt = env.cal_global_mean_wt()
            minwt = env.cal_global_max_wt()
            cost = env.cal_cost(order, candidateList[action])
            reward = env.cal_reward(wt, meanwt, trs, cost)

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

    print(f"Day {env.cityDay} mean reward: {meanDayReward}")
    print(f"Day {env.cityDay} mean waiting time: {meanDaywt}")
    print(f"Day {env.cityDay} mean inter-region fairness {regionVarDayMeanwt}")
    print(f"Day {env.cityDay} mean intra-region fairness {regionMeanDayVarwt}")
    print(" ")

    wholeRewardList.append(meanDayReward)
    wholewtList.append(meanDaywt)
    wholeInterFairnessList.append(regionVarDayMeanwt)
    wholeIntraFairnessList.append(regionMeanDayVarwt)

    dayIndex += 1



print(f"whole mean reward: {statistics.mean(wholeRewardList)}")
print(f"whole mean waiting time: {statistics.mean(wholewtList)}")
print(f"whole inter fairness: {statistics.mean(wholeInterFairnessList)}")
print(f"whole intra fairness: {statistics.mean(wholeIntraFairnessList)}")
#
regionwt = []
regionOrder = []
regionIntraFairness = []
for region in env.regionList:
    regionwt.append(region.accwt)
    regionOrder.append(region.accOrder)
    regionIntraFairness.append(np.var(np.array(region.accwtList)))
regionMeanwt = [x/y if y != 0 else 0 for x,y in zip(regionwt, regionOrder)]
wtLoc = f'result/Test{testNum}/global_region_wt.pkl'
fairnessLoc = f'result/Test{testNum}/region_Intra_Fairness.pkl'
individualwtLoc =  f'result/Test{testNum}/individual_wt_list.pkl'
with open(wtLoc, 'wb') as f:
    pickle.dump(regionMeanwt, f)
with open(fairnessLoc,'wb') as f:
    pickle.dump(regionIntraFairness, f)
with open(individualwtLoc,'wb') as f:
    pickle.dump(wholeIndividualwtList, f)


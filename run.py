import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.envs import *
from algorithm.Habic import *
from algorithm.AC import *

import pickle


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


seed = 20
set_seed(seed)

dayIndex = 0
maxTime = 180
maxDay = 50

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
limit = 6
lagrange = 1
epochs = 5
eps = 0.2
gamma = 0.95
memorySize = 10000
batchSize = 1000

testNum = 1

orderInfo = pd.read_pickle('data/order.pkl')

driverPreInit = pd.read_pickle('data/driver_preference.pkl')
driverLocInit = pd.read_pickle('data/driver_location.pkl')

DN = 150

env = Env(driverPreInit, driverLocInit, orderInfo, locRange, DN, M, N, maxDay, maxTime)

agent = Habic(stateDim, actionDim, actorLr, criticLr, lagLr, limit, lagrange, epochs, eps, gamma, batchSize)
replayBuffer = ReplayBuffer(memorySize, batchSize)

env.set_driver_info(driverPreInit, driverLocInit)
env.set_region_info()

updateRound = 0

wholeEpisodeRewardList = []
rewardList = []
wtList = []
actorLossList = []
rewardCriticLossList = []
costCriticLossList = []
moneyList = []
orderCancellationRate = 0
pvrList = []

while dayIndex < maxDay:
    episodeList = []

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
                print('All drivers have been full!')
                continue
            candidateList = env.generate_candidate_set(order, driverList)
            driverStateArray = env.driver_state_calculate(candidateList)
            actionStateArray = env.action_state_calculate(candidateList, order)

            stateArray = driverStateArray
            matchingStateArray = np.hstack((stateArray, actionStateArray))

            action = agent.take_action(matchingStateArray, dayIndex)

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
            maxwt = env.cal_global_max_wt()
            cost = env.cal_cost(order, candidateList[action])
            # 修改奖励调用，传入 region_id 和当前时间 T
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
            # 将当前订单的等待时间记录到时间片列表中（供后续方差计算）
            env.region_time_slot_wt[order.oriRegion][T].append(wt)
            episodeList.append(reward)

        env.step(dDict, replayBuffer)
        T += 1

        if (T % 20) == 0 and dayIndex >= 2:
            updateRound += 1
            for k in range(1):
                matchingState, state, action, reward, cost, nextState = replayBuffer.sample()
                actorLoss, rewardCriticLoss, costCriticLoss = agent.update_theta(matchingState, state, action, reward,
                                                                                 cost, nextState, updateRound, k)
                agent.update_lagrange(matchingState, state, action, reward, cost, nextState, updateRound)

                # 由于已经在 Habic.py 里转为 .item()，这里 append 的是安全的标量
                actorLossList.append(actorLoss)
                rewardCriticLossList.append(rewardCriticLoss)
                costCriticLossList.append(costCriticLoss)
                wholeEpisodeRewardList.append(np.mean(np.array(episodeList)))
                episodeList = []

    regionDaywt = []
    regionDayVarwt = []
    regionDayOrder = []
    regionDayReward = []
    for region in env.regionList:
        regionDaywt.append((region.dayAccwt))
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

    print(f"Day {env.cityDay + 1} mean reward: {meanDayReward}")
    print(f"Day {env.cityDay + 1} mean waiting time: {meanDaywt}")
    print(f"Day {env.cityDay + 1} mean inter-region fairness {regionVarDayMeanwt}")
    print(f"Day {env.cityDay + 1} mean intra-region fairness {regionMeanDayVarwt}")
    print(f"Day {env.cityDay + 1} mean PVR (Preference Violation Rate): {round(dayPVR, 4)}")
    print(" ")

    rewardList.append(meanDayReward)
    wtList.append(meanDaywt)
    pvrList.append(dayPVR)

    dayIndex += 1

torch.save(agent, f'result/Test{testNum}/agent.pth')

env.plot(rewardList, actorLossList, rewardCriticLossList, testNum)
import pandas as pd
import numpy as np
import statistics
from simulator.regions import *
from simulator.orders import *
from simulator.unitity import *
from simulator.drivers import *
import random

import math
from scipy.stats import expon

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
plt.rcParams["font.family"] = "Times New Roman"

#rewardList = []
#gamma = 0.98

class Env(object):
    def __init__(self,driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime):
        self.driverPreInit = driverPreInit
        self.driverLocInit = driverLocInit
        self.orderInfo = orderInfo
        self.locRange = locRange # locRange = [minlon,maxlon,minlat,maxlat]
        self.length = M
        self.width = N
        self.maxTime = maxTime
        self.maxDay = maxDay


        self.cityTime = 0 #
        self.cityDay = 0
        self.maxCityTime = maxTime


        self.driverList = []
        self.driverDict = {}
        self.driverNum = driverNum



        self.M = M
        self.N = N
        self.regionNum = self.M * self.N
        self.regionList = [Region(i,self.regionNum) for i in range(self.regionNum)]  # 区域的结点列表
        self.regionDict = {}

        self.candidateDriverSize = 20
        self.maxDriverPreNum = 0

        # speed
        self.speed = 30

        #
        self.alpha = 1.5
        self.gamma = 0.98

        self.globalwtList = []


    def set_region_info(self):
        for i in range(self.regionNum):
            region = self.regionList[i]
            self.regionDict[i] = region
            region.set_neighbors()
            region.set_region_meanwt()

    def set_driver_info(self, driverPreInit, driverLocInit):
        for i in range(self.driverNum):
            driverID = i
            driverPre = driverPreInit[i]
            driverRegion = driverLocInit[i]
            driverLoc = generate_loc(driverRegion)
            driver = Driver(driverID, driverPre, driverLoc)
            self.driverList.append(driver)
            self.driverDict[driverID] = driver

    def set_day_info(self, dayIndex):
        self.cityDay = dayIndex
        for driver in self.driverList:
            driver.set_day_info(dayIndex)
        for region in self.regionList:
            region.set_day_info(dayIndex)

    def reset_clean(self):
        self.cityTime = 0
        self.dayOrder = []
        for driver in self.driverList:
            driver.reset_driver_info()
        for region in self.regionList:
            region.reset_region_info()
        self.boost_one_day_order()
        self.boost_step_order_info(self.cityTime)
        self.boost_step_region_info()

    def boost_step_order_info(self, T):
        stepOrderList = self.dayOrder[T]
        for order in stepOrderList:
            region = order.oriRegion
            self.regionList[region].add_order(order)


    def boost_step_region_info(self):
        for driver in self.driverList:
            if driver.state == 1:
                region = driver.region
                self.regionList[region].add_driver(driver)

    def boost_one_day_order(self):
        dayOrderList = [[] for _ in np.arange(self.maxCityTime)]
        for dayOrder in self.orderInfo[self.cityDay]:
            for order in dayOrder:
                startTime = order[2]
                # orderID,orderDay,orderMin,orderRegion,oriLon,oriLat,destLon,destLat,price
                orderRegion = self.regionList[order[3]]
                dayOrderList[startTime].append(Order(order[0], order[1], order[2], orderRegion, order[4], order[5],
                                                     order[7], order[8], order[9]))
        self.dayOrder = dayOrderList


    def add_global_wt(self,wt):
        self.globalwtList.append(wt)

    def cal_global_mean_wt(self):
        return float(np.mean(np.array(self.globalwtList)))

    def cal_global_max_wt(self):
        regionMeanWTList = []
        for region in self.regionList:
            regionMeanWTList.append(region.meanwt)
        MaxWT = max(regionMeanWTList)
        return MaxWT



    def driver_collect(self,order):
        orderRegion = order.orderRegion
        neighborLevelIndex = 3
        driverList = []
        neighborList = orderRegion.neighborLevel[neighborLevelIndex]
        for neighbor in neighborList:
            driverList.append(self.regionList[neighbor].driverList)
        driverList = [x for y in driverList for x in y]

        while len(driverList) == 0:
            neighborLevelIndex += 1
            if neighborLevelIndex == 4:
                return 0
            driverList = []
            neighborList = orderRegion.neighborLevel[neighborLevelIndex]
            for neighbor in neighborList:
                driverList.append(self.regionList[neighbor].driverList)
            driverList = [x for y in driverList for x in y]

        return driverList

    def generate_candidate_set(self,order,driverList):
        disList = []
        for i in range(len(driverList)):
            driver = driverList[i]
            dis = cal_dis(order.oriLoc,driver.loc)
            disList.append((dis,i))
        disList = sorted(disList,key = lambda x:x[0],reverse = False)
        disList = disList[:self.candidateDriverSize]
        disList = [x[1] for x in disList]
        candidateList = [driverList[x] for x in disList]

        return candidateList

    def driver_state_calculate(self,driverList):
        driverArray = np.zeros((len(driverList),self.maxDriverPreNum + 6))
        index = 0
        for driver in driverList:
            region = np.array([driver.region + 1]) / self.regionNum # 1
            regionWT = self.regionList[driver.region].meanwt
            regionWT = np.array([regionWT])
            t = np.array([driver.cityTime]) / self.maxCityTime
            lon = (driver.loc.lon - minlon) / (maxlon - minlon)
            lon = np.array([lon])
            lat = (driver.loc.lat - minlat) / (maxlat - minlat)
            lat = np.array([lat])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderDis = cal_dis(driver.loc,order.oriLoc)
                if orderDis <= nearwt:
                    nearwt = orderDis
            nearwt = np.array([nearwt])
            preRegionList = driver.preRegion
            preRegionList = [(x+1)/self.regionNum for x in preRegionList]
            driverState = np.concatenate((region,regionWT,lon,lat,nearwt,t))
            driverArray[index,:] = driverState
            index += 1
        return driverArray

    # def driver_next_state_calculate(self,driver,order,trs):
    #     region = np.array([order.destRegion + 1])
    #     t = np.array([driver.cityTime + trs])
    #     lon = round((order.destLoc.lon - minlon) / (maxlon - minlon), 5)
    #     lon = np.array([lon])
    #     lat = round((order.destLoc.lon - minlat) / (maxlat - minlat), 5)
    #     lat = np.array([lat])
    #     preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')
    #     nextDriverState = np.concatenate((region,t,lon,lat,preRegion))
    #     return nextDriverState

    def action_state_calculate(self,driverList,order):
        actionArray = np.zeros((len(driverList),10))
        oriRegion = np.array([order.oriRegion + 1]) / self.regionNum
        destRegion = np.array([order.destRegion + 1]) / self.regionNum
        oriLon = (order.oriLoc.lon - minlon) / (maxlon - minlon)
        oriLon = np.array([oriLon])
        oriLat = (order.oriLoc.lat - minlat) / (maxlat - minlat)
        oriLat = np.array([oriLat])
        destLon = (order.destLoc.lon - minlon) / (maxlon - minlon)
        destLon = np.array([destLon])
        destLat = (order.destLoc.lat - minlat) / (maxlat - minlat)
        destLat = np.array([destLat])
        oriWT = self.regionList[order.oriRegion].meanwt
        oriWT = np.array([oriWT])
        destWT = self.regionList[order.destRegion].meanwt
        destWT = np.array([destWT])
        dt = np.array([(cal_dis(order.oriLoc, order.destLoc) / self.speed) * 60])
        index = 0
        for driver in driverList:
            wt = np.array([(cal_dis(driver.loc,order.oriLoc) / self.speed) * 60])
            orderState = np.concatenate((oriRegion, destRegion, oriLon, oriLat, destLon, destLat, oriWT, destWT, dt, wt))
            actionArray[index,:] = orderState
            index += 1
        return actionArray

    def con_state_calcualte(self):
        supplyDemandList = []
        for region in self.regionList:
            supply = len(region.driverList)
            demand = len(region.orderList)
            supplyDemandList.append(supply)
            supplyDemandList.append(demand)
        return np.array(supplyDemandList)

    def cal_reward(self,wt,meanwt,trs,cost):
        reward = (-wt) + (self.alpha * (-1) * abs(wt - meanwt) / 3)
        a = (1 - self.alpha) * (-wt)
        b = self.alpha * (-1) * abs(wt - meanwt)
        return reward
        # rewardList.append(reward)
        # if len(rewardList) == 1:
        #     reward = 0
        # else:
        #     reward = reward/statistics.stdev(rewardList)
         #   rt = reward / trs
         #   gammaReward = 0
         #   for t in range(trs):
         #       gammaReward += rt * pow(self.gamma,t)

    def cal_money_reward(self,money):
        return money

    def cal_absolute_reward(self,wt,meanwt,trs,cost):
        reward = ((1 - self.alpha) * (-wt) + self.alpha * (-1) * abs(wt - meanwt))
        return reward


    def cal_maxmin_reward(self,wt,meanwt,trs,cost):
        reward = ((1 - self.alpha) * (-wt) + self.alpha * (-1) * meanwt)
        return reward



    def cal_cost(self,order,driver):
        dest = order.destRegion
        if dest in driver.preRegion:
            cost = 1
        else:
            cost = 0
        return cost

    def cal_money(self,wt,money):
        lambda_ = 7
        wt = (wt - 5) / 60
        pro = expon.cdf(wt, scale=1 / lambda_)
        num = random.random()
        if num < pro:
            money1 = 0
            symbol = 1
        else:
            money1 = money
            symbol = 0
        return money1,symbol






    def step(self,dDict,replayBuffer):
        updateDriverList = []
        for driver in self.driverList:
            symbol = driver.step_update_driver_info()
            if symbol == 1:
                updateDriverList.append(driver)
        for region in self.regionList:
            region.step_update_region_info()
        if self.cityTime < self.maxCityTime - 1:
            self.boost_step_order_info(self.cityTime + 1)
            self.boost_step_region_info()  # update supply and demand
        for driver in updateDriverList:
            region = np.array([driver.region + 1]) / self.regionNum  # 1
            regionWT = self.regionList[driver.region].meanwt
            regionWT = np.array([regionWT])
            t = np.array([driver.cityTime]) / self.maxCityTime
            lon = (driver.loc.lon - minlon) / (maxlon - minlon)
            lon = np.array([lon])
            lat = (driver.loc.lat - minlat) / (maxlat - minlat)
            lat = np.array([lat])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderWT = (cal_dis(driver.loc, order.oriLoc) / self.speed) * 60
                if orderWT <= nearwt:
                    nearwt = orderWT
            nearwt = np.array([nearwt])
            preRegionList = driver.preRegion
            preRegionList = [(x + 1) / self.regionNum for x in preRegionList]
          #  preRegion = np.pad(preRegionList, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')  # 62
            driverState = np.concatenate((region,regionWT,lon,lat,nearwt,t))
            # contextualState = self.con_state_calcualte()
            #next_driver_state = np.concatenate((driverState,contextualState))
            next_driver_state = driverState
            dDict[driver].add_nextState(next_driver_state)
            replayBuffer.add(dDict[driver].matchingState, dDict[driver].state, dDict[driver].action,
                             dDict[driver].reward, dDict[driver].cost,dDict[driver].nextState)
            dDict.pop(driver, None)
        self.cityTime += 1

    def plot(self, rewardList, actorLossList, rewardCriticLossList, testNum):
        # 配置中文显示和负号正常显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，适配Windows；Linux/Mac可换为['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (15, 5)  # 统一画布大小

        # 核心：数据预处理函数 - 统一转为标量，过滤无效值
        def preprocess_data(data_list):
            processed = []
            for item in data_list:
                # 跳过空值/None
                if item is None:
                    continue

                # 处理TensorFlow/PyTorch张量（根据你的框架二选一启用）
                try:
                    # 适配TensorFlow Tensor
                    import tensorflow as tf
                    if isinstance(item, tf.Tensor):
                        item = item.numpy().item()  # 转numpy标量
                except ImportError:
                    pass

                try:
                    # 适配PyTorch Tensor
                    import torch
                    if isinstance(item, torch.Tensor):
                        item = item.detach().cpu().numpy().item()  # 转CPU+numpy标量
                except ImportError:
                    pass

                # 处理numpy数组/列表
                if isinstance(item, (np.ndarray, list)):
                    # 展平单元素数组/列表为标量
                    if len(np.array(item).flatten()) == 1:
                        item = np.array(item).flatten()[0]
                    else:
                        continue  # 跳过多元素非标量（避免形状不均）

                # 过滤NaN/Inf，确保是数值类型
                if isinstance(item, (int, float)) and not np.isnan(item) and not np.isinf(item):
                    processed.append(item)

            return processed

        # 预处理所有绘图数据
        reward_list = preprocess_data(rewardList)
        actor_loss_list = preprocess_data(actorLossList)
        critic_loss_list = preprocess_data(rewardCriticLossList)

        # 确保x轴长度与y轴严格匹配
        x_reward = range(len(reward_list))
        x_actor = range(len(actor_loss_list))
        x_critic = range(len(critic_loss_list))

        # 创建子图（3列1行）
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # 子图1：Reward趋势（核心指标）
        if len(reward_list) > 0:
            ax1.plot(x_reward, reward_list, linewidth=2, color='#1f77b4', label='Mean Reward')
            ax1.set_title('Daily Mean Reward Trend', fontsize=12)
            ax1.set_xlabel('Training Day', fontsize=10)
            ax1.set_ylabel('Reward Value', fontsize=10)
            ax1.grid(alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No valid reward data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Daily Mean Reward Trend', fontsize=12)

        # 子图2：Actor Loss趋势（原报错行修复）
        if len(actor_loss_list) > 0:
            ax2.plot(x_actor, actor_loss_list, linewidth=2, color='#ff7f0e', label='Actor Loss')
            ax2.set_title('Actor Loss Trend', fontsize=12)
            ax2.set_xlabel('Training Step', fontsize=10)
            ax2.set_ylabel('Loss Value', fontsize=10)
            ax2.grid(alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid actor loss data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Actor Loss Trend', fontsize=12)

        # 子图3：Critic Loss趋势
        if len(critic_loss_list) > 0:
            ax3.plot(x_critic, critic_loss_list, linewidth=2, color='#2ca02c', label='Critic Loss')
            ax3.set_title('Critic Loss Trend', fontsize=12)
            ax3.set_xlabel('Training Step', fontsize=10)
            ax3.set_ylabel('Loss Value', fontsize=10)
            ax3.grid(alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No valid critic loss data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Critic Loss Trend', fontsize=12)

        # 调整布局，避免标签重叠
        plt.tight_layout()

        # 保存图片到result目录（确保目录存在）
        import os
        if not os.path.exists('result'):
            os.makedirs('result')  # 自动创建result目录，避免保存失败

        # 保存图片（带testNum区分不同实验）
        save_path = f'result/reward_loss_trend_test_{testNum}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"绘图完成，图片已保存至：{save_path}")

        # 关闭画布，释放内存
        plt.close(fig)






















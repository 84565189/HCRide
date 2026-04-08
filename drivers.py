from simulator.unitity import *


class Driver(object):
    def __init__(self, driverID, preRegionFreq, oriLoc):
        self.driverID = driverID
        self.oriLoc = oriLoc

        # 保存包含历史访问频次的字典 {region_id: frequency}
        self.preRegionFreq = preRegionFreq

        # 兼容旧代码逻辑提取区域 ID 列表
        if isinstance(preRegionFreq, dict):
            self.preRegion = list(preRegionFreq.keys())
        else:
            self.preRegion = preRegionFreq

        self.region = 0  # the current region
        self.loc = None  # the current lon-lat

        self.cityTime = 0

        self.state = 1  # 1 active  0 full
        self.servingTime = 0
        self.destinationLoc = None

        self.serveOrderNum = 0
        self.reward = 0
        self.cost = 0
        self.money = 0

    def set_day_info(self, day):
        self.cityDay = day

    def reset_driver_info(self):
        self.cityTime = 0
        self.loc = self.oriLoc
        self.region = cal_region(self.loc)
        self.serveOrderNum = 0
        self.servingTime = 0
        self.state = 1
        self.reward = 0
        self.cost = 0

    def accept_order(self, trs, loc, reward, wt, cost):
        self.state = 0  # service order
        self.servingTime = trs
        self.destinationLoc = loc
        self.serveOrderNum += 1
        self.reward += reward
        self.cost += cost

    def step_update_driver_info(self):
        self.cityTime += 1
        if self.servingTime > 0 and self.state == 0:
            self.servingTime -= 1
            if self.servingTime == 0:  # arrive
                self.state = 1  # active
                self.loc = self.destinationLoc
                self.region = cal_region(self.loc)
                self.destinationLoc = None
                return 1
            else:
                return 0  # not arrive
        else:
            return 0  # active
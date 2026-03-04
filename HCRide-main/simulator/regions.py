


class Region(object):
    def __init__(self,regionID,regionNum):
        self.regionID = regionID

        self.firstNeighbors = []
        self.secondNeighbors = []
        self.thirdNeighbors = []

        self.cityTime = 0

        self.meanwt = 0

        self.regionNum = regionNum
        self.driverList = []
        self.orderList = []

        self.wtList = []
        self.daywtList = []
        self.accwt = 0
        self.accOrder = 0
        self.dayAccwt = 0
        self.dayAccOrder = 0
        self.dayAccReward = 0

        self.accwtList = []




    def set_day_info(self, day):
        self.cityDay = day



    def set_region_meanwt(self):
        self.meanwt = 0

    def set_neighbors(self):
        x = self.regionID % 10
        y = int(self.regionID / 10)
        for i in range(self.regionNum):
            x1 = (i % 10)
            y1 = int(i/10)
            if pow(x - x1,2) + pow(y - y1,2) <= 2:
                self.firstNeighbors.append(i)
            elif pow(x - x1,2) + pow(y - y1,2) <= 8:
                self.secondNeighbors.append(i)
            elif pow(x - x1,2) + pow(y - y1,2) <= 18:
                self.thirdNeighbors.append(i)
            else:
                pass
        self.thirdNeighbors = self.thirdNeighbors + self.secondNeighbors + self.firstNeighbors
        self.secondNeighbors = self.secondNeighbors + self.firstNeighbors

        self.neighborLevel = [self.firstNeighbors, self.secondNeighbors, self.thirdNeighbors, list(range(100))]



    def reset_region_info(self):
        self.cityTime = 0
        self.driverList = []
        self.orderList = []
        self.wtList = []
        self.dayAccwt = 0
        self.daywtList = []
        self.dayAccOrder = 0
        self.dayAccReward = 0

    def add_driver(self,driver):
        self.driverList.append(driver)

    def add_order(self,order):
        self.orderList.append(order)

    def accept_order(self,reward,wt):
        self.wtList.append(wt)
        self.accwt += wt
        self.accwtList.append(wt)
        self.accOrder += 1
        self.dayAccwt += wt
        self.daywtList.append(wt)
        self.dayAccOrder += 1
        self.dayAccReward += reward

    def step_update_region_info(self):
        self.cityTime += 1
        self.driverList = []
        self.orderList = []
        if self.accOrder > 0:
            self.meanwt = self.accwt / self.accOrder






from simulator.unitity import *



from simulator.unitity import *


class Order(object):
    def __init__(self,orderID,orderDay,orderMin,orderRegion,oriLon,oriLat,destLon,destLat,price):
        self.orderID = orderID
        self.orderDay = orderDay
        self.orderMin = orderMin
        self.orderRegion = orderRegion
        self.oriLon = oriLon
        self.oriLat = oriLat
        self.destLon = destLon
        self.destLat = destLat
        self.oriLoc = Loc(self.oriLon,self.oriLat)
        self.destLoc = Loc(self.destLon,self.destLat)
        self.oriRegion = cal_region(self.oriLoc)
        self.destRegion = cal_region(self.destLoc)
        self.price = price
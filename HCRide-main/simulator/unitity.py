
import random
import numpy as np

minlon = 113.90
maxlon = 114.05
minlat = 22.530
maxlat = 22.670

londis = (maxlon - minlon) / 10
latdis = (maxlat - minlat) / 10


class Loc(object):
    def __init__(self,lon,lat):
        self.lon = lon
        self.lat = lat


class DispatchSolution:

    def __init__(self):
        self.driverID = None
        self.state = None
        self.matchingState = None
        self.action = 0
        self.trs = 0
        self.reward = None
        self.cost = None
        self.nextState = None

    def add_driver_ID(self,ID):
        self.driverID = ID

    def add_state(self,state):
        self.state = state

    def add_matchingState(self,matchingState):
        self.matchingState = matchingState

    def add_trs(self,trs):
        self.trs = trs

    def add_action(self,action):
        self.action = action

    def add_reward(self,reward):
        self.reward = reward

    def add_cost(self,cost):
        self.cost = cost

    def add_nextState(self,nextState):
        self.nextState = nextState


def cal_region(loc):
    lon = loc.lon
    lat = loc.lat
    x = int((lon - minlon) / londis)
    y = int((lat - minlat) / latdis)
    if x >= 10:
        x = 9
    if y >= 10:
        y = 9
    region = x + 10 * y
    return  region

def generate_loc(region):
    x = int(region % 10)
    y = int(region / 10)
    minRegionLon = minlon + x * londis
    minRegionLat = minlat + y * latdis
    lon = minRegionLon + random.random() * londis
    lat = minRegionLat + random.random() * latdis

    return Loc(lon,lat)


def cal_dis(loc1,loc2):
    lon1 = loc1.lon
    lat1 = loc1.lat
    lon2 = loc2.lon
    lat2 = loc2.lat

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of the Earth in kilometers (mean value)
    r = 6371.0

    # Calculate the distance
    distance = r * c
    return round(distance, 3)
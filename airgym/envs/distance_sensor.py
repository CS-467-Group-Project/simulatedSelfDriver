import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class DistanceInterface():
    def __init__(self, car_client):

        self._car_client = car_client
        self._distance_threshold = 20.0
        self.current_sensor_reading = 40.0

        self.state = {
            "distance_envelope": 40.0,
            "dist_thresh_breached": False
        }

    def get_car_client(self):
        return self._car_client

    def get_distance_threshold(self):
        return self._distance_threshold

    def get_current_sensor_reading(self):
        return self.get_car_client().getDistanceSensorData(vehicle_name="Car1")

    def update_envelope(self):
        self.state["distance_envelope"] =  float(self.get_current_sensor_reading().distance)

    def has_breached_threshold(self):
        self.update_envelope()
        if self.state["distance_envelope"] < self.get_distance_threshold():
            return True
        else:
            return False

    
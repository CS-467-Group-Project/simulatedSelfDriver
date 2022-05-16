from turtle import speed
import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class CameraInterface:
    
    def __init__(self, car_client):
        self.car_client = car_client
    
    def get_car_client(self):
        return self.car_client
    
    def get_image_information(self):
        '''
            returns a numpy array which describes a captured image. Call this whenever you want to capture an array 
        '''
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0",
                                airsim.ImageType.Scene,
                                False,
                                False)])
        response = responses[0]
        # Get Numpy Array
        img_values = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # Reshape the Array - H x W x 4
        img_values = img_values.reshape(response.height, response.width)
        return img_values
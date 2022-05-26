import setup_path
import airsim
import numpy as np
import math
import os
import time
import tempfile

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)

        self.action_space = spaces.Discrete(6)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        self.car_controls.throttle = 1
        self.car_controls.brake = 0
        self.car_controls.steering = 0
        self.car.setCarControls(self.car_controls)
        time.sleep(4.)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            print('Break')
        elif action == 1:
            self.car_controls.steering = 0
            print('Straight')
        elif action == 2:
            self.car_controls.steering = 0.5
            print('Hard Right')
        elif action == 3:
            self.car_controls.steering = -0.5
            print('Hard Left')
        elif action == 4:
            self.car_controls.steering = 0.25
            print('Slight Right')
        else:
            self.car_controls.steering = -0.25
            print('Slight Left')

        self.car.setCarControls(self.car_controls)
        time.sleep(.5)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])
        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 200
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]

        car_pt = self.state["pose"].position.to_numpy_array()
        #print(car_pt)

        dist = 10000000
        for i in range(0, len(pts) - 1):
            cross = np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
            norm = np.linalg.norm(pts[i] - pts[i + 1])
            cross_norm = np.linalg.norm(cross / norm)

            #print('Prev: %3.2f | New: %3.2f from [%4d, %4d], [%4d, %4d]' % 
            #      (dist, cross_norm, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            #)

            dist = min(dist, cross_norm)

        #print('Distance: %f' % dist)
        if dist > THRESH_DIST or self.car_state.speed >= 40:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5

            #print('Car Speed: %3.1f| Reward: %3.3f' % (self.car_state.speed, reward_speed))
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            print('====Negative Reward====')
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                print('====Car Stuck====')
                done = 1
        if self.car_controls.brake == 1:
            if self.car_state.speed == 0:
                print('====Car Stopped====')
                done = 1
        if self.state["collision"]:
            print('====Collision====')
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        return self._get_obs()

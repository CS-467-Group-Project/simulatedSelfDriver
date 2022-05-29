#import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.distance_sensor import DistanceInterface as DistInt


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
            "distance_envelope": 40.0,
            "dist_thresh_breached": False
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)
        """
        self.observation_space = gym.spaces.Dict(
            spaces={
                "ir": gym.spaces.Box(0, 255, image_shape, dtype=np.uint8),
                #"dist": gym.spaces.Box(),
                "cam": gym.spaces.Box(0, 255, image_shape, dtype=np.uint8),
                #"gps": gym.spaces.Box(),
            }
        )"""
        self.state_mapping = []
        self.image_request = self.car.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Infrared, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        
        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.2)

    def __del__(self):
        #print("SELF.CAR=",self.car)
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        im_final = []
        im_temp = []
        #print(response)

        for i in range(len(response)):
            if response[i].image_data_float:
                img1d = np.array(response[i].image_data_float, dtype=np.float)
                img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
                img2d = np.reshape(img1d, (response[i].height, response[i].width))
            
            else:
                img1d = np.frombuffer(response[i].image_data_uint8, dtype=np.uint8)
                img2d = np.reshape(img1d, (response[i].height, response[i].width, 3))
           

            from PIL import Image

            image = Image.fromarray(img2d)
            im_temp.append(np.array(image.resize((84, 84)).convert("L")))
            im_final.append(im_temp[i].reshape([84, 84, 1]))

        final_obs = {"ir": im_final[0], "cam": im_final[1]}
        #print(final_obs)
        return final_obs
        #self.state_mapping.append({"ir": im_final[0], "cam": im_final[1]})

        #return im_final[0]

    def _get_obs(self):
        responses = self.image_request
        image = self.transform_obs(responses)
        #print(image)

        # Update where the car is located on x,y,z plane
        self.car_state = self.car.getCarState()

        # Update car speed, gear, whether it has crashed, etc
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        data_car1 = self.car.getDistanceSensorData(vehicle_name="Car1")
        distanceSensorKit = DistInt(self.car)
        pastThreshold = DistInt.has_breached_threshold(distanceSensorKit)

        if pastThreshold:
            self.state["collision"] = True
            self.car.simGetCollisionInfo().has_collided = True
            print("Safety threshold breached")
            self.state["distance_envelope"] = data_car1.distance,
            self.state["dist_thresh_breached"] = True
            print(f"Car1 data: {data_car1.distance}")

        return image

    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        # Points of the figure 8 we want the car to traverse
        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        if dist > THRESH_DIST:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1
            print("collision reward")
        if self.state["dist_thresh_breached"]:
            done = 1
            # reward = reward - 10
            self.state["dist_thresh_breached"] = False
            print("breach reward")

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        print(obs)
        reward, done = self._compute_reward()


        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(0)
        self._do_action(1)
        return self._get_obs()

import setup_path
import gym
import airgym
import time
import subprocess
from datetime import date

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    n_steps=4,
    batch_size=4,
    n_epochs=10,
    gamma=.99,
    ent_coef=0.001,
    tensorboard_log="./tb_logs_ppo/",
    verbose=1,
)

# Create Tensorboard log
tb_log = "ppo_airsim_car_run_" + str(time.time())

# Train for a certain number of timesteps
model.learn(
    total_timesteps=220000, tb_log_name=tb_log, 
) 

# Training 1
# learning_rate=0.00025,
# n_steps=4,
# batch_size=4,
# n_epochs=10,
# gamma=.99,
# ent_coef=0.00,
# total_timesteps = 55000 (time_elapsed 83,926 seconds = 23 hours 18 minutes 46 seconds)
# car_env action sleep = .5
# reset sleep = .2
# no RPM on speed reset

# Training 2
# learning_rate=0.0003,
# n_steps=4,
# batch_size=64,
# n_epochs=10,
# gamma=.99,
# ent_coef=0.01,
# total_timesteps = 50000 (time_elapsed 73,008 seconds = 20 hours 16 minutes 48 seconds)
# car_env action sleep = .75
# reset sleep = .25
# >= 2400 RPM to speed reset

# Training 3
# learning_rate=0.00025,
# n_steps=4,
# batch_size=4,
# n_epochs=10,
# gamma=.99,
# ent_coef=0.001,
# total_timesteps = 220000 (time_elapsed ? seconds = ? hours ? minutes ? seconds)
# car_env action sleep = .5
# reset sleep = .25
# added do_action(1) to reset
# no RPM to speed reset

# Save policy weights
model.save("ppo_airsim_car_policy_" + date.today().isoformat())

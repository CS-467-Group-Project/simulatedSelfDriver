import setup_path
import gym
import airgym
import time
import subprocess

from stable_baselines3 import DQN
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
eval = 1000

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=eval,
    learning_starts=20000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

#model = DQN.load(
#    "dqn_airsim_car_policy_short",
#    env,
#)

#obs = env.reset()

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=eval,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Create Tensorboard log
tb_log="dqn_airsim_car_run_" + str(time.time())

# Train for a certain number of timesteps
model.learn(
    total_timesteps=100000, tb_log_name=tb_log, **kwargs
)

# Save policy weights
model.save("dqn_airsim_car_policy_" + str(time.gmtime()))

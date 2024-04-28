import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO, DQN, TD3, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
import os
import time
from robotEnv import CustomEnv
import pygame

import matplotlib.pyplot as plt

model_dir = f"report/A2C_Robot/model"
logdir = f"report/A2C_Robot/logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# screen = pygame.display.set_mode((800, 600))
# Create and wrap your custom environment
env = CustomEnv()  # Replace 'screen' with your screen object if needed
env = Monitor(env, logdir)  # Wrap with Monitor for logging
# env.reset()

# Initialize and train the PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, batch_size=4)
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 80_000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

    # Render the environment in rgb_array mode after training

    obs = env.render()

# Close the environment
env.close()
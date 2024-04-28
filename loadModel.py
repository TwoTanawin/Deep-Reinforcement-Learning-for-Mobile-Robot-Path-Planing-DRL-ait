from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from robotEnv import CustomEnv

# Create a new environment for inference
env = CustomEnv()

# Load the trained model
model = A2C.load("report/A2C_Robot/model/99000", env=env)

# Lists to store rewards and episode lengths
episode_rewards = []
episode_lengths = []

# Inference loop
episodes = 5
for _ in range(episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        env.render(mode='human')  # Render the environment in human mode during inference
        print(f"reward : {reward}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

# Plotting rewards
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards during Inference')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(episode_lengths)), episode_lengths)
plt.xlabel('Episodes')
plt.ylabel('Episode Length')
plt.title('Episode Lengths during Inference')

plt.tight_layout()
plt.savefig("report/img1.jpg")
plt.show()

env.close()  # Close the environment after inference

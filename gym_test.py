# -*- coding = utf-8 -*-
# copyright: sukien
# fileName: gym_test.py
# creatTime: 2024/7/31 21:50
# @Software: PyCharm
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) su_kien. All Rights Reserved
#
# @Time    : 31/07/2024 18:28
# @Author  : su_kien
# @File    : test.py
# @IDE     : PyCharm

import gym
import imageio
import argparse
from TD3 import TD3
from utils import scale_action

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/LunarLander.gif')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--render', type=str, default='True')  # Use str type
parser.add_argument('--save_video', type=str, default='True')  # Use str type

args = parser.parse_args()

def str_to_bool(s):
    return s.lower() in ('true', '1', 't', 'yes')

def main():
    render = str_to_bool(args.render)
    save_video = str_to_bool(args.save_video)

    env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.99,
                tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
    agent.load_models(400)

    if save_video:
        video = imageio.get_writer(args.figure_file, fps=args.fps)

    done = False
    observation, info = env.reset()
    while not done:
        if render:
            env.render()
        action = agent.choose_action(observation, train=True)
        action_ = scale_action(action, low=env.action_space.low, high=env.action_space.high)
        observation_, reward, terminated, truncated, info = env.step(action_)
        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_

        if save_video:
            video.append_data(env.render())

    if save_video:
        video.close()
    env.close()

if __name__ == '__main__':
    main()

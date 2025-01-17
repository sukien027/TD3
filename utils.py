#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) su_kien. All Rights Reserved 
#
# @Time    : 31/07/2024 18:27
# @Author  : su_kien
# @Email   : sukien027@gmail.com
# @File    : utils.py
# @IDE     : PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt


def create_directory(path: str, sub_path_list: list):
    for sub_path in sub_path_list:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Path: {} create successfully!'.format(path + sub_path))
        else:
            print('Path: {} is already existence!'.format(path + sub_path))


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='b', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.savefig(figure_file)


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_

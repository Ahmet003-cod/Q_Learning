# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
print(env.render())

"""
0 = güney
1 = kuzey
2 = doğu
3 = batı
4 = yolcuyu almak
5 = yolcuyu bırakmak
"""

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))
print(q_table)

alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Eğitim
for i in tqdm(range(1, 100001)):
    state, _ = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, reward, done, info, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

print("Training finished.")

# Test
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in tqdm(range(episodes)):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        new_state, reward, done, info, _ = env.step(action)
        state = new_state

        if reward == -10:
            penalties += 1
        epochs += 1

    total_epochs += epochs
    total_penalties += penalties
print()
print("Results after {} episodes:".format(episodes))
print("Average timesteps per episode:", total_epochs / episodes)
print("Average penalties per episode:", total_penalties / episodes)

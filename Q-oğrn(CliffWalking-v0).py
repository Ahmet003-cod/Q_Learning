# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 22:59:11 2025

@author: Huzur Bilgisayar
"""
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import gym
import numpy as np

env=gym.make('CliffWalking-v0',render_mode="ansi")
env.reset()
print(env.render())

nb_state=env.observation_space.n
nb_action=env.action_space.n

q_table=np.zeros((nb_state,nb_action))
print("Q_table:")
print(q_table)

#0: Move up
#
#:1 Move right
#
#2: Move down

#3: Move left
episodes=500
alpha=0.1
gamma=0.6
epsilon=0.1

for i in tqdm(range(1,105001)):
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
        state=new_state
print("Training finished.")

print(q_table)
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
        
        if reward == -100:
            penalties += 1
        epochs += 1

    total_epochs += epochs
    total_penalties += penalties
print()
print("Results after {} episodes:".format(episodes))
print("Average timesteps per episode:", total_epochs / episodes)
print("Average penalties per episode:", total_penalties / episodes)












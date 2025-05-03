# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 22:03:39 2025

@author: Huzur Bilgisayar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:59:43 2025

@author: Huzur Bilgisayar
"""

import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
print(env.render())

"""
Actions:
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

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Training
for i in tqdm(range(1, 100001)):
    state, _ = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])
        state = new_state

print("Training finished.")

# Evaluation
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in tqdm(range(episodes)):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = new_state
        
        if reward == -10:
            penalties += 1
        
        epochs += 1
    
    total_epochs += epochs
    total_penalties += penalties

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

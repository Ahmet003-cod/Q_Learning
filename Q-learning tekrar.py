# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 18:09:47 2025

@author: Huzur Bilgisayar
"""
import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
env=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
env.reset()

nb_state=env.observation_space.n
nb_actions=env.action_space.n

q_table=np.zeros((nb_state,nb_actions))

"""
0: Move left

1: Move down

2: Move right

3: Move up
"""
print("Q_table")
print(q_table)
episodes=400
alpha=0.5
gamma=0.9
outcomes=[]

for _ in tqdm(range(episodes)):
    state,_=env.reset()
    done=False
    outcomes.append("Failura")
    while not done:
        if np.max(q_table[state]>0):
            action=np.argmax(q_table[state])
        else:
            action=env.action_space.sample()
        new_state,reward,done,info,_=env.step(action)
        q_table[state,action]= q_table[state,action]+alpha*(reward+gamma*np.max(q_table[new_state])-q_table[state,action])
        
        state=new_state
        if reward:
            outcomes[-1]="Succes"
            
print("After trainin q table:::")
print(q_table)

plt.bar(range(episodes),outcomes)


episodes=100
nb_succes=0
for _ in tqdm(range(episodes)):
    state,_=env.reset()
    done=False
    while not done:
       if np.max(q_table[state])>0:
           action=np.argmax(q_table[state])
       else:
           action=env.action_space.sample()
        
       new_state,reward,done,info,_=env.step(action)   
       state=new_state
       nb_succes+=reward

print("succes rote=",nb_succes)
        



        
    




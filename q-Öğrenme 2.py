# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:56:01 2025

@author: Huzur Bilgisayar
"""

import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

Enviroment=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi") 
Enviroment.reset()
nb_states=Enviroment.observation_space.n#toplam kaç satırldan yani tüm dürüm toplamı
nb_action=Enviroment.action_space.n #kaç sütünda oluştuğunu bize verir

q_table=np.zeros((nb_states,nb_action))
action=Enviroment.action_space.sample()

print("Q_table:::")
print(q_table)#ajan beyni
episodes=5000 #episode
alpha=0.5 #learning rate
gamma=0.9 #dicount rate
outcome=[] #başarılımı yok başarısızmı buraya depolar

#training
for _ in tqdm(range(episodes)):
    state,_=Enviroment.reset()
    done=False #Ajanın başarı durumu
    outcome.append("Failure")
    
    while not done:#ajan başarı olana kadar state içerisinde hareket et
    #action
        if np.max(q_table[state])>0:
           action=np.argmax(q_table[state])
        else:
           action=Enviroment.action_space.sample()

        new_state,reword,done,info,_=Enviroment.step(action)
        #Update q table
        q_table[state,action]=q_table[state,action]+alpha*(reword+gamma*np.max(q_table[new_state])-q_table[state,action])
    
        state=new_state
        if reword:
           outcome[-1]="Succes"

print("Q table after training")
print(q_table)

plt.bar(range(episodes),outcome)

#test
episodes=100
nb_succees=0
for _ in tqdm(range(episodes)):
    state,_=Enviroment.reset()
    done=False
    
    while not done:
        if np.max(q_table[state])>0:
            action=np.argmax(q_table[state])
        else:
            action=Enviroment.action_space.sample()
        new_state,reword,done,info,_=Enviroment.step(action)
        
        state=new_state
        nb_succees+=reword
print()
print("Succes rote:",100*nb_succees/episodes)
            
    















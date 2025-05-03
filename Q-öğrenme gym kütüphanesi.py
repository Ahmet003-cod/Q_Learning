# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:36:20 2025
@author: Huzur Bilgisayar
"""
import gym
import random
import numpy as np

Enviroment=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi") 
Enviroment.reset()
nb_states=Enviroment.observation_space.n#toplam kaç satırldan yani tüm dürüm toplamı
nb_action=Enviroment.action_space.n #kaç sütünda oluştuğunu bize verir

q_table=np.zeros((nb_states,nb_action))
action=Enviroment.action_space.sample()

print("Q_table:::")
#print(q_table)
"""
sol=0
aşagi=1
sağa=2
yukarı=3
"""
new_state,reword,done,info,_=Enviroment.step(action)
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:27:55 2020

@author: robin
"""

from domain import Domain
from agent import Agent

if __name__=="__main__":
    setting = int(input("Press 0 for deterministic setting or 1 for stochastic setting"))
    protocol = int(input("Choose the protocol you want to display: 1, 2 or 3"))
    domain = Domain()
    domain.discount_factor = 0.4
    domain.setting = setting
    domain.update()
    print(domain.Q)

    if protocol == 1:
        print("Agent1 first protocol")
        print("-----------------------")
        agent1 = Agent(domain)
        agent1.train(100)
    elif protocol == 2:
        print("Agent2 second protocol")
        print("-----------------------")
        agent2 = Agent(domain)
        agent2.train2(100)
    elif protocol == 3:
        print("Agent3 third protocol")
        print("-----------------------")
        agent3 = Agent(domain)
        agent3.train3(100)
    else:
        print("You didn't choose a valid protocol")
        print("Protocols are between 1 and 3")
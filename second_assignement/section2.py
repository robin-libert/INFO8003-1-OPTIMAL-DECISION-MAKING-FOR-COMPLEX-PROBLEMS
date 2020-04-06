# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:12:20 2020

@author: robin
"""
from domain import Domain
from agent import Agent

if __name__ == "__main__":
    domain = Domain()
    agent = Agent(domain)
    policy = agent.ACCELERATE_POLICY
    ht = agent.run(policy)
    # Display the trajectory for a policy which make the agent always accelerate
    counter = 1
    for e in range(0, len(ht) - 1, 3):
        print("State {}: {} --- Action: {} --- Reward: {} --- State {}: {}".format(counter, ht[e], ht[e + 1], ht[e + 2],
                                                                                   counter + 1, ht[e + 3]))
        counter += 1
    # agent.create_video(ht)

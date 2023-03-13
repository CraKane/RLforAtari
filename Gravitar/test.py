import gym
import time
import numpy as np

env=gym.make('Pendulum-v0') #创建对应的游戏环境
env.seed(1) #可选，设置随机数，以便让过程重现
# env=env.unwrapped #可选，为环境增加限制，对训练有利
for episode in range(100): #每个回合
    print(episode)
    score = 0.0
    s=env.reset() #重新设置环境，并得到初始状态
    while True: #每个步骤
        env.render() #展示环境
        a=env.action_space.sample() # 智能体随机选择一个动作
        # print(s)
        print(env.action_space.low[0])
        # print(s)
        # time.sleep(5)
        s_,r,done,info=env.step(a) #环境返回执行动作a后的下一个状态、奖励值、是否终止以及其他信息
        s = s_
        # print(s/255.0)
        score += r
        if done:
            break
    print(score)
    time.sleep(10)
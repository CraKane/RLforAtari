# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 17:37
# @Author  : youngleesin
# @FileName: RL.py
# @Software: PyCharm
# imports
import gym
import collections
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# hyperparameters
learning_rate = 0.001
gamma = 0.9
d = 20
c = 2
buffer_limit = 10000
batch_size = 32
soft = 0.02
ad = 80
epsilon = 0.2


# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer、
# env = gym.make('Asterix-ram-v0')
env = gym.make('Pendulum-v0')
# env = gym.wrappers.Monitor(env, "drive/My Drive/rl/video1",
#                            video_callable=False, force=True)# lambda episode_id: (episode_id % video_every) == 0, force=True)
# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 2021
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
# env.action_space.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_n = env.observation_space.shape[0]
action_n = env.action_space.shape[0]


def soft_update(q_target, q, soft):
    for target_param, param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_((1-soft)*target_param.data+soft*param.data)

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def categorical_sample(probs, use_cuda=False):
    # https://blog.csdn.net/monchin/article/details/79787621
    int_acs = torch.multinomial(probs, 1) # action idx
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    # acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1) # one-hot representation
    return int_acs

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc1 = nn.Linear(state_n, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, action_n)
        self.fc4 = nn.Linear(84, action_n)

    def forward(self, obs):
        # print(obs.shape)
        x = obs.view(obs.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))
        sigma = torch.relu(self.fc4(x))
        return mu.detach(), sigma.detach()

class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_n, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, s):
        # print(self.fc1(obs)[0].shape)
        # x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.Actor = policy()
        self.Actor_target = policy()
        hard_update(self.Actor_target, self.Actor)
        self.Critic = critic()
        self.Critic_target = critic()
        hard_update(self.Critic_target, self.Critic)
        self.ctrain = torch.optim.Adam(self.Critic.parameters(),lr=learning_rate)
        self.atrain = torch.optim.Adam(self.Actor.parameters(),lr=learning_rate)
        self.loss_td = nn.MSELoss()

    def sample_action(self, obs):
        # print(obs.shape)
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        mu, sigma = self.Actor(obs)
        a = torch.clip(torch.normal(mu, sigma), -c, c)
        # print(torch.normal(mu, sigma))
        return a.squeeze(0).detach().numpy()

    def critic_learn(self, memory):
        s, a, r, s_prime = memory.sample(batch_size)
        q_a = self.Critic(s)
        next_qs = self.Critic_target(s_prime)
        q_target = r + gamma * next_qs # q_target = 负的
        td_error = self.loss_td(q_target, q_a)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def actor_learn(self, memory):
        s, _, r, s_prime = memory.sample(batch_size)
        # surrogate
        mu, sigma = self.Actor(s)
        mu_, sigma_ = self.Actor_target(s)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        m_ = torch.distributions.Normal(loc=mu_, scale=sigma_)
        a = torch.clip(m.sample(), -c, c)
        ISW = (torch.exp(m.log_prob(a)) / (torch.exp(m_.log_prob(a)) + 1e-5)).unsqueeze(1)
        v = self.Critic(s)
        q = r + gamma * self.Critic_target(s_prime)
        Adv = q - v
        surr = ISW*Adv
        surr_ = torch.clip(ISW, 1-epsilon, 1+epsilon)*Adv
        # print("ssss", Adv.shape, q.shape, v.shape, surr.shape, surr_.shape)
        ## PPO2
        loss_a = -torch.mean(torch.min(surr, surr_))
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()


q = PPO().to(device)

def plot(score, mean):
    plt.figure()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score, color='r')
    plt.plot(mean, color='blue')
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show()

memory = ReplayBuffer()
scores = []
mean = []

for episode in range(1000):
    s = env.reset()
    score = 0.0

    for step in range(1000):
        # env.render()
        a = q.sample_action(s)
        s_prime, r, done, _ = env.step(a)
        score += r
        memory.put((s, a, r, s_prime))
        s = s_prime

        if memory.size() > batch_size:
            # train(q, q_target, memory, optimizer)
            q.critic_learn(memory)
            if step % d == 0:
                q.actor_learn(memory)
                soft_update(q.Critic_target, q.Critic, soft)
                if step % ad == 0:
                    soft_update(q.Actor_target, q.Actor, soft)

    scores.append(score)
    mean.append(sum(scores) / len(scores))
    print(episode, ': ', sum(scores) / len(scores), score)


plot(scores, mean)
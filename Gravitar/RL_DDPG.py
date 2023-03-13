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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import logging
from torch.autograd import Variable
from datetime import datetime

# hyperparameters
learning_rate = 0.001
gamma = 0.99
buffer_limit = 10000
batch_size = 32
video_every = 50
print_every = 1000
soft = 0.02


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
# print(device, torch.cuda.device_count())


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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.shape[0])

    def forward(self, obs, return_all_probs=False):
        # print(obs.shape)
        x = obs.view(obs.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # x = F.leaky_relu(self.fc1(obs))
        # x = F.leaky_relu(self.fc2(x))
        # out = self.fc3(x)
        # probs = F.softmax(out, dim=1)
        # on_gpu = next(self.parameters()).is_cuda
        # int_act = categorical_sample(probs)
        # print(act, int_act)
        # rets = [int_act]
        # log_probs = F.log_softmax(out, dim=1)
        # if return_all_probs:
        #     rets.append(probs)
        # rets.append(log_probs.gather(1, int_act))  # entropy regularization
        return x # int_act

class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        # print(self.fc1(obs)[0].shape)
        # print(s.shape, a.squeeze(1).shape)
        x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fcs(s)
        # y = self.fca(a.float())
        # net = F.relu(x + y)
        # actions_value = self.out(net)

        return x# actions_value

class DDPG(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(DDPG, self).__init__()
        self.a_dim, self.s_dim = a_dim, s_dim
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
        out = self.Actor(obs)
        return out.squeeze(0).detach().numpy()

    def learn(self, memory):
        for i in range(5):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)
            q_a = self.Critic(s, a)
            a_ = self.Actor_target(s_prime)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            next_qs = self.Critic_target(s_prime, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q_target = r + gamma * next_qs * done_mask  # q_target = 负的
            abs_error = torch.abs((q_target - q_a)).detach().squeeze().numpy()
            # print(abs_error.shape)
            # memory.batch_update(tree_idx, abs_error)
            # q_target *= torch.from_numpy(ISWeights)
            # print((target*torch.from_numpy(ISWeights)).shape)
            # q *= torch.from_numpy(ISWeights)
            td_error = self.loss_td(q_target, q_a)
            # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
            # print(td_error)
            self.ctrain.zero_grad()
            td_error.backward()
            self.ctrain.step()

            # soft target replacement
            #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
            s, a, r, s_prime, done_mask = memory.sample(batch_size)

            a = self.Actor(s)
            q = self.Critic(s,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
            # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
            loss_a = -torch.mean(q)
            #print(q)
            #print(loss_a)
            self.atrain.zero_grad()
            loss_a.backward()
            self.atrain.step()



q = DDPG(env.action_space.shape[0], env.observation_space.shape[0]).to(device)

memory = ReplayBuffer()
score = 0.0
marking = []
format_ = 'logs/ddpg/%Y%M%d_%H%M%S.log'
logfile = datetime.strftime(datetime.now(), format_)
writer = SummaryWriter(logfile)


for n_episode in range(int(1e32)):
    s = env.reset()
    # print(len(s))
    done = False
    score = 0.0

    while True:
        # print(torch.from_numpy(s).float().shape)
        a = q.sample_action(s)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        score += r
        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime


        if done:
            break


    if memory.size() > 2000:
        # train(q, q_target, memory, optimizer)
        q.learn(memory)
        soft_update(q.Actor_target, q.Actor, soft)
        soft_update(q.Critic_target, q.Critic, soft)
        # q_target.load_state_dict(q.state_dict())
        # hard_update(q.target_policy, q.policy)
        # hard_update(q.target_critic, q.critic)

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    writer.add_scalar("mean_reward", np.array(marking).mean(), n_episode)
    if n_episode % 100 == 0:
        # print(score)
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []
        # state = {'net': q.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': n_episode, 'lr': learning_rate}
        # torch.save(state, 'saves/ddpg.dat')


    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode % print_every == 0 and n_episode != 0:
        print("episode: {}, score: {:.1f}".format(n_episode, score))

writer.close()

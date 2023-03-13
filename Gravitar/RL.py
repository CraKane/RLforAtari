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

# hyperparameters
learning_rate = 0.001
gamma = 0.9
buffer_limit = 50000
batch_size = 32
video_every = 50
print_every = 5


# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer、
env = gym.make('Gravitar-ram-v0')
env = gym.wrappers.Monitor(env, "drive/My Drive/rl/video1",
                           video_callable=False, force=True)# lambda episode_id: (episode_id % video_every) == 0, force=True)
# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.device_count())
MSELoss = torch.nn.MSELoss()


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
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
               torch.tensor(r_lst).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
               torch.tensor(done_mask_lst).cuda()

    def size(self):
        return len(self.buffer)


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
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, obs, return_all_probs=False):
        # print(self.fc1(obs)[0].shape)
        x = F.leaky_relu(self.fc1(obs))
        x = F.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        # print(act, int_act)
        rets = [int_act]
        log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        rets.append(log_probs.gather(1, int_act))
        return rets

class SAC(nn.Module):
    def __init__(self):
        super(SAC, self).__init__()
        self.reward_scale = 100.0
        self.policy = policy()
        self.target_policy = policy()

        hard_update(self.target_policy, self.policy)
        input_size = int(np.array(env.observation_space.shape).prod() + np.array(env.action_space.shape).prod())
        # print(input_size)
        self.critic = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 84),
            nn.LeakyReLU(inplace=True),
            nn.Linear(84, 1)
        )

        self.target_critic = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 84),
            nn.LeakyReLU(inplace=True),
            nn.Linear(84, 1)
        )
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def update_critic(self, memory, soft=True):
        for i in range(4):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)
            # print(s.shape, a.shape)
            a_prime, next_log_pi = self.target_policy(s_prime)
            # print(a_prime.shape, next_log_pi.shape)
            x_prime = torch.cat((s_prime, a_prime), axis=1).cuda()
            x = torch.cat((s, a), axis=1).cuda()
            next_qs = self.target_critic(x_prime)
            q_value = self.critic(x)
            q_loss = 0
            target_q = r.cuda() + gamma * next_qs * done_mask.cuda()
            if soft:
                target_q -= next_log_pi / self.reward_scale
            q_loss += MSELoss(q_value, target_q.detach())

            q_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()



    def update_policy(self, memory, soft=True):
        for i in range(4):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)

            curr_ac, probs, log_pi = self.policy(s, return_all_probs=True)
            all_q = []
            x = torch.cat((s, curr_ac), axis=1).cuda()
            a_list = torch.Tensor(range(18))
            a_new = a_list.expand(s.shape[0], 18).contiguous().view(s.shape[0], 18, 1).transpose(1, 0).cuda()
            s_new = s.expand((18, s.shape[0], s.shape[1])).cuda()
            x_all = torch.cat((s_new, a_new), axis=2).cuda()
            # print(x_all)
            # for j in range(18):
            #     all_q.append(self.critic(x_all[j]).cpu().detach().numpy())
            all_q = self.critic(x_all)
            # print(all_q.shape)

            q = self.critic(x)
            # print(q.shape)
            all_q_ = all_q.transpose(1, 0).squeeze(2).cuda()
            # print(all_q_.shape, probs.shape)
            v = (all_q_ * probs).sum(dim=1, keepdim=True)
            # print(v.shape)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi / self.reward_scale - pol_target).detach().mean()
                pol_loss = Variable(pol_loss, requires_grad=True)
            else:
                pol_loss = -pol_target.detach().mean()
                pol_loss = Variable(pol_loss, requires_grad=True)

            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()


    def sample_action(self, obs, epsilon):
        # print(obs.shape)
        [out, log] = self.policy(obs)
        return out.item()


q = SAC().to(device)

memory = ReplayBuffer()

score = 0.0
marking = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
writer = SummaryWriter('logs')

for n_episode in range(int(1e32)):
    epsilon = 0.06
    s = env.reset()
    # print(len(s))
    done = False
    score = 0.0

    while True:
        # print(torch.from_numpy(s).float().shape)
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0).cuda(), epsilon)
        # print(a)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r / 100.0, s_prime, done_mask))
        s = s_prime

        score += r
        if done:
            break

    if memory.size() > 2000:
        # train(q, q_target, memory, optimizer)
        q.update_critic(memory)
        q.update_policy(memory)

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode % 100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        logging.basicConfig(level=logging.INFO,
                            filename='gravitar-log1.txt',
                            filemode='a',
                            format='%(message)s')
        # print(len(batch['utilities']))
        logging.info("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []
        # writer.add_scalar("mean_reward", np.array(marking).mean(), n_episode)

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode % print_every == 0 and n_episode != 0:
        # q_target.load_state_dict(q.state_dict())
        hard_update(q.target_policy, q.policy)
        hard_update(q.target_critic, q.critic)
        torch.save({'SAC':q.state_dict(), 'policy_optimizer':q.policy_optimizer.state_dict(), 'critic_optimizer':q.critic_optimizer.state_dict()}, 'logs/save.chkpt')
        print("episode: {}, score: {:.1f}, epsilon: {:.2f}".format(n_episode, score, epsilon))

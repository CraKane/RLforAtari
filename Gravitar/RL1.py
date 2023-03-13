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


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class ReplayBuffer():
    """
        This Memory class is modified based on the original code from:
        https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
        """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.tree = SumTree(buffer_limit)

    def put(self, transition):
        transition = np.hstack(transition)
        # print(transition[:env.observation_space.shape[0]])
        self.buffer.append(transition)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / (min_prob+0.0001), -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

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
            tree_idx, batch_memory, ISWeights = memory.sample(batch_size)
            s = torch.from_numpy(batch_memory[:, :env.observation_space.shape[0]]).float().cuda()
            a = torch.from_numpy(batch_memory[:, env.observation_space.shape[0]]).unsqueeze(1).float().cuda()
            r = torch.from_numpy(
                batch_memory[:, env.observation_space.shape[0] + 1:-env.observation_space.shape[0] - 1]).float().cuda()
            s_prime = torch.from_numpy(batch_memory[:, -env.observation_space.shape[0] - 1:-1]).float().cuda()
            done_mask = torch.from_numpy(batch_memory[:, -1]).unsqueeze(1).float().cuda()
            # print(s.shape, a.shape)
            a_prime, next_log_pi = self.target_policy(s_prime)
            # print(a_prime.shape, next_log_pi.shape)
            x_prime = torch.cat((s_prime, a_prime), axis=1).cuda()
            x = torch.cat((s, a), axis=1).cuda()
            next_qs = self.target_critic(x_prime)
            q_value = self.critic(x)
            q_loss = 0
            gammas = torch.tensor(list(map(lambda x: pow(gamma, x), torch.arange(r.shape[1])))).cuda()
            # print(gammas)
            # print((r*gammas).sum(1).shape)
            target_q = (r * gammas).sum(1).unsqueeze(1) + pow(gamma, r.shape[1]) * next_qs * done_mask  # q_target = 负的
            if soft:
                target_q -= next_log_pi / self.reward_scale
            q_loss += MSELoss(q_value, target_q.detach())

            q_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()



    def update_policy(self, memory, soft=True):
        for i in range(4):
            tree_idx, batch_memory, ISWeights = memory.sample(batch_size)
            s = torch.from_numpy(batch_memory[:, :env.observation_space.shape[0]]).float().cuda()
            r = torch.from_numpy(
                batch_memory[:, env.observation_space.shape[0] + 1:-env.observation_space.shape[0] - 1]).float().cuda()
            s_prime = torch.from_numpy(batch_memory[:, -env.observation_space.shape[0] - 1:-1]).float().cuda()
            done_mask = torch.from_numpy(batch_memory[:, -1]).unsqueeze(1).float().cuda()

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


    def sample_action(self, obs):
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
        s_tmp = s.copy()
        r = []
        for i in range(3):
            a_tmp = q.sample_action(torch.from_numpy(s_tmp).float().unsqueeze(0).cuda())
            if i == 0:
                a = a_tmp
            # print(a)
            s_prime, r_tmp, done, info = env.step(a_tmp)
            score += r_tmp
            s_tmp = s_prime
            r.append(r_tmp)
            done_mask = 0.0 if done else 1.0
            if done:
                break
        while True:
            if len(r) == 3:
                break
            r.append(0)
        # print(s, a, np.array(r)/100.0, s_prime, done_mask)
        memory.put((s, a, np.array(r) / 100.0, s_prime, done_mask))
        s = s_prime.copy()
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
                            filename='gravitar-log.txt',
                            filemode='a',
                            format='%(message)s')
        # print(len(batch['utilities']))
        logging.info("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []
        torch.save({'SAC': q.state_dict(), 'policy_optimizer': q.policy_optimizer.state_dict(),
                    'critic_optimizer': q.critic_optimizer.state_dict()}, 'logs/save.chkpt')
        # writer.add_scalar("mean_reward", np.array(marking).mean(), n_episode)

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode % print_every == 0 and n_episode != 0:
        # q_target.load_state_dict(q.state_dict())
        hard_update(q.target_policy, q.policy)
        hard_update(q.target_critic, q.critic)
        # print("episode: {}, score: {:.1f}".format(n_episode, score))

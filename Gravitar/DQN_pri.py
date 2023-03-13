# imports
import gym
import collections
import random
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# hyperparameters
learning_rate = 0.0005
gamma = 0.9
buffer_limit = 100000
batch_size = 32
video_every = 50
print_every = 100
soft = 0.4


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



class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # input expands to a flat vector
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, env.action_space.n-1)
        else:
            return out.argmax().item()


def soft_update(q_target, q, soft):
    for target_param, param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_((1-soft)*target_param.data+soft*param.data)


def train(q, q_target, memory, optimizer):
    for i in range(10):
        tree_idx, batch_memory, ISWeights = memory.sample(batch_size)
        s = torch.from_numpy(batch_memory[:, :env.observation_space.shape[0]]).float()
        a = torch.from_numpy(batch_memory[:, env.observation_space.shape[0]]).unsqueeze(1).long()
        r = torch.from_numpy(
            batch_memory[:, env.observation_space.shape[0] + 1:-env.observation_space.shape[0] - 1]).float()
        s_prime = torch.from_numpy(batch_memory[:, -env.observation_space.shape[0] - 1:-1]).float()
        done_mask = torch.from_numpy(batch_memory[:, -1]).unsqueeze(1).float()
        # print(a)
        q_out = q(s)
        q_a = q_out.gather(1, a)  # gather: select value according to the index
        # print(q_a.shape)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # print(max_q_prime.shape)
        target = r + gamma * max_q_prime * done_mask
        abs_error = torch.abs((target - q_a)).detach().squeeze().numpy()
        # print(abs_error.shape)
        memory.batch_update(tree_idx, abs_error)
        target *= torch.from_numpy(ISWeights)
        # print((target*torch.from_numpy(ISWeights)).shape)
        q_a *= torch.from_numpy(ISWeights)
        loss = F.smooth_l1_loss(target, q_a)
        # print(loss)
        # if loss.shape != torch.Size([]):
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
env = gym.make('Asterix-ram-v0')
# env = gym.wrappers.Monitor(env, "video",
#                            video_callable=lambda episode_id: (episode_id % video_every) == 0, force=True)

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 2021
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
format_ = 'logs/dqn_pri/%Y%M%d_%H%M%S.log'
logfile = datetime.strftime(datetime.now(), format_)
writer = SummaryWriter(logfile)
# env.action_space.seed(seed)

q = QNetwork()
q_target = QNetwork()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

score = 0.0
marking = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

for n_episode in range(int(1e32)):
    epsilon = max(0.01, 1 - 0.01 * (n_episode / 1000))  # linear annealing from 8% to 1%
    s = env.reset()
    # print(len(s))
    done = False
    score = 0.0

    while True:
        # print(torch.from_numpy(s).float().shape)
        a = q.sample_action(torch.from_numpy(s / 255.0).float().unsqueeze(0), epsilon)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.put((s / 255.0, a, r / 100.0, s_prime / 255.0, done_mask))
        s = s_prime

        score += r
        if done:
            break

    if memory.size() > 2000:
        train(q, q_target, memory, optimizer)
        # q.update_critic(memory)
        # q.update_policy(memory)

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    writer.add_scalar("mean_reward", np.array(marking).mean(), n_episode)
    if n_episode % 100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []
        state = {'net':q.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':n_episode, 'lr':learning_rate}
        torch.save(state, 'saves/dqn_pri.dat')


    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode % print_every == 0 and n_episode != 0:
        soft_update(q_target, q, soft)
        # q_target.load_state_dict(q.state_dict())
        # hard_update(q.target_policy, q.policy)
        # hard_update(q.target_critic, q.critic)
        print("episode: {}, score: {:.1f}, epsilon: {:.2f}".format(n_episode, score, epsilon))

# writer.close()
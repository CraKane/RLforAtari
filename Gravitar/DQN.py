# imports
import gym
import collections
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# hyperparameters
learning_rate = 0.005
gamma = 0.9
buffer_limit = 10000
batch_size = 32
video_every = 50
print_every = 1000
soft = 0.02
epsi_high = 0.8
epsi_low = 0.05
decay = 400
# setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
env = gym.make('CartPole-v0')
# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 2021
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
state_n = env.observation_space.shape[0]
action_n = env.action_space.n

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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # input expands to a flat vector
        self.fc1 = nn.Linear(state_n, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, action_n)

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
            return random.randint(0, action_n-1)
        else:
            return out[:, :-1].argmax().item()

def soft_update(q_target, q, soft):
    for target_param, param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_((1-soft)*target_param.data+soft*param.data)

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_values = q(s)
        q_a = q_values.gather(1, a)  # gather: select value according to the index
        # print(q_a.shape)
        max_q_idx = q_target(s_prime)[:, :-1].argmax(1).unsqueeze(1)
        max_q_primes = q_target(s_prime)
        max_q_prime = max_q_primes.gather(1, max_q_idx)
        # print(max_q_prime.shape)
        target = r + gamma * max_q_prime * done_mask
        loss_f = torch.nn.MSELoss()
        loss = loss_f(q_a, target)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


q = QNetwork()
q_target = QNetwork()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)


def plot(score, mean):
    plt.figure()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score, color='r')
    plt.plot(mean, color='blue')
    # plt.text(len(score) - 1, score[-1], str(score[-1]))
    # plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show()

scores = []
mean = []
steps = 0

for episode in range(1000):
    s = env.reset()
    score = 0
    while True:
        # env.render()
        epsi = epsi_low + (epsi_high - epsi_low) * (math.exp(-1.0 * steps / decay))
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsi)
        steps += 1
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        if done:
                r = -1

        memory.put((s, a, r, s_prime, done_mask))

        if done:
            break

        score += r
        s = s_prime
        if memory.size() > batch_size:
            # train(q, q_target, memory, optimizer)
            train(q, q_target, memory, optimizer)
            soft_update(q_target, q, soft)
    print(episode, ': ', score)

    scores.append(score)
    mean.append(sum(scores[-100:]) / 100)

state = {'net':q.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':n_episode, 'lr':learning_rate}
torch.save(state, 'saves/dqn.dat')
plot(scores, mean)

from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))
Transition1 = namedtuple('Transition1', ('loss'))


class Memory(object):
    def __init__(self, algorithm=2):
        self.memory = []
        self.a = algorithm

    def push(self, *args):
        """Saves a transition."""
        if self.a == 1:
            self.memory.append(Transition1(*args))
        else:
            self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            if self.a == 1:
                return Transition1(*zip(*self.memory))
            else:
                return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            if self.a == 1:
                return Transition1(*zip(*random_batch))
            else:
                return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
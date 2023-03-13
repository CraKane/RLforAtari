
import torch
from utils import to_device
import numpy as np

def get_log_pi(s, a, theta):
    result = torch.zeros([2, 3])
    result[0][0] = -theta[0][0] * torch.pow(s[0], 2) - theta[0][1] * s[0] * s[1] - theta[0][2] * s[0] + a[0] * s[0]
    result[0][1] = -theta[0][1] * torch.pow(s[1], 2) - theta[0][0] * s[0] * s[1] - theta[0][2] * s[1] + a[0] * s[1]
    result[0][2] = -theta[0][2] - theta[0][0] * s[0] - theta[0][1] * s[1] + a[0]
    result[1][0] = -theta[1][0] * torch.pow(s[0], 2) - theta[1][1] * s[0] * s[1] - theta[1][2] * s[0] + a[1] * s[0]
    result[1][1] = -theta[1][1] * torch.pow(s[1], 2) - theta[1][0] * s[0] * s[1] - theta[1][2] * s[1] + a[1] * s[1]
    result[1][2] = -theta[1][2] - theta[1][0] * s[0] - theta[1][1] * s[1] + a[1]
    return result


def estimate_net_grad(states, actions, rewards, masks, values, gamma, tau, device, theta, next_values=None):
    # these computations would be performed on CPU
    if values != None:
        rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    else:
        rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, torch.tensor(1))
    tensor_type = type(rewards)

    """ ESTIMATE RETURNS"""
    returns = []
    R = 0
    eps = np.finfo(np.float32).eps.item()
    saved_log_pi = []
    for s, a in zip(states, actions):
        saved_log_pi.append(get_log_pi(s, a, theta))

    rewards = torch.flip(rewards, dims=[0])
    for r, msk, log_pi in zip(rewards, masks, saved_log_pi):
        R = r + gamma * R * msk
        returns.insert(0, R)  # insert R into index 0

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # regularization
    # print(log_pis.shape)
    if values != None and next_values != None:
        Advantages = values
    elif values != None:
        Advantages = returns - values
    else:
        Advantages = returns
    """ ESTIMATE NET GRADIENT"""

    grads = []
    for log_prob, R in zip(saved_log_pi, Advantages):
        grads.append((log_prob * R).detach().numpy())

    grad = torch.from_numpy(np.array(grads)).mean(0)
    # Roughly normalize the gradient
    grad = grad / (torch.norm(grad) + 1e-8)

    return_ = to_device(device, grad)
    return return_[0], returns

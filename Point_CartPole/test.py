import argparse
import gym
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
from utils import *
from torch import nn
from agent import Agent as Agent
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models.mlp_critic import Value, Critic
from models.mlp_policy_disc import DiscretePolicy
import torch.nn.functional as F
from agent import point_get_action

from point_env import PointEnv
from solutions.point_mass_solutions import estimate_net_grad

parser = argparse.ArgumentParser(description='Pytorch Policy Gradient')
parser.add_argument('--env-name', default="Point-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=0.01, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seeds', type=list, default=[1, 10, 100, 2021, 10000],
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=2000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=2000, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--algorithm', type=int, default=1, metavar='N',
                    help='choose the algorithm you need (default: 1)(choice: [1, 2, 3, 4])')
args = parser.parse_args()

"""save file path"""
directory_for_saving_files = "."

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
if args.env_name == 'CartPole-v0':
    running_state = ZFilter((state_dim,), clip=5)
theta = torch.normal(0, 0.01, size=(action_dim, state_dim + 1))

"""cuda setting"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)

def load(critic):
    with open("learned_models/dqn.dat", 'rb') as f:
        state = torch.load(f)
    critic.load_state_dict(state['net'])

def plot(scores, mean):
    plt.figure()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    for score in scores:
        plt.plot(score, color='#66CCEE')
        plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.plot(mean, color='#60F09A')
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show()

def main_loop():
    scores = []
    mean = []
    for i in args.seeds:

        """seeding"""
        np.random.seed(i)
        torch.manual_seed(i)
        env.seed(i)

        """define actor and critic"""
        if args.env_name == 'Point-v0':
            # we use only a linear policy for this environment
            global theta
            theta = torch.normal(0, 0.01, size=(action_dim, state_dim + 1))
            policy_net = None
            value_net = Value(state_dim)
            theta = theta.to(dtype).to(device)
            to_device(device, value_net)
            optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
        else:
            # we use both a policy and a critic network for this environment
            policy_net = DiscretePolicy(state_dim, env.action_space.n)
            theta = None
            value_net = Value(state_dim)
            critic = Critic(state_dim, 256, env.action_space.n)
            load(critic)
            to_device(device, policy_net, value_net, critic)

            # Optimizers
            optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
            optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)


        """create agent"""
        if args.env_name == 'Point-v0':
            agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                      running_state=None, num_threads=args.num_threads)
        else:
            agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                      running_state=running_state, num_threads=args.num_threads)

        def update_params(batch, i_iter, arg):
            states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
            actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
            next_states = torch.from_numpy(np.stack(batch.next_state)).to(dtype).to(device)
            rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
            masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
            if args.env_name == 'CartPole-v0':
                log_pis = policy_net.get_log_prob(states, actions)
                if args.algorithm == 3:
                    values = value_net(states).squeeze()
                returns = []
                return_ = []
                R = 0
                eps = np.finfo(np.float32).eps.item()
                rewards = torch.flip(rewards, dims=[0])
                for r, msk in zip(rewards, masks):
                    R = r + R * msk
                    return_.insert(0, R)  # insert R into index 0
                    if msk == 0:
                        return_ = torch.tensor(return_)
                        return_ = (return_ - return_.mean()) / (return_.std() + eps)  # regularization
                        returns.append(return_)
                        R = 0
                        return_ = []

                returns = torch.cat(returns)
                optimizer_policy.zero_grad()

                if args.algorithm == 3:
                    advantages = returns - values
                    optimizer_value.zero_grad()
                    value_losses = F.smooth_l1_loss(values, returns)
                    policy_loss = (-log_pis * advantages).mean().requires_grad_(True)  # mean
                    value_losses.backward(retain_graph=True)
                elif args.algorithm == 1:
                    policy_loss = (-log_pis * returns).mean().requires_grad_(True)  # mean
                elif args.algorithm == 2:
                    # print(states.shape, actions.view(actions.shape[0], -1).shape)
                    qs = critic(states).gather(1, actions.long().view(actions.shape[0], -1))
                    # print(qs.shape)
                    policy_loss = (-log_pis * qs).mean().requires_grad_(True)  # mean

                policy_loss.backward()
                optimizer_policy.step()
                optimizer_value.step()
            else:

                """get values estimates from the trajectories"""

                next_values = None
                global theta
                if args.algorithm == 1:
                    values = None
                else:
                    values = value_net(states).squeeze()
                    if args.algorithm == 2:
                        next_values = value_net(next_states).squeeze()
                        nq = rewards + args.gamma*next_values
                net_grad, returns = estimate_net_grad(states, actions, rewards, masks, values, args.gamma, args.tau, device,
                                             theta, next_values)

                if args.algorithm != 1:
                    optimizer_value.zero_grad()
                    if args.algorithm == 2:
                        value_losses = F.smooth_l1_loss(values, nq)
                    else:
                        value_losses = F.smooth_l1_loss(values, returns)
                    value_losses.backward()
                    optimizer_value.step()

                """update policy parameters"""
                theta += args.learning_rate * net_grad

                agent.theta = theta

        score = []

        for i_iter in range(args.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = agent.collect_samples(args.min_batch_size, render=args.render)

            t0 = time.time()
            update_params(batch, i_iter, args)
            t1 = time.time()
            """evaluate with determinstic action (remove noise for exploration)"""
            _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
            t2 = time.time()

            score.append(log['avg_reward'])
            if i == 1:
                mean.append(np.mean(score[-10:]))
            if i_iter % args.log_interval == 0:
                print('{}\tTime_sample {:.4f}\tTime_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, log['avg_reward'], log_eval['avg_reward']))

            if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
                if args.env_name == 'CartPole-v0':
                    to_device(torch.device('cpu'), policy_net, value_net)
                    pickle.dump((policy_net, value_net), open(
                        os.path.join(directory_for_saving_files, 'learned_models/{}_policy_grads.p'.format(args.env_name)),
                        'wb'))
                    to_device(device, policy_net, value_net, critic)
                else:
                    to_device(torch.device('cpu'), value_net)
                    pickle.dump((value_net), open(
                        os.path.join(directory_for_saving_files, 'learned_models/{}_policy_grads.p'.format(args.env_name)),
                        'wb'))
                # you will have to specify a proper directory to save files

            scores.append(score)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

    plot(scores, mean)

main_loop()
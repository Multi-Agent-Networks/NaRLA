import sys
import gym
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ion()

############## REPRODUCIBILITY ############
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###########################################

from environment import Env
from multi_agent import Network

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
parser.add_argument('--neuron_type', default='DQN')
parser.add_argument('--render', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_neurons', type=int, default=15)
parser.add_argument('--repeat_num', default=0, type=int)
parser.add_argument('--update_every', type=int, default=1)
parser.add_argument('--env', default='binary',choices=['binary', 'cart', 'mount'])
parser.add_argument('--update_type', default='async', choices=['sync', 'async'])
parser.add_argument('--train_last', action='store_true', help='Only train the last neuron')
parser.add_argument('--reward_type', default='task', choices=['all','task','bio','bio_then_all'])
args = parser.parse_args()

# CREATE ENVIRONMENT
if args.env == 'cart': env = gym.make('CartPole-v1')
elif args.env == 'mount': env = gym.make('MountainCar-v0')
else: sys.exit(); env = Env()

# BUILD NETWORK AND SET ENV THRESHOLD
network        = Network(args, input_space=env.observation_space.shape, num_outputs=env.action_space.n)
stop_threshold = 75 if args.env == 'binary' else 300

for e in range(1,20000):
    done      = False
    state     = env.reset()
    ep_reward = 0

    while not done:
        # take action
        action = network.forward(state)
        # if args.env == 'mount' and action == 1: action += 1
        next_state, reward, done, _ = env.step(action)

        ep_reward += reward

        # distribute reward to all neurons
        network.distribute_task_reward(reward)

        # must come after distribute
        network.store(False)

        if args.render and e // 250 == 1: env.render()

        state = next_state

    if args.neuron_type == 'DQN':
        # hack to get state,next_state to play nicely in DQN
        action = network.forward(next_state)
        # distribute reward to all neurons
        network.distribute_task_reward(reward)
        # must come after distribute
        network.store(done=True)

    # save episode rewards
    network.end_episode(
        ep_reward
    )

    # learn every X episodes
    if e % args.update_every == 0:
        network.learn()

    # TRAIN USING BIO REWARD THEN ADD TASK REWARD
    if args.reward_type == 'bio_then_all' and e > 500:
        print('Switching to task reward...')
        network.args['reward_type'] = 'all'

    if args.plot: network.plot()

    if args.verbose and e % 10 == 0:
        print('Episode:', e, 'Reward:', np.mean(network.episode_rewards[-10:]))

    if np.mean(network.episode_rewards[-100:]) > 500 and args.env == 'binary': break
    
    # stop when agent has sufficiently learned
    if e > 1000 and np.mean(network.episode_rewards[-100:]) > stop_threshold: break

network.save()

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from torch.distributions import Categorical

CUDA = 0
default_args = {'reward_type': 'task', 'env': 'cartpole'}


class Agent(nn.Module):
    def __init__(self, in_shape, softmax=True, num_layers=2, num_outputs=2):
        super(Agent, self).__init__()

        layers = []
        out_shape = 32
        for i in range(num_layers):
            if i + 1 == num_layers:
                out_shape = num_outputs

            layers.append(
                nn.Linear(in_shape, out_shape)
            )

            # NOT LAST LAYER ADD ACTIVATION FUNCTION
            if i + 1 != num_layers:
                layers.append(
                    nn.LeakyReLU(.3)
                )
            in_shape = out_shape

        if softmax:
            layers.append(
                nn.Softmax(dim=1)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Neuron:
    def __init__(self):
        pass

    def _process_state(self, state):
        # MAKING SURE SHAPES AND TENSORS MATCH UP
        if state.dtype == np.int or state.dtype == np.float:
            state = torch.from_numpy(state).float()
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if CUDA: return state.cuda()
        return state

    def store_reward(self, r, type):
        self.episode_rewards[type].append(r)
        if type == 'firing':
            # CALCULATE TRACE BASED ON PREVIOUS FIRING
            x = abs(
                np.mean(self.episode_rewards['firing']) - .5
            )
            R = -1746463 + (0.9256812 - -1746463)/(1 + (x/236.1357)**2.160334)
            self.episode_rewards['trace'].append(R)

    def _reset_episode_storage(self):
        if hasattr(self, 'actions'):
            if self.actions != []:
                self.old_actions = self.actions

        self.states = []
        self.actions = []

        self.episode_rewards = {
            'task'       : [],
            'sparsity'   : [],
            'firing'     : [],
            'prediction' : [],
            'trace'      : [],
        }

    def _reset_reward_storage(self):
        self.rewards = {
            'task'       : [],
            'sparsity'   : [],
            'firing'     : [],
            'prediction' : [],
            'trace'      : [],
        }

        self._reset_episode_storage()

        self.returns = []

    def learn(self):
        return 0

    def store(self, done):
        pass

    def calculate_reward(self, idx):
        # SUM REWARDS FROM DIFFERENT FEATURES
        if self.args['reward_type'] == 'all':
            reward = sum(
                v[idx] for v in self.episode_rewards.values()
            )
        elif self.args['reward_type'] == 'bio' or self.args['reward_type'] == 'bio_then_all':
            reward = sum(
                self.episode_rewards[k][idx] for k in ['prediction', 'sparsity', 'firing', 'trace']
            )
        else:
            reward = self.episode_rewards['task'][idx]

        return reward

    def end_episode(self):

        # SAVE METRICS FROM THE EPISODE
        for k, v in self.episode_rewards.items():
            # self.rewards[k].extend(v)
            self.rewards[k].append(np.mean(v))

        # RESET EPISODE STORAGE
        self._reset_episode_storage()


class Random(Neuron):
    def __init__(self, in_shape, num_outputs=2, ID=0, args=default_args):
        self.ID = ID
        self.num_outputs = num_outputs
        self._reset_reward_storage()

    def act(self, state):
        self.actions.append(
            np.random.randint(self.num_outputs)
        )
        return self.actions[-1]

class DQN(Neuron):
    def __init__(self, in_shape, num_outputs=2, ID=0, args=default_args, target=True):
        self.ID            = ID
        self.args          = args
        self.count         = 1
        self.memory        = []
        self.target        = target
        self.batch_size    = 128

        self.epsilon       = 0.99
        self.epsilon_decay = 0.9999
        self.gamma         = 0. if args['env'] == 'binary' else 0.99

        self.neuron        = Agent(
            in_shape,
            softmax=False,
            num_outputs=num_outputs
        )
        self.target_neuron = deepcopy(self.neuron) if target else self.neuron

        if CUDA: self.neuron.cuda()
        self.opt = optim.Adam(
            self.neuron.parameters(),
            lr=1e-2  # /float(args['update_every'])
        )

        self._reset_reward_storage()

    def _sample_memory(self):
        if len(self.memory) > self.batch_size:
            return random.sample(self.memory, self.batch_size)

        return self.memory

    def store(self, done):
        if len(self.states) == 1: return

        # SUM REWARDS FROM DIFFERENT FEATURES
        reward = self.calculate_reward(idx=-2)

        # CONVERT TO TENSOR - CHECK DIMS
        state = self._process_state(self.states[-2])
        next_state = self._process_state(self.states[-1])

        # STORE INFORMATION
        self.memory.append(
            [state, self.actions[-2], next_state, done, reward]
        )

        if len(self.memory) > 2000:
            self.memory.pop(0)

    def act(self, state):
        self.states.append(state)

        # CONVERT TO TENSOR - CHECK DIMS
        state = self._process_state(state)

        # SAMPLE RANDOM ACTION
        if np.random.uniform() < self.epsilon:
            action = np.random.choice([0, 1])
        # CHOOSE ACTION FROM Q-NETWORK
        else:
            with torch.no_grad():
                qvalues = self.neuron(state)
                action  = int(torch.argmax(qvalues))

        # DECAY EPSILON - IF GREATER THAN MIN EPSILON VAL
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        self.actions.append(action)

        return action

    def learn(self):
        total_loss = 0
        trials = self._sample_memory()

        # RUN MULTIPLE EPOCHS OF TRAINING
        for i in range(1):
            loss = 0
            for state, action, next_state, done, reward in trials:
                qvalues = self.neuron(state)[0][action]
                if not done:
                    with torch.no_grad():
                        q_prime = torch.max(
                            self.target_neuron(next_state)
                        )
                else:
                    q_prime = 0

                q_target = torch.Tensor([reward + q_prime * self.gamma]).squeeze()
                if CUDA: q_target = q_target.cuda()
                loss += F.smooth_l1_loss(qvalues, q_target)

            self.opt.zero_grad()
            loss.backward()
            for param in self.neuron.parameters():
                param.grad.data.clamp_(-1, 1)
            self.opt.step()

            total_loss += loss.item()
        self.count += 1
        if self.count % 100 == 0 and self.target:
            self.target_neuron = deepcopy(self.neuron)

        return total_loss


class PG(Neuron):
    def __init__(self, in_shape, num_outputs=2, ID=0, args=default_args):
        self.ID              = ID
        self.args            = args
        self.saved_log_probs = []
        self.gamma           = 0. if args['env'] == 'binary' else 0.99

        self.neuron          = Agent(in_shape, num_outputs=num_outputs)
        if CUDA: self.neuron.cuda()
        self.opt             = optim.Adam(
            self.neuron.parameters(),
            lr=1e-2 #/float(args['update_every'])
        )

        self._reset_reward_storage()

    def act(self, state):
        # CONVERT TO TENSOR - CHECK DIMS
        state = self._process_state(state)

        # ACTION PROBABILITIES
        probs    = self.neuron(state)
        cat_dist = Categorical(probs)

        # SAMPLE ACTION FROM DISTRIBUTION
        action   = cat_dist.sample()
        # STORE ACTION
        self.actions.append(action.item())
        # STORE LOG PROBS
        self.saved_log_probs.append(
            cat_dist.log_prob(action)
        )
        return action.item()

    def learn(self):
        policy_loss = []

        for log_prob, R in zip(self.saved_log_probs, self.returns):
            # WEIGHT LOG PROBABILITIES BY DISCOUNTED REWARD
            policy_loss.append(
                -log_prob * R
            )

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.opt.step()

        self.returns = []
        self.saved_log_probs = []
        # self._reset_reward_storage()

        return policy_loss.item()

    def _discount_rewards(self):
        R       = 0
        returns = []
        eps     = np.finfo(np.float32).eps.item()

        # DISCOUNTING REWARDS BY GAMMA
        for r in self._iter_rewards():
            R = r + self.gamma * R
            returns.insert(0, R)

        # STANDARDIZE REWARDS
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        return returns

    def _iter_rewards(self):
        for i in reversed(range(len(self.episode_rewards['task']))):
            # SUM REWARDS FROM DIFFERENT FEATURES
            yield self.calculate_reward(idx=i)

    def end_episode(self):
        # DISCOUNT REWARDS FOR THE EPISODE
        self.returns.extend(
            self._discount_rewards()
        )

        # SAVE METRICS FROM THE EPISODE
        for k,v in self.episode_rewards.items():
            # self.rewards[k].extend(v)
            self.rewards[k].append(np.mean(v))

        # RESET EPISODE STORAGE
        self._reset_episode_storage()

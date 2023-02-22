import os
import torch
import fcntl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import sample
from agent import PG, DQN, Random

colors = {
    'loss': 'brown',
    'task': 'orange',
    'sparsity': 'blue',
    'firing': 'red',
    'trace': 'purple',
    'prediction': 'green',
}
class Layer:
    def __init__(self, args, in_shape=32, out_shape=32, ID=0):
        self.args      = args
        self.ID        = ID         # LAYER ID
        self.in_shape  = in_shape   # NUMBER OF INCOMING SIGNALS
        self.out_shape = out_shape  # NUMBER OF OUTGOING SIGNALS - ALSO NUM NEURONS
        self.neurons   = []
        self.loss      = []
        self.fig       = None
        self.rewards   = {
            'task'       : [],
            'sparsity'   : [],
            'firing'     : [],
            'trace'      : [],
        }
        # LAST LAYER HAS NO PREDICTION REWARD
        if self.ID != self.args['num_layers']:
            self.rewards['prediction'] = []

        self._build_layer()

    def _get_all_rewards(self):
        all_rewards = np.array([
            v for v in self.rewards.itervalues()
        ])
        return np.mean(all_rewards, axis=0)

    def store(self, done):
        for neuron in self.neurons:
            neuron.store(done)

    def end_episode(self):
        for neuron in self.neurons:
            neuron.end_episode()

    def plot(self, bar_width=.35):
        # FIGURE SETUP
        if self.fig is None:
            self.fig   = plt.figure()
            self.ax    = self.fig.add_subplot(211)
            self.lines = {}

            # CREATE LINES ; ONE FOR EACH REWARD
            for k,v in self.rewards.iteritems():
                line,         = self.ax.plot(v, label=k, color=colors[k])
                self.lines[k] = line

            self.ax.set_xlabel('Episode'); self.ax.set_ylabel('Reward')
            # BOLD LINE FOR AVERAGE REWARD
            all_rewards       = self._get_all_rewards()
            line,             = self.ax.plot(all_rewards, label='Avg', linewidth=4, color='black')
            self.lines['avg'] = line

            # line,             = self.ax.plot(self.loss, label='Loss', linewidth=4)
            # self.lines['loss'] = line

            self.ax2 = self.fig.add_subplot(212)
            zeros, ones = self._get_neuron_activity()
            self.ax2.bar(np.arange(len(zeros))-bar_width/2., zeros, bar_width)
            self.ax2.bar(np.arange(len(ones))+bar_width/2., ones, bar_width)

        # UPDATE LINES WITH NEW VALUES
        for k,v in self.rewards.iteritems():
            self.lines[k].set_data(range(len(v)), v)

        # UPDATE AVERAGE REWARD LINE
        all_rewards = self._get_all_rewards()
        self.lines['avg'].set_data(range(len(v)), all_rewards)
        # self.lines['loss'].set_data(range(len(self.loss)),self.loss)

        zeros, ones = self._get_neuron_activity(); self.ax2.clear()
        self.ax2.bar(np.arange(len(zeros))-bar_width/2., zeros, bar_width)
        self.ax2.bar(np.arange(len(ones))+bar_width/2., ones, bar_width)
        self.ax2.set_ylim(0, 1)

        # AXIS LIMITS
        min_y_val = np.min([np.min(v) for v in self.rewards.itervalues()]) - 1
        max_y_val = np.max([np.max(v) for v in self.rewards.itervalues()]) + 1

        self.ax.legend(loc='upper left')
        self.ax.set_ylim(min_y_val, max_y_val)
        self.ax.set_xlim(0, len(v))
        # DYNAMICALLY UPDATE PLOT
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_neuron_activity(self):
        zeros = []; ones = []
        for neuron in self.neurons:
            zeros.append(
                (len(neuron.old_actions) - sum(neuron.old_actions)) / float(len(neuron.old_actions))
            )
            ones.append(
                sum(neuron.old_actions) / float(len(neuron.old_actions))
            )
        return zeros, ones

    def _build_layer(self):
        # ALL NEURONS ARE BINARY EXCEPT OUTPUT
        num_outputs = self.args['num_outputs'] if self.ID == self.args['num_layers'] else 2

        for ID in range(self.out_shape):
            if self.args['neuron_type'] == 'PG':
                neuron = PG(
                    args=self.args,
                    in_shape=self.in_shape,
                    ID=ID,
                    num_outputs=num_outputs
                )
            elif self.args['neuron_type'] == 'DQN':
                neuron = DQN(
                    args=self.args,
                    in_shape=self.in_shape,
                    ID=ID,
                    num_outputs=num_outputs
                )
            else:
                neuron = Random(
                    args=self.args,
                    in_shape=self.in_shape,
                    ID=ID,
                    num_outputs=num_outputs
                )

            self.neurons.append(neuron)

    def forward(self, X):
        output = []
        for i, neuron in enumerate(self.neurons):
            output.append(
                neuron.act(X[:,i])
            )

        return torch.Tensor(output).unsqueeze(0)

    def reward_neurons(self, R, type):
        # SEND SCALAR TASK REWARD TO ALL NEURONS
        if type == 'task': R = [R] * len(self.neurons)

        # SEND REWARDS BASED ON ACTIVITY TO NEURONS
        for r,neuron in zip(R, self.neurons):
            # if type == 'firing': r = 0.
            neuron.store_reward(r, type)

    def learn(self, loss=[]):
        rewards = {
            'task'       : [],
            'sparsity'   : [],
            'firing'     : [],
            'prediction' : [],
            'trace'      : [],
        }

        # UPDATE POLICY FOR EACH NEURON
        for neuron in self.neurons:
            # STORE REWARD HISTORY FROM NEURON
            for k in self.rewards:
                rewards[k].append(
                    # AVG REWARD FROM LAST EPISODE
                    neuron.rewards[k][-1]
                )
            # ONLY NEURONS IN THE LAST LAYER WILL LEARN
            if self.args['train_last'] and self.ID != self.args['num_layers']: continue

            # STORE LOSS FROM TRAINING
            loss.append(
                neuron.learn()
            )

        # AVERAGE LAYER LOSS
        self.loss.append(
            np.mean(loss)
        )
        # AVERAGE REWARDS OVER ALL NEURONS
        for k in self.rewards:
            self.rewards[k].append(
                np.mean(rewards[k])
            )


class Network:
    def __init__(self, args, input_space, num_outputs=2, sparsity=0.2):
        self.args            = vars(args)
        self.layers          = []
        self.neurons         = []
        self.sparsity_goal   = sparsity
        self.connections     = []
        self.episode_rewards = []
        self.fig             = None
        self.bio_then_all    = False
        
        if len(input_space) == 1:
            layer_descr          = [input_space[0], 1]

        if self.args['reward_type'] == 'bio_then_all':
            self.bio_then_all = True

        self.args['num_outputs'] = num_outputs

        # ADD NEURONS FOR EACH ADDITIONAL LAYER
        for i in range(args.num_layers):
            layer_descr.insert(1, self.args['num_neurons'])

        self._build_network(layer_descr)

    def end_episode(self, R):
        # STORE REWARD ACHIEVED IN EPISODE
        self.episode_rewards.append(R)

        # END EPISODE FOR LAYERS
        for layer in self.layers:
            layer.end_episode()

    def _build_network(self, layer_descr):
        print(layer_descr)
        for i in range(len(layer_descr) - 1):
            # FIRST LAYER IS FULLY CONNECTED
            if i == 0:
                connections = torch.ones(
                    layer_descr[i],
                    layer_descr[i+1]
                )
            # LATER LAYERS HAVE LOCAL CONNECTIVITY
            else:
                connections = torch.zeros(
                    layer_descr[i],         # FROM THE INPUT        - ROWS
                    layer_descr[i+1]        # TO THE OUTPUT NEURON  - COLUMNS
                )

                # SET COL CORRESPONDING TO NEURONs INPUT CONNECTIVITY
                if layer_descr[i+1] == 1:
                    connections[:,:] = 1.
                else:
                    w = layer_descr[i+1] / float(layer_descr[i])

                    info = np.ceil(np.arange(0,layer_descr[i+1], w)).astype(np.int)
                    # NOTE : I think a similar result can be achieved by setting the diagonals to 1.
                    for r,c in enumerate(info):
                        if r + 1 == len(info):
                            connections[r,c:] = 1.
                        else:
                            connections[r, c:info[r+1]+1] = 1.

            self.connections.append(
                connections
            )

            # CREATE NEW LAYER
            self.layers.append(
                Layer(
                    args=self.args,
                    in_shape=layer_descr[i],
                    out_shape=layer_descr[i+1],
                    ID=i
                )
            )

            # RECORD ALL NEURONS IN NETWORK
            self.neurons.extend(
                self.layers[-1].neurons
            )

    def store(self, done):
        for layer in self.layers:
            layer.store(done)

    def distribute_task_reward(self, R):
        # TASK LEVEL REWARD DISTRIBUTED ACROSS NETWORK
        for layer in self.layers:
            layer.reward_neurons(R, type='task')

    def learn(self):
        # SAMPLE ONE NEURON TO LEARN
        if self.args['update_type'] == 'async':
            neuron = sample(self.neurons, 1)[0]
            neuron.learn()

        # EACH LAYER IN THE NETWORK LEARNS
        else:
            for layer in self.layers:
                 layer.learn()

    def forward(self, X):
        X = torch.Tensor(X).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            # BROADCAST INPUT ACROSS NEURON CONNECTIONS
            X = X.t() * self.connections[i]

            # PASS INPUT THROUGH NEURONS IN LAYER
            O = layer.forward(X)

            # CAN'T REWARD THE ORIGINAL INPUT
            if i != 0:
                output = O.clone()
                output[output == 0] = -1.

                prediction_reward   = (X * output).sum(dim=-1)

                # REWARD PREVIOUS LAYER FOR PREDICTING CORRECTLY
                self.layers[i-1].reward_neurons(
                    prediction_reward,
                    type='prediction'
                )

            # DON'T REWARD THE LAST LAYER - SINGLE NEURON
            if i == self.args['num_layers']:
                layer.reward_neurons([int(O.squeeze())], type='firing')
                layer.reward_neurons([0.], type='sparsity')
            else:
                # REWARD LAYER FOR SPARSITY
                sparsity       = O.sum() / float(len(O.reshape(-1)))   # CALCULATE SPARSITY FOR LAYER
                sparsity_error = sparsity - self.sparsity_goal         # (+) NOT SPARSE ENOUGH (-) TOO SPARSE

                output              = O.clone().reshape(-1)
                output[output == 0] = -1.

                if sparsity_error > 0:
                    sparsity_rewards = -sparsity_error * output
                elif sparsity_error < 0:
                    sparsity_rewards =  sparsity_error * output
                elif sparsity_error == 0:
                    sparsity_rewards = output.abs() * self.sparsity_goal

                layer.reward_neurons(sparsity_rewards, type='sparsity')

                # REWARD LAYER FOR FIRING
                layer.reward_neurons(O.reshape(-1), type='firing')

            # OUTPUT OF PREV LAYER IS INPUT TO NEXT
            X = O

        # LAST LAYER CAN'T HAVE A PREDICTION REWARD
        self.layers[-1].reward_neurons(
            [0.] * self.layers[-1].out_shape,
            type='prediction'
        )

        return int(O.squeeze())

    def plot(self):
        # MAKE PLOTS FOR EACH LAYER
        for layer in self.layers:
            layer.plot()

        # FIGURE SETUP
        if self.fig is None:
            self.fig   = plt.figure()
            self.ax    = self.fig.add_subplot(111)
            self.line, = self.ax.plot(self.episode_rewards)

        # UPDATE LINES WITH NEW VALUES
        self.line.set_data(
            range(len(self.episode_rewards)),
            self.episode_rewards
        )
        # AXIS LIMITS
        self.ax.set_ylim(min(self.episode_rewards) - 5, max(self.episode_rewards) + 5)
        self.ax.set_xlim(0, len(self.episode_rewards))
        # DYNAMICALLY UPDATE PLOT
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, file_name='results.csv'):

        # STORE CURRENT RUN
        df = pd.DataFrame({'rewards': self.episode_rewards})
        for arg in self.args:
            df[arg] = self.args[arg]
        df['trials'] = range(len(self.episode_rewards))

        if self.bio_then_all:
            df['reward_type'] = 'bio_then_all'

        columns    = sorted(df.columns)
        format_str = ','.join('{}' for _ in columns) + '\n'

        with open(file_name, 'a+') as g:
            fcntl.flock(g, fcntl.LOCK_EX)

            if not g.readlines():
                g.write(','.join(columns) + '\n')

            for idx, row in df.iterrows():
                g.write(
                    format_str.format(
                        *row[columns].values
                    )
                )
            fcntl.flock(g, fcntl.LOCK_UN)

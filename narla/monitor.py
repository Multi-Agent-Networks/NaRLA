"""
Monitor
---
Records reward progress for nodes in MAN graph
Provides ability to live update plots
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from narla.settings import DISPLAY_REWARD_NAMES

class Monitor:
    def __init__(self, man, vis=False):
        self.man = man
        self.vis = vis
        self.reward_names = []
        self.episode_rewards = []

        self._init_plot()
        self._init_storage()

    def _init_storage(self):
        """
        Initialize storage data structure to monitor MAN activity and rewards
        The storage object is structured like this:

        self.storage -> [
            layer storage -> [
                node storage -> {
                    'name' : str
                    rewards : [] # TRACK REWARDS OVER TIME
                }
            ]
        ]
        """

        storage = []
        for layer_idx, layer in enumerate(self.man.layers):

            layer_storage = []
            for node_idx, node in enumerate(layer.nodes):
                # CREATE STORAGE OBJECT FOR EACH NODE
                node_storage = {'name' : node.name}
                for reward_name in DISPLAY_REWARD_NAMES:
                    node_storage[reward_name] = []

                if self.vis:
                    # MAKE A LINE FOR EACH NODE IN EACH REWARD PLOT
                    lines = []
                    for i, reward_name in enumerate(self.reward_names):
                        line, = self.axs[layer_idx][i].plot([], linewidth=.5)
                        lines.append(line)
                    node_storage['lines'] = lines

                # NODES IN LAYER
                layer_storage.append(node_storage)
            # LAYER IN MAN
            storage.append(layer_storage)

        self.storage = storage

    def _init_plot(self):
        """
        Initialize plotting objects

        self.fig : matplotlib.Figure
        self.axs : list[list]
            Containing ax for each layer and reward type
        self.reward_line : line
            Line object to update with task reward
        """
        if not self.vis:
            return

        if self.man.use_bio_rewards:
            num_rewards = len(DISPLAY_REWARD_NAMES)
            self.reward_names = sorted(DISPLAY_REWARD_NAMES.keys()) # 'Sparsity'
        else:
            num_rewards = 1
            self.reward_names = ['fire_reward']

        self.fig = plt.figure(figsize=(10, 20), constrained_layout=True)
        self.axs = []

        spec = self.fig.add_gridspec(
            nrows=self.man.num_layers + 2,
            ncols=num_rewards,
        )

        for row in range(self.man.num_layers):

            layer_axs = []
            for col, reward_name in enumerate(self.reward_names):

                ax = self.fig.add_subplot(spec[row, col])
                ax.set_xlim(0, 1000); ax.set_ylim(-1, 1)
                layer_axs.append(ax)

                if row == 0:
                    title = DISPLAY_REWARD_NAMES[reward_name]
                    layer_axs[-1].set_title(title, fontsize=20)

            layer_axs[0].set_ylabel(f'Layer {row+1}', fontsize=20)
            self.axs.append(layer_axs)

        reward_ax = self.fig.add_subplot(spec[self.man.num_layers:, :])
        reward_ax.set_title('Task Reward', fontsize=20)
        self.reward_line, = reward_ax.plot([])

        reward_ax.set_ylim(0, 500); reward_ax.set_xlim(0, 1000)

    def record(self):
        """
        Note: Must be called before man.learn()
        """
        self.episode_rewards.append(
            np.sum(self.man.rewards)
        )

        for layer_idx, layer in enumerate(self.man.layers):
            for reward_name in self.reward_names:
                reward = tf.reduce_mean(
                    tf.concat(
                        layer.bio_rewards[reward_name],
                        axis=0
                    ), axis=0
                ).numpy()

                for node_idx, node in enumerate(layer.nodes):
                    self.storage[layer_idx][node_idx][reward_name].append(reward[node_idx])

    def plot(self):
        for layer in self.storage:
            for node in layer:
                for line, reward_name in zip(node['lines'], self.reward_names):
                    line.set_data(
                        range(len(node[reward_name])),
                        node[reward_name]
                    )

    def log_status(self):
        if self.vis:
            self.plot()

            self.reward_line.set_data(
                range(len(self.episode_rewards)),
                self.episode_rewards
            )

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        print(
            f'Episode: {len(self.episode_rewards)}\
            25 Avg: {np.mean(self.episode_rewards[-25:])}\
            Best: {np.max(self.episode_rewards)}'
        )

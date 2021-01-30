"""
main.py

Example:
>>> python main.py --num_layers 5 --reward_type bio
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import time
import narla
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()

# PARSE NARLA SPECIFIC COMMAND LINE ARGS
args = narla.utils.parse_args()

tf.random.set_seed(123)
tf.keras.backend.set_floatx('float32')

# CREATE GYM ENV
env = gym.make("CartPole-v1")
state = tf.convert_to_tensor(
    env.reset().reshape(1, -1),
    dtype=tf.float32
)

# CREATE MAN WITH SETTINGS
man = narla.man.MultiAgentNetwork(
    input_size=4,
    num_layers=args.num_layers,
    num_nodes_per_layer=args.num_nodes,
    use_bio_rewards=args.reward_type == 'bio'
)
mon = narla.Monitor(man, vis=args.plot)

for episode in range(1, args.num_episodes):

    done = False
    state = narla.utils.prepare_state(env.reset())

    while not done:
        # FORWARD PASS THROUGH MAN
        action = man(state)

        # TAKE ACTION IN ENV
        next_state, reward, done, _ = env.step(action)

        # RECORD
        man.record_reward(reward)
        state = narla.utils.prepare_state(next_state)

        if done:
            mon.record()
            man.learn()
            
            if episode % args.update_every == 0:
                mon.log_status()

# Neurons as Reinforcement Learning Agents
This repository accompanies the paper [*Giving Up Control: Neurons as Reinforcement Learning Agents*](https://arxiv.org/abs/2003.11642)
```
@misc{ott2020giving,
    title={Giving Up Control: Neurons as Reinforcement Learning Agents},
    author={Jordan Ott},
    year={2020},
    eprint={2003.11642},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```
![](https://github.com/Multi-Agent-Networks/NaRLA/blob/master/Figures/network.png?raw=true)

### Set Up
```bash
git clone https://github.com/Multi-Agent-Networks/NaRLA.git
cd NaRLA
git checkout tf-dev
pip install -e .
```

### Simple Example
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import tensorflow as tf
from narla.man import MultiAgentNetwork

env = gym.make("CartPole-v1")
state = tf.convert_to_tensor(
    env.reset().reshape(1, -1),
    dtype=tf.float32
)

man = MultiAgentNetwork(
  input_size=4,
  num_layers=1,
  use_bio_rewards=False
)

action = man(state)        
next_state, reward, done, _ = env.step(action)

man.record_reward(reward)
man.learn()
```

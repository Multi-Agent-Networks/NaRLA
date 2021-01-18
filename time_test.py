import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from narla.man import MultiAgentNetwork

tf.keras.backend.set_floatx('float32')

times = []
learn_times = []
per_layer = list(range(1, 21, 3))
x = tf.random.uniform((1, 4), dtype=tf.float32)

f = open('time.out', 'w')
for p in per_layer:
    man = MultiAgentNetwork(input_size=4, num_layers=p, num_nodes_per_layer=15)
    output = man(x)
    man.record_reward(1)
    man.learn()

    start_time = time.time()
    for i in range(100):
        x = tf.convert_to_tensor(
            np.random.uniform(size=(1,4)),
            dtype=tf.float32
        )
        output = man(x)
        man.record_reward(1)

    times.append(
        time.time() - start_time
    )

    start_time = time.time()
    man.learn()
    learn_times.append(
        time.time() - start_time
    )

    print(f'{p} Train: {times[-1]} Learn: {learn_times[-1]}')

    f.write(f'{p} Train: {times[-1]} Learn: {learn_times[-1]}\n')
    f.flush()

plt.plot(per_layer, times, label='Forward Pass')
plt.plot(per_layer, learn_times, label='Backward Pass')
plt.legend()
plt.show()

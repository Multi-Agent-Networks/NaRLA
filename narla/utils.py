import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def pg_action_loss(prob, action, reward):
    dist = tfp.distributions.Categorical(
        probs=prob,
        dtype=tf.float32
    )

    log_prob = dist.log_prob(action)
    loss = -log_prob*reward

    return loss

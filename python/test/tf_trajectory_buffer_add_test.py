import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import numpy as np


data_spec =  (
        tf.TensorSpec([3], tf.float32, 'action'),
        (
            tf.TensorSpec([5], tf.float32, 'lidar'),
            tf.TensorSpec([3, 2], tf.float32, 'camera')
        )
)

batch_size = 32
max_length = 1000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec,
    batch_size=batch_size,
    max_length=max_length)

action = tf.constant(1 * np.ones(data_spec[0].shape.as_list(), dtype=np.float32))
lidar = tf.constant(2 * np.ones(data_spec[1][0].shape.as_list(), dtype=np.float32))
camera = tf.constant(3 * np.ones(data_spec[1][1].shape.as_list(), dtype=np.float32))

values = (action, (lidar, camera))
values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size), values)

replay_buffer.add_batch(values_batched)
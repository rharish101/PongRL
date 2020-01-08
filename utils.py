"""Utilities for the DQN."""
import random

import tensorflow as tf

PRE_CROP_SIZE = (110, 84)
IMG_SIZE = (84, 84)
SHIFT = 18  # the row from where to crop the frame
STATE_FRAMES = 4


class ReplayBuffer:
    """A class for the replay buffer as a limited length FIFO list."""

    def __init__(self, limit):
        """Initialize the buffer.

        Args:
            limit (int): The limit for the buffer.

        Raises:
            ValueError: If the limit is not positive

        """
        if limit <= 0:
            raise ValueError(
                "Buffer limit must be positive; got: {}".format(limit)
            )

        self.buffer = []
        self.limit = limit
        self.next = 0  # where to insert items if buffer is full

    def append(self, item):
        """Append the item to the buffer.

        If the buffer is not full, then the item will be appended. If it is
        full, then the item which was inserted earlier (by the index) will be
        overwritten.
        """
        if len(self.buffer) < self.limit:
            self.buffer.append(item)
        else:
            # `self.next` is initialized with 0. It points to the oldest item.
            self.buffer[self.next] = item
            # Increment `self.next`, as the next oldest item is the next one by
            # index. Once the limit is reached, it wraps around, as the oldest
            # item now is the one at the first index.
            self.next = (self.next + 1) % self.limit

    def __len__(self):
        """Return the total number of items in the buffer."""
        return len(self.buffer)

    def sample(self, *args, **kwargs):
        """Return a random sample from the buffer."""
        return random.sample(self.buffer, *args, **kwargs)


@tf.function
def preprocess(img):
    """Preprocess a frame."""
    img = tf.image.rgb_to_grayscale(img)
    # `tf.image.resize` converts to float32 in range [0, 255]. Cast to uint8 to
    # save space in the replay buffer.
    img = tf.cast(tf.image.resize(img, PRE_CROP_SIZE), tf.uint8)
    # Cropping from the `SHIFT` row, and removing the single channel
    img = img[SHIFT : (SHIFT + IMG_SIZE[0]), :, 0]
    return img


@tf.function
def choose(model, state, epsilon):
    """Choose an action wrt an epsilon-greedy policy.

    NOTE: The action choosing is non-differentiable.

    Args:
        model (`tf.keras.Model`): The DQN model
        state (`tf.Tensor`): The input state to the model as a 3D tensor
        epsilon (float): The epsilon for the epsion-greedy policy

    Returns:
        int: The action to be taken

    """
    # Convert from uint8 to float32
    inputs = tf.image.convert_image_dtype(state, tf.float32)
    pred = model(tf.expand_dims(inputs, axis=0))[0]  # not training
    random = tf.random.uniform([])
    if random < epsilon:
        action = tf.random.uniform(
            [], minval=0, maxval=len(pred), dtype=tf.int64
        )
    else:
        action = tf.argmax(pred)
    return action


def sample_replay(replay, batch_size):
    """Randomly sample transitions from the replay buffer.

    The replay should consist of tuples representing transitions which contain:
        `tf.Tensor`: A 3D uint8 tensor representing the initial state
        `tf.Tensor`: A 2D uint8 tensor representing the final frame (not state)
        int: The action taken
        float: The reward obtained
        bool: Whether the final frame is terminal

    Args:
        replay (`utils.ReplayBuffer`): The experience replay buffer
        batch_size (int): The no. of states to sample from the replay buffer at
            one instance

    Returns:
        `tf.Tensor`: The input states as a float32 batch
        `tf.Tensor`: The corresponding output states as a float32 batch
        `tf.Tensor`: The corresponding actions as a int64 batch
        `tf.Tensor`: The corresponding rewards as a float32 batch
        `tf.Tensor`: The corresponding terminal indicators as a bool batch

    """
    exp_sample = replay.sample(batch_size)
    inputs, outputs, actions, rewards, terminal = zip(*exp_sample)

    inputs = tf.stack(inputs, axis=0)
    actions = tf.stack(actions, axis=0)
    terminals = tf.stack(terminal, axis=0)

    # Quantize rewards, as per the paper
    rewards = tf.sign(tf.stack(rewards, axis=0))

    # Make it into a 4D tensor with a single channel for easy concatenation
    outputs = tf.expand_dims(tf.stack(outputs, axis=0), axis=-1)
    # Ignore the oldest frames in the inputs
    outputs = tf.concat([inputs[:, :, :, 1:], outputs], axis=-1)

    # Convert from uint8 to float32
    inputs = tf.image.convert_image_dtype(inputs, tf.float32)
    outputs = tf.image.convert_image_dtype(outputs, tf.float32)

    return inputs, outputs, actions, rewards, terminals

"""Utilities for the DQN."""
import random
from typing import Generic, List, Tuple, TypeVar

import tensorflow as tf
from typing_extensions import Final

PRE_CROP_SIZE: Final = (110, 84)
IMG_SIZE: Final = (84, 84)
SHIFT: Final = 18  # the row from where to crop the frame
STATE_FRAMES: Final = 4


BufItemType = TypeVar("BufItemType")
TransitionType = Tuple[tf.Tensor, tf.Tensor, int, float, bool]


class ReplayBuffer(Generic[BufItemType]):
    """A class for the replay buffer as a limited length FIFO list."""

    def __init__(self, limit: int):
        """Initialize the buffer.

        Args:
            limit: The limit for the buffer.

        Raises:
            ValueError: If the limit is not positive
        """
        if limit <= 0:
            raise ValueError(
                "Buffer limit must be positive; got: {}".format(limit)
            )

        self.buffer: List[BufItemType] = []
        self.limit = limit
        self.next = 0  # where to insert items if buffer is full

    def append(self, item: BufItemType) -> None:
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

    def __len__(self) -> int:
        """Return the total number of items in the buffer."""
        return len(self.buffer)

    def sample(self, k: int) -> List[BufItemType]:
        """Return a random sample from the buffer."""
        return random.sample(self.buffer, k)


@tf.function
def preprocess(img: tf.Tensor) -> tf.Tensor:
    """Preprocess a frame."""
    img = tf.image.rgb_to_grayscale(img)
    # `tf.image.resize` converts to float32 in range [0, 255]. Cast to uint8 to
    # save space in the replay buffer.
    img = tf.cast(tf.image.resize(img, PRE_CROP_SIZE), tf.uint8)
    # Cropping from the `SHIFT` row, and removing the single channel
    img = img[SHIFT : (SHIFT + IMG_SIZE[0]), :, 0]
    return img


@tf.function
def choose(model: tf.keras.Model, state: tf.Tensor, epsilon: float) -> int:
    """Choose an action wrt an epsilon-greedy policy.

    NOTE: The action choosing is non-differentiable.

    Args:
        model: The DQN model
        state: The input state to the model as a 3D tensor
        epsilon: The epsilon for the epsion-greedy policy

    Returns:
        The action to be taken
    """
    # Convert from uint8 to float32
    inputs = tf.image.convert_image_dtype(state, tf.float32)
    pred = model(tf.expand_dims(inputs, axis=0))[0]  # not training
    rand = tf.random.uniform([])
    if rand < epsilon:
        action = tf.random.uniform(
            [], minval=0, maxval=len(pred), dtype=tf.int64
        )
    else:
        action = tf.argmax(pred)
    return action


def sample_replay(
    replay: ReplayBuffer[TransitionType],
    batch_size: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Randomly sample transitions from the replay buffer.

    The replay should consist of tuples representing transitions which contain:
        * A 3D uint8 tensor representing the initial state
        * A 2D uint8 tensor representing the final frame (not state)
        * The action taken
        * The reward obtained
        * Whether the final frame is terminal

    Args:
        replay: The experience replay buffer
        batch_size: The no. of states to sample from the replay buffer at one
            instance

    Returns:
        The input states as a float32 batch
        The corresponding output states as a float32 batch
        The corresponding actions as a int64 batch
        The corresponding rewards as a float32 batch
        The corresponding terminal indicators as a bool batch
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

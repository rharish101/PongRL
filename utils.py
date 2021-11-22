"""Utilities for the DQN."""
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Deque, Optional, Tuple

import tensorflow as tf
import toml
from typing_extensions import Final

PRE_CROP_SIZE: Final = (110, 84)
IMG_SIZE: Final = (84, 84)
SHIFT: Final = 18  # the row from where to crop the frame
STATE_FRAMES: Final = 4
ENV_NAME: Final = "ALE/Pong-v5"


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        lr: The learning rate for the optimizer
        batch_size: The no. of states to sample from the replay buffer at one
            instance
        episodes: The max episodes to train the model
        init_epsilon: Initial value of epsilon for the epsilon-greedy policy
        min_epsilon: Lower bound for epsilon after decay
        decay_wait: No. of episodes to wait before decaying epsilon
        decay_eps: No. of episodes for epsilon decay
        discount: Discount factor for reward
        replay_size: The max size of the experience replay buffer
        frame_skips: How much frames to skip when running the environment
        reset_steps: Steps after which the fixed model is to be updated
        seed: The random seed for reproducibility
    """

    lr: float = 2.5e-4
    batch_size: int = 32
    episodes: int = 20000
    init_epsilon: float = 1.0
    min_epsilon: float = 0.01
    decay_wait: int = 1000
    decay_eps: int = 2000
    discount: float = 0.99
    replay_size: int = 100000
    frame_skips: int = 4
    reset_steps: int = 10000
    seed: Optional[int] = None


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path is None, then an empty dict is returned.
    """
    if config_path is not None:
        with open(config_path, "r") as f:
            args = toml.load(f)
    else:
        args = {}
    return Config(**args)


class ReplayBuffer(Deque[Tuple[tf.Tensor, tf.Tensor, int, float, bool]]):
    """Replay buffer for experience replay."""

    def __init__(self, config: Config):
        """Initialize the buffer.

        Args:
            config: The hyper-param config

        Raises:
            ValueError: If the limit is not positive
        """
        super().__init__(maxlen=config.replay_size)
        self.rng = Random(config.seed)

    def sample_tensors(
        self, batch_size: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Randomly sample transitions from the replay buffer.

        Args:
            batch_size: The no. of states to sample from the replay buffer

        Returns:
            The input states as a float32 batch
            The corresponding output states as a float32 batch
            The corresponding actions as a int64 batch
            The corresponding rewards as a float32 batch
            The corresponding terminal indicators as a bool batch
        """
        exp_sample = self.rng.sample(self, batch_size)
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

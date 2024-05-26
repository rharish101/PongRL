# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""DQN model."""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.random import Generator

from utils import IMG_SIZE, STATE_FRAMES, Config


class DQN(Model):
    """The DQN model."""

    def __init__(self, num_actions: int, config: Config):
        """Initialize the model."""
        super().__init__()
        self.rng = Generator.from_seed(config.seed)
        initializer = VarianceScaling(2.0, seed=config.seed)

        layers = [
            Conv2D(
                16,
                8,
                strides=4,
                activation="relu",
                kernel_initializer=initializer,
                input_shape=(*IMG_SIZE, STATE_FRAMES),
            ),
            Conv2D(
                32,
                4,
                strides=2,
                activation="relu",
                kernel_initializer=initializer,
            ),
            Flatten(),
            Dense(256, activation="relu", kernel_initializer=initializer),
            Dense(num_actions, kernel_initializer=initializer),
        ]

        self.model = Sequential(layers)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Get the model's predictions."""
        return self.model(inputs)

    @tf.function
    def choose_action(self, state: tf.Tensor, epsilon: float = 0.0) -> int:
        """Choose an action wrt an epsilon-greedy policy.

        NOTE: The action choosing is non-differentiable.

        Args:
            state: The input state to the model as a 3D tensor
            epsilon: The epsilon for the epsion-greedy policy

        Returns:
            The action to be taken
        """
        # Convert from uint8 to float32
        inputs = tf.image.convert_image_dtype(state, tf.float32)
        pred = self.model(tf.expand_dims(inputs, axis=0), training=False)[0]
        rand = self.rng.uniform([])
        if rand < epsilon:
            action = self.rng.uniform(
                [], minval=0, maxval=len(pred), dtype=tf.int64
            )
        else:
            action = tf.argmax(pred)
        return action

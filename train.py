#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Train the DQN for Pong."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import gym
import numpy as np
import tensorflow as tf
import toml
from tqdm import tqdm
from typing_extensions import Final

from model import DQN
from utils import (
    ENV_NAME,
    STATE_FRAMES,
    Config,
    PiecewiseLinearDecay,
    ReplayBuffer,
    load_config,
    preprocess,
)

CONFIG_NAME: Final = "config.toml"


class DQNTrainer:
    """Class to train a DQN for Pong."""

    MODEL_NAME: Final = "model.ckpt"
    FIXED_NAME: Final = "fixed.ckpt"
    OPTIM_NAME: Final = "optim.npz"
    DATA_NAME: Final = "data.toml"

    SAVE_DIR: Final = "checkpoints"
    VID_DIR: Final = "videos"

    def __init__(
        self,
        env: gym.Env,
        model: tf.keras.Model,
        config: Config,
        log_steps: int,
        video_eps: int,
        log_dir: Path,
    ):
        """Store the main model and other info.

        Args:
            env: The Atari Pong environment
            model: The model to be trained
            config: The hyper-param config
            log_steps: Steps after which model is to be logged
            video_eps: Episodes after which video is to be saved
            log_dir: Path where to save logs
        """
        # The Pong environment, with a video monitor attached
        self.env = gym.wrappers.RecordVideo(
            env,
            log_dir / self.VID_DIR,
            episode_trigger=lambda count: count % video_eps == 0,
        )

        # The main model
        self.model = model

        # DQN helpers
        self.fixed = tf.keras.models.clone_model(model.model)
        self.replay = ReplayBuffer(config)

        # Optimizer setup
        self.optimizer = tf.keras.optimizers.Adam(config.lr)

        # Other helpers
        self.loss_fn = tf.keras.losses.Huber()  # to avoid gradient explosion
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.eps_scheduler = PiecewiseLinearDecay(
            config.init_epsilon,
            config.min_epsilon,
            config.decay_wait,
            config.decay_eps,
        )

        # Track current position
        self.global_step = 0
        self.episode = 0

        # Hyperparams
        self.config = config
        self.log_steps = log_steps

        # Save directory setup
        self.save_dir = log_dir / self.SAVE_DIR
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

    def load_info(self, load_dir: Path) -> None:
        """Load models and training parameters.

        This also overrides the log directory.
        """
        # Reset all uses of the log directory
        self.env = gym.wrappers.RecordVideo(
            self.env.env,
            load_dir / self.VID_DIR,
            episode_trigger=self.env.episode_trigger,
        )
        self.writer = tf.summary.create_file_writer(str(load_dir))
        self.save_dir = load_dir / self.SAVE_DIR

        self.model.load_weights(self.save_dir / self.MODEL_NAME)
        self.fixed.load_weights(self.save_dir / self.FIXED_NAME)

        optim_weights = np.load(self.save_dir / self.OPTIM_NAME).values()
        # Optimizer variables can only be loaded after initialization
        self.optimizer.apply_gradients(
            (tf.zeros_like(var), var) for var in self.model.trainable_variables
        )
        self.optimizer.set_weights(optim_weights)

        with open(self.save_dir / self.DATA_NAME, "r") as data_file:
            data = toml.load(data_file)

        self.episode = data["episode"]
        self.global_step = data["global_step"]
        self.model.rng.reset(data["rng_state"])

        print("Loaded model and training data")

    def save_info(self) -> None:
        """Save models and training parameters."""
        self.model.save_weights(self.save_dir / self.MODEL_NAME)
        self.fixed.save_weights(self.save_dir / self.FIXED_NAME)
        np.savez(
            self.save_dir / self.OPTIM_NAME, *self.optimizer.get_weights()
        )

        data = {
            "episode": self.episode,
            "global_step": self.global_step,
            "rng_state": self.model.rng.state.numpy().tolist(),
        }
        with open(self.save_dir / self.DATA_NAME, "w") as data_file:
            toml.dump(data, data_file)

    @tf.function
    def exp_replay(
        self,
        inputs: tf.Tensor,
        outputs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        terminals: tf.Tensor,
    ) -> tf.Tensor:
        """Train the model on a random sample from the replay buffer.

        Args:
            inputs: The float32 initial states for the batch of transitions
            outputs: The float32 corresponding final states for the batch of
                transitions
            actions: The int64 corresponding actions for the batch of
                transitions
            rewards: The float32 corresponding rewards for the batch of
                transitions
            terminals: The bool corresponding terminal indicators for the batch
                of transitions

        Returns:
            The loss
        """
        with tf.GradientTape() as tape:
            q_initial = self.model(inputs, training=True)
            q_final_main = self.model(outputs, training=True)
            q_final_fixed = self.fixed(outputs, training=True)

            # If final state is terminal, then target is only the reward
            mask = tf.cast(tf.logical_not(terminals), tf.float32)
            # Double DQN: Choose target values based on fixed model's values
            # but main model's actions.
            batch_range = tf.range(actions.shape[0], dtype=tf.int64)
            tgt_indices = tf.stack(
                [batch_range, tf.argmax(q_final_main, axis=1)], axis=1
            )
            q_final = tf.gather_nd(q_final_fixed, tgt_indices)
            targets = rewards + mask * self.config.discount * q_final

            # Choose q-values based on actions taken
            pred_indices = tf.stack([batch_range, actions], axis=1)
            pred = tf.gather_nd(q_initial, pred_indices)
            loss = self.loss_fn(y_true=targets, y_pred=pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        # Needed for logging loss
        return loss

    def train_episode(self) -> Optional[tf.Tensor]:
        """Run one episode and train the model on it.

        Returns:
            The first state encountered
        """
        state = Deque[tf.Tensor](maxlen=STATE_FRAMES)
        state.append(preprocess(self.env.reset()))  # initial state

        first: Optional[tf.Tensor] = None

        while True:
            if len(state) < STATE_FRAMES:
                initial = None
                action = self.env.action_space.sample()
            else:
                initial = tf.stack(state, axis=-1)
                epsilon = self.eps_scheduler(self.episode)
                action = self.model.choose_action(initial, epsilon)

            state_new, reward, done, _ = self.env.step(action)
            state_new = preprocess(state_new)
            state.append(state_new)

            if initial is not None:
                # The inputs for this transition are well-defined, ie. a
                # proper x-frames state, so add it to the replay buffer.
                self.replay.append((initial, state_new, action, reward, done))
                if first is None:
                    first = initial

            if len(self.replay) >= self.config.batch_size:
                loss = self.exp_replay(
                    *self.replay.sample_tensors(self.config.batch_size)
                )

                if self.global_step % self.log_steps == 0:
                    with self.writer.as_default(), tf.name_scope("losses"):
                        tf.summary.scalar("loss", loss, step=self.global_step)

            if self.global_step % self.config.reset_steps == 0:
                self.fixed.set_weights(self.model.get_weights())

            self.global_step += 1

            if done:
                break

        # Needed for logging metrics
        return first

    def train(self, save_eps: int, resume: Optional[Path] = None) -> None:
        """Train the DQN on Pong.

        Args:
            save_eps: Episodes after which model and data are to be saved
            resume: The path from where to resume training
        """
        if resume is not None:
            self.load_info(resume)

        try:
            for _ in tqdm(
                range(self.episode, self.config.episodes),
                initial=self.episode,
                total=self.config.episodes,
            ):
                first = self.train_episode()
                self.episode += 1

                with self.writer.as_default(), tf.name_scope("metrics"):
                    first = tf.image.convert_image_dtype(first, tf.float32)
                    # Not training, but evaluation
                    pred = self.model(tf.expand_dims(first, axis=0))[0]
                    tf.summary.scalar(
                        "max q", tf.reduce_max(pred), step=self.episode
                    )

                if self.episode % save_eps == 0:
                    self.save_info()

        except KeyboardInterrupt:
            pass
        finally:
            self.save_info()


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    config = load_config(args.config)
    tf.keras.utils.set_random_seed(config.seed)

    # Automatically implements frame skipping internally
    env = gym.make(ENV_NAME, frameskip=config.frame_skips)
    env.seed(config.seed)

    model = DQN(env.action_space.n, config)

    # Save each run into a directory by its timestamp (ISO 8601 timezone-aware)
    time_stamp = datetime.now().astimezone().isoformat()
    log_dir = args.log_dir / time_stamp

    if not log_dir.exists():
        # Also creates log directory
        log_dir.mkdir(parents=True)

    with open(log_dir / CONFIG_NAME, "w") as conf:
        toml.dump(vars(config), conf)

    trainer = DQNTrainer(
        env,
        model,
        config=config,
        log_steps=args.log_steps,
        video_eps=args.video_eps,
        log_dir=log_dir,
    )

    trainer.train(args.save_eps, resume=args.resume)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train the DQN for Pong",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=100,
        help="steps after which model is to be logged",
    )
    parser.add_argument(
        "--video-eps",
        type=int,
        default=50,
        help="episodes after which video is to be saved",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="./logs/",
        help="path where to save logs",
    )
    parser.add_argument(
        "--save-eps",
        type=int,
        default=50,
        help="episodes after which model and data are to be saved",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume training from the model saved at the given path",
    )
    main(parser.parse_args())

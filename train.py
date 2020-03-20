#!/usr/bin/env python3
"""Train the DQN for Pong."""
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque
from datetime import datetime

import gym
import tensorflow as tf
import yaml
from gym.wrappers import Monitor  # gym.wrappers doesn't work
from tqdm import tqdm

from model import get_model
from utils import (
    IMG_SIZE,
    STATE_FRAMES,
    ReplayBuffer,
    choose,
    preprocess,
    sample_replay,
)

CONFIG_NAME = "config.yaml"


class DQNTrainer:
    """Class to train a DQN for Pong."""

    MODEL_NAME = "model.ckpt"
    FIXED_NAME = "fixed.ckpt"
    DATA_NAME = "data.pkl"

    def __init__(
        self,
        env,
        model,
        fixed,
        replay,
        optimizer,
        writer,
        batch_size,
        discount,
        reset_steps,
        log_steps,
        video_eps,
        log_dir,
        save_dir,
    ):
        """Store the main model and other info.

        Args:
            env (`gym.Wrapper`): The Atari Pong environment
            model (`tf.keras.Model`): The model to be trained
            fixed (`tf.keras.Model`): The model with fixed weights used for the
                Q-targets
            replay (`utils.ReplayBuffer`): The experience replay buffer
            optimizer (`tf.keras.optimizers.Optimizer`): The optimizer
            writer (`tf.summary.SummaryWriter`): The summary writer for saving
                logs
            batch_size (int): The no. of states to sample from the replay
                buffer at one instance
            discount (float): Discount factor for reward
            reset_steps (int): Steps after which the fixed model is to be
                updated
            log_steps (int): Steps after which model is to be logged
            video_eps (int): Episodes after which video is to be saved
            log_dir (str): Path where to save logs
            save_dir (str): Path where to save the model and data

        """
        # The Pong environment, with a video monitor attached
        self.env = Monitor(
            env,
            os.path.join(log_dir, "videos"),
            resume=False,  # don't retain older videos
            force=True,  # overwrite existing videos
            video_callable=lambda count: count % video_eps == 0,
        )

        # The main model
        self.model = model

        # DQN helpers
        self.fixed = fixed
        self.replay = replay

        # Other helpers
        self.optimizer = optimizer
        self.writer = writer

        # Hyperparams
        self.batch_size = batch_size
        self.discount = discount
        self.reset_steps = reset_steps
        self.log_steps = log_steps
        self.save_dir = save_dir

    def load_info(self):
        """Load models and training parameters.

        Returns:
            episode (int): The count of the current episode, previously
            epsilon (float): Current value of epsilon for the epsilon-greedy
                policy, previously

        """
        self.model.load_weights(os.path.join(self.save_dir, self.MODEL_NAME))
        self.fixed.load_weights(os.path.join(self.save_dir, self.FIXED_NAME))
        with open(os.path.join(self.save_dir, self.DATA_NAME), "rb") as data:
            start, epsilon = pickle.load(data)
        print("Loaded model and training data")
        return start, epsilon

    def save_info(self, episode, epsilon):
        """Save models and training parameters.

        Args:
            episode (int): The count of the current episode
            epsilon (float): Current value of epsilon for the epsilon-greedy
                policy

        """
        self.model.save_weights(os.path.join(self.save_dir, self.MODEL_NAME))
        self.fixed.save_weights(os.path.join(self.save_dir, self.FIXED_NAME))
        with open(os.path.join(self.save_dir, self.DATA_NAME), "wb") as data:
            pickle.dump((episode, epsilon), data)

    @tf.function
    def exp_replay(self, inputs, outputs, actions, rewards, terminals):
        """Train the model on a random sample from the replay buffer.

        Args:
            inputs (`tf.Tensor`): The float32 initial states for the batch of
                transitions
            outputs (`tf.Tensor`): The float32 corresponding final states for
                the batch of transitions
            actions (`tf.Tensor`): The int64 corresponding actions for the
                batch of transitions
            rewards (`tf.Tensor`): The float32 corresponding rewards for the
                batch of transitions
            terminals (`tf.Tensor`): The bool corresponding terminal indicators
                for the batch of transitions

        Returns:
            `tf.Tensor`: The loss

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
            targets = rewards + mask * self.discount * q_final

            # Choose q-values based on actions taken
            pred_indices = tf.stack([batch_range, actions], axis=1)
            pred = tf.gather_nd(q_initial, pred_indices)

            # Huber loss, to avoid gradient explosion
            loss = tf.keras.losses.Huber()(y_true=targets, y_pred=pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        # Needed for logging loss
        return loss

    def train_episode(self, epsilon, global_step):
        """Run one episode and train the model on it.

        Args:
            epsilon (float): Current value of epsilon for the epsilon-greedy
                policy
            global_step (int): The no. of frames processed so far

        Returns:
            `collections.deque`: The first state encountered
            int: The updated global step

        """
        state = deque(maxlen=STATE_FRAMES)
        state.append(preprocess(self.env.reset()))  # initial state

        first = None

        while True:
            if len(state) < STATE_FRAMES:
                initial = None
                action = self.env.action_space.sample()
            else:
                initial = tf.stack(state, axis=-1)
                action = choose(self.model, initial, epsilon)

            state_new, reward, done, _ = self.env.step(action)
            state_new = preprocess(state_new)
            state.append(state_new)

            if initial is not None:
                # The inputs for this transition are well-defined, ie. a
                # proper x-frames state, so add it to the replay buffer.
                self.replay.append((initial, state_new, action, reward, done))
                if first is None:
                    first = initial

            if len(self.replay) >= self.batch_size:
                loss = self.exp_replay(
                    sample_replay(self.replay, self.batch_size)
                )

            if global_step % self.log_steps == 0:
                with self.writer.as_default(), tf.name_scope("losses"):
                    tf.summary.scalar("loss", loss, step=global_step)

            if global_step % self.reset_steps == 0:
                self.fixed.set_weights(self.model.get_weights())

            global_step += 1

            if done:
                break

        # Needed for logging metrics
        return first, global_step

    def train(
        self,
        episodes,
        epsilon,
        min_epsilon,
        decay_wait,
        decay_eps,
        save_eps,
        start=0,
    ):
        """Train the DQN on Pong.

        Args:
            episodes (int): The max episodes to train the model
            epsilon (float): Initial value of epsilon for the epsilon-greedy
                policy
            min_epsilon (float): Lower bound for epsilon after decay
            decay_wait (int): No. of episodes to wait before decaying epsilon
            decay_eps (int): No. of episodes for epsilon decay
            save_eps (int): Episodes after which model and data are to be saved
            start (int): The starting episode (useful when resuming progress)

        """
        # Epsilon is decayed linearly for a few episodes, then kept constant
        if decay_wait <= start < decay_wait + decay_eps:
            shift = start - decay_wait
        else:
            # Either decay hasn't started, in which case shift is zero, or
            # decay is finished, in which case shift is kept zero as it might
            # cause div-by-zero.
            shift = 0
        epsilon_decay = (epsilon - min_epsilon) / (decay_eps - shift)

        global_step = 1

        try:
            ep = start  # in case start == episodes
            for ep in tqdm(
                range(start + 1, episodes + 1), initial=start, total=episodes
            ):
                first, global_step = self.train_episode(epsilon, global_step)

                with self.writer.as_default(), tf.name_scope("metrics"):
                    first = tf.image.convert_image_dtype(first, tf.float32)
                    # Not training, but evaluation
                    pred = self.model(tf.expand_dims(first, axis=0))[0]
                    tf.summary.scalar("max q", tf.reduce_max(pred), step=ep)

                if ep % save_eps == 0:
                    self.save_info(ep, epsilon)

                if decay_wait <= ep <= decay_wait + decay_eps:
                    epsilon -= epsilon_decay

        except KeyboardInterrupt:
            pass
        finally:
            self.save_info(ep, epsilon)


def main(args):
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    # Automatically implements frame skipping internally
    env = gym.make("Pong-v4", frameskip=args.frame_skips)

    model = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )

    fixed = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )
    replay = ReplayBuffer(limit=args.replay_size)

    # Save each run into a directory by its timestamp.
    # Remove microseconds and convert to ISO 8601 YYYY-MM-DDThh:mm:ss format.
    time_stamp = datetime.now().replace(microsecond=0).isoformat()
    log_dir = os.path.join(args.log_dir, time_stamp)

    for directory in log_dir, args.save_dir:
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, CONFIG_NAME), "w") as conf:
            yaml.dump(vars(args), conf)

    optimizer = tf.keras.optimizers.Adam(args.lr)
    writer = tf.summary.create_file_writer(log_dir)

    trainer = DQNTrainer(
        env,
        model,
        fixed,
        replay,
        optimizer,
        writer,
        batch_size=args.batch_size,
        discount=args.discount,
        reset_steps=args.reset_steps,
        log_steps=args.log_steps,
        video_eps=args.video_eps,
        log_dir=log_dir,
        save_dir=args.save_dir,
    )

    if args.resume:
        start, epsilon = trainer.load_info()
    else:
        fixed.set_weights(model.get_weights())
        start = 0
        epsilon = args.epsilon

    trainer.train(
        args.episodes,
        epsilon,
        args.min_epsilon,
        args.decay_wait,
        args.decay_eps,
        args.save_eps,
        start,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train the DQN for Pong",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="learning rate for Adam"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for sampling from the replay buffer",
    )
    parser.add_argument(
        "--episodes", type=int, default=20000, help="total epsiodes to train"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="initial value of epsilon for the epsilon-greedy policy",
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=0.01,
        help="lower bound for epsilon after decay",
    )
    parser.add_argument(
        "--decay-wait",
        type=int,
        default=1000,
        help="no. of episodes to wait before decaying epsilon",
    )
    parser.add_argument(
        "--decay-eps",
        type=int,
        default=2000,
        help="no. of episodes for epsilon decay",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="discount factor for reward",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=int(1e5),
        help="max size of experience replay buffer",
    )
    parser.add_argument(
        "--frame-skips", type=int, default=4, help="how much frames to skip"
    )
    parser.add_argument(
        "--reset-steps",
        type=int,
        default=10000,
        help="steps after which the fixed model is to be updated",
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
        type=str,
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
        "--save-dir",
        type=str,
        default="./checkpoints/",
        help="path where to save the model and data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from a model saved earlier",
    )
    main(parser.parse_args())

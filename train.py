#!/usr/bin/env python3
"""Train the DQN for Pong."""
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import gym
import tensorflow as tf
import yaml
from gym.wrappers import Monitor  # gym.wrappers doesn't work
from tqdm import tqdm
from typing_extensions import Final

from model import get_model
from utils import (
    IMG_SIZE,
    STATE_FRAMES,
    ReplayBuffer,
    TransitionType,
    choose,
    preprocess,
    sample_replay,
    set_all_seeds,
)

CONFIG_NAME: Final = "config.yaml"


class DQNTrainer:
    """Class to train a DQN for Pong."""

    MODEL_NAME: Final = "model.ckpt"
    FIXED_NAME: Final = "fixed.ckpt"
    DATA_NAME: Final = "data.pkl"

    def __init__(
        self,
        env: gym.Wrapper,
        model: tf.keras.Model,
        fixed: tf.keras.Model,
        replay: ReplayBuffer[TransitionType],
        optimizer: tf.keras.optimizers.Optimizer,
        writer: tf.summary.SummaryWriter,
        batch_size: int,
        discount: float,
        reset_steps: int,
        log_steps: int,
        video_eps: int,
        log_dir: Path,
        save_dir: Path,
    ):
        """Store the main model and other info.

        Args:
            env: The Atari Pong environment
            model: The model to be trained
            fixed: The model with fixed weights used for the Q-targets
            replay: The experience replay buffer
            optimizer: The optimizer
            writer: The summary writer for saving logs
            batch_size: The no. of states to sample from the replay buffer at
                one instance
            discount: Discount factor for reward
            reset_steps: Steps after which the fixed model is to be updated
            log_steps: Steps after which model is to be logged
            video_eps: Episodes after which video is to be saved
            log_dir: Path where to save logs
            save_dir: Path where to save the model and data
        """
        # The Pong environment, with a video monitor attached
        self.env = Monitor(
            env,
            log_dir / "videos",
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

    def load_info(self) -> int:
        """Load models and training parameters.

        Returns:
            The episode when the previous model was terminated
        """
        self.model.load_weights(self.save_dir / self.MODEL_NAME)
        self.fixed.load_weights(self.save_dir / self.FIXED_NAME)
        with open(self.save_dir / self.DATA_NAME, "rb") as data:
            start = pickle.load(data)
        print("Loaded model and training data")
        return start

    def save_info(self, episode: int) -> None:
        """Save models and training parameters.

        Args:
            episode: The count of the current episode
        """
        self.model.save_weights(self.save_dir / self.MODEL_NAME)
        self.fixed.save_weights(self.save_dir / self.FIXED_NAME)
        with open(self.save_dir / self.DATA_NAME, "wb") as data:
            pickle.dump(episode, data)

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

    def train_episode(
        self, epsilon: float, global_step: int
    ) -> Tuple[Optional[tf.Tensor], int]:
        """Run one episode and train the model on it.

        Args:
            epsilon: Current value of epsilon for the epsilon-greedy policy
            global_step: The no. of frames processed so far

        Returns:
            The first state encountered
            The updated global step
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
        episodes: int,
        init_epsilon: float,
        min_epsilon: float,
        decay_wait: int,
        decay_eps: int,
        save_eps: int,
        start: int = 0,
    ) -> None:
        """Train the DQN on Pong.

        Args:
            episodes: The max episodes to train the model
            init_epsilon: Initial value of epsilon for the epsilon-greedy
                policy
            min_epsilon: Lower bound for epsilon after decay
            decay_wait: No. of episodes to wait before decaying epsilon
            decay_eps: No. of episodes for epsilon decay
            save_eps: Episodes after which model and data are to be saved
            start: The starting episode
        """
        # Epsilon is decayed linearly for a few episodes, then kept constant
        epsilon_decay = (init_epsilon - min_epsilon) / decay_eps
        epsilon = max(
            init_epsilon - epsilon_decay * max(start - decay_wait, 0),
            min_epsilon,
        )

        global_step = 1
        ep = start  # if start == episodes, the for loop isn't executed

        try:
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
                    self.save_info(ep)

                if decay_wait <= ep <= decay_wait + decay_eps:
                    epsilon -= epsilon_decay

        except KeyboardInterrupt:
            pass
        finally:
            self.save_info(ep)


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    # Automatically implements frame skipping internally
    env = gym.make("Pong-v4", frameskip=args.frame_skips)

    if args.seed is not None:
        set_all_seeds(env, args.seed)

    model = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )

    fixed = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )
    replay = ReplayBuffer[TransitionType](limit=args.replay_size)

    # Save each run into a directory by its timestamp.
    # Remove microseconds and convert to ISO 8601 YYYY-MM-DDThh:mm:ss format.
    time_stamp = datetime.now().replace(microsecond=0).isoformat()
    log_dir = Path(args.log_dir, time_stamp)

    save_dir = Path(args.save_dir)
    for directory in log_dir, save_dir:
        if not directory.exists():
            directory.mkdir(parents=True)
        with open(directory / CONFIG_NAME, "w") as conf:
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
        save_dir=save_dir,
    )

    if args.resume:
        start = trainer.load_info()
    else:
        fixed.set_weights(model.get_weights())
        start = 0

    trainer.train(
        args.episodes,
        args.init_epsilon,
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
        "--init-epsilon",
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
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility",
    )
    main(parser.parse_args())

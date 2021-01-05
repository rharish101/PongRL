#!/usr/bin/env python3
"""Test the DQN on Pong."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Deque

import gym
import tensorflow as tf
from gym.wrappers import Monitor  # gym.wrappers doesn't work

from model import get_model
from train import DQNTrainer
from utils import IMG_SIZE, STATE_FRAMES, choose, preprocess, set_all_seeds


def test(env: gym.Wrapper, model: tf.keras.Model, log_dir: str) -> None:
    """Test the DQN on Pong.

    Args:
        env: The Atari Pong environment
        model: The model to be trained
        log_dir: Path where to save the video
    """
    env = Monitor(
        env,
        log_dir,
        force=True,  # overwrite existing videos
        video_callable=lambda count: True,  # force save this episode
    )

    state = Deque[tf.Tensor](maxlen=STATE_FRAMES)
    state.append(preprocess(env.reset()))  # initial state

    print("Starting testing...")
    while True:
        if len(state) < STATE_FRAMES:
            initial = None
            action = env.action_space.sample()
        else:
            initial = tf.stack(state, axis=-1)
            action = choose(model, initial, 0)  # choose greedily

        state_new, _, done, _ = env.step(action)
        state_new = preprocess(state_new)
        state.append(state_new)

        if done:
            break
    print("Testing done")


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    env = gym.make("Pong-v4", frameskip=args.frame_skips)

    if args.seed is not None:
        set_all_seeds(env, args.seed)

    model = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )
    model.load_weights(os.path.join(args.load_dir, DQNTrainer.MODEL_NAME))
    print("Loaded model")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    test(env, model, log_dir=args.log_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Test the DQN on Pong",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="./checkpoints/",
        help="path from where to load the model and data",
    )
    parser.add_argument(
        "--frame-skips", type=int, default=4, help="how much frames to skip"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/test/",
        help="path where to save the video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility",
    )
    main(parser.parse_args())

#!/usr/bin/env python3
"""Test the DQN on Pong."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Deque

import gym
import tensorflow as tf
from gym.wrappers import Monitor  # gym.wrappers doesn't work

from model import get_model
from train import DQNTrainer
from utils import (
    ENV_NAME,
    STATE_FRAMES,
    choose,
    load_config,
    preprocess,
    set_all_seeds,
)


def test(env: gym.Wrapper, model: tf.keras.Model, log_dir: Path) -> None:
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
    config = load_config(args.config)

    env = gym.make(ENV_NAME, frameskip=config.frame_skips)

    if config.seed is not None:
        set_all_seeds(env, config.seed)

    model = get_model(env.action_space.n)
    model.load_weights(args.load_dir / DQNTrainer.MODEL_NAME)
    print("Loaded model")

    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True)

    test(env, model, log_dir=args.log_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Test the DQN on Pong",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default="./checkpoints/",
        help="path from where to load the model and data",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="./logs/test/",
        help="path where to save the video",
    )
    main(parser.parse_args())

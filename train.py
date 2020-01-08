#!/usr/bin/env python3
"""Train the DQN for Pong."""
import json
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque
from datetime import datetime

import gym
import tensorflow as tf
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

MODEL_SAVE_NAME = "model.ckpt"
FIXED_SAVE_NAME = "fixed.ckpt"
DATA_SAVE_NAME = "data.pkl"
CONFIG_SAVE_NAME = "config.json"


def saver(model, fixed, episode, epsilon, save_dir):
    """Save model and training parameters.

    Args:
        model (`tf.keras.Model`): The model to be trained
        fixed (`tf.keras.Model`): The model with fixed weights used for the
            Q-targets
        episode (int): The count of the current episode
        epsilon (float): Current value of epsilon for the epsilon-greedy policy
        save_dir (str): Path where to save the model and data

    """
    model.save_weights(os.path.join(save_dir, MODEL_SAVE_NAME))
    fixed.save_weights(os.path.join(save_dir, MODEL_SAVE_NAME))
    with open(os.path.join(save_dir, DATA_SAVE_NAME), "wb") as dataf:
        pickle.dump((episode, epsilon), dataf)


@tf.function
def exp_replay(
    model,
    fixed,
    optimizer,
    inputs,
    outputs,
    actions,
    rewards,
    terminals,
    discount,
):
    """Train the model on a random sample from the replay buffer.

    Args:
        model (`tf.keras.Model`): The model to be trained
        fixed (`tf.keras.Model`): The model with fixed weights used for the
            Q-targets
        optimizer (`tf.keras.optimizers.Optimizer`): The optimizer
        inputs (`tf.Tensor`): The float32 initial states for the batch of
            transitions
        outputs (`tf.Tensor`): The float32 corresponding final states for the
            batch of transitions
        actions (`tf.Tensor`): The int64 corresponding actions for the batch of
            transitions
        rewards (`tf.Tensor`): The float32 corresponding rewards for the batch
            of transitions
        terminals (`tf.Tensor`): The bool corresponding terminal indicators for
            the batch of transitions
        discount (float): Discount factor for reward

    """
    with tf.GradientTape() as tape:
        q_initial = model(inputs, training=True)
        q_final_main = model(outputs, training=True)
        q_final_fixed = fixed(outputs, training=True)

        # If final state is terminal, then target is only the reward
        mask = tf.cast(tf.logical_not(terminals), tf.float32)
        # Double DQN: Choose target values based on fixed model's values but
        # main model's actions.
        batch_range = tf.range(actions.shape[0], dtype=tf.int64)
        tgt_indices = tf.stack(
            [batch_range, tf.argmax(q_final_main, axis=1)], axis=1
        )
        q_final = tf.gather_nd(q_final_fixed, tgt_indices)
        targets = rewards + mask * discount * q_final

        # Choose q-values based on actions taken
        pred_indices = tf.stack([batch_range, actions], axis=1)
        pred = tf.gather_nd(q_initial, pred_indices)

        # Huber loss, to avoid gradient explosion
        loss = tf.keras.losses.Huber()(y_true=targets, y_pred=pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Needed for logging loss
    return loss


def train_episode(
    env,
    model,
    fixed,
    optimizer,
    replay,
    batch_size,
    epsilon,
    discount,
    global_step,
    reset_steps,
    writer,
    log_steps,
):
    """Run one episode and train the model on it.

    Args:
        env (`gym.Wrapper`): The Atari Pong environment
        model (`tf.keras.Model`): The model to be trained
        fixed (`tf.keras.Model`): The model with fixed weights used for the
            Q-targets
        optimizer (`tf.keras.optimizers.Optimizer`): The optimizer
        replay (`utils.ReplayBuffer`): The experience replay buffer
        batch_size (int): The no. of states to sample from the replay buffer at
            one instance
        epsilon (float): Initial value of epsilon for the epsilon-greedy policy
        discount (float): Discount factor for reward
        global_step (int): The no. of frames processed so far
        reset_steps (int): Steps after which the fixed model is to be updated
        writer (`tf.summary.SummaryWriter`): The summary writer for saving logs
        log_steps (int): Steps after which model is to be logged

    """
    state = deque(maxlen=STATE_FRAMES)
    state.append(preprocess(env.reset()))  # initial state

    first = None

    while True:
        if len(state) < STATE_FRAMES:
            initial = None
            action = env.action_space.sample()
        else:
            initial = tf.stack(state, axis=-1)
            action = choose(model, initial, epsilon)

        state_new, reward, done, _ = env.step(action)
        state_new = preprocess(state_new)
        state.append(state_new)

        if initial is not None:
            # The inputs for this transition are well-defined, ie. a
            # proper x-frames state, so add it to the replay buffer.
            replay.append((initial, state_new, action, reward, done))
            if first is None:
                first = initial

        if len(replay) >= batch_size:
            inputs, outputs, actions, rewards, terms = sample_replay(
                replay, batch_size
            )
            loss = exp_replay(
                model,
                fixed,
                optimizer,
                inputs,
                outputs,
                actions,
                rewards,
                terms,
                discount,
            )

        if global_step % log_steps == 0:
            with writer.as_default(), tf.name_scope("losses"):
                tf.summary.scalar("loss", loss, step=global_step)

        if global_step % reset_steps == 0:
            fixed.set_weights(model.get_weights())

        global_step += 1

        if done:
            break

    # Needed for logging metrics
    return first


def train(
    env,
    model,
    fixed,
    optimizer,
    replay,
    batch_size,
    episodes,
    epsilon,
    min_epsilon,
    decay_wait,
    decay_eps,
    discount,
    reset_steps,
    writer,
    log_steps,
    video_eps,
    log_dir,
    save_eps,
    save_dir,
    start=0,
):
    """Train the DQN on Pong.

    Args:
        env (`gym.Wrapper`): The Atari Pong environment
        model (`tf.keras.Model`): The model to be trained
        fixed (`tf.keras.Model`): The model with fixed weights used for the
            Q-targets
        optimizer (`tf.keras.optimizers.Optimizer`): The optimizer
        replay (`utils.ReplayBuffer`): The experience replay buffer
        batch_size (int): The no. of states to sample from the replay buffer at
            one instance
        episodes (int): The max episodes to train the model
        epsilon (float): Initial value of epsilon for the epsilon-greedy policy
        min_epsilon (float): Lower bound for epsilon after decay
        decay_wait (int): No. of episodes to wait before decaying epsilon
        decay_eps (int): No. of episodes for epsilon decay
        discount (float): Discount factor for reward
        reset_steps (int): Steps after which the fixed model is to be updated
        writer (`tf.summary.SummaryWriter`): The summary writer for saving logs
        log_steps (int): Steps after which model is to be logged
        video_eps (int): Episodes after which video is to be saved
        log_dir (str): Path where to save logs
        save_eps (int): Episodes after which model and data are to be saved
        save_dir (str): Path where to save the model and data
        start (int): The starting episode (useful when resuming progress)

    """
    env = Monitor(
        env,
        os.path.join(log_dir, "videos"),
        resume=False,  # don't retain older videos
        force=True,  # overwrite existing videos
        video_callable=lambda count: count % video_eps == 0,
    )

    # Epsilon is decayed linearly for a few episodes, then kept constant
    epsilon_decay = (epsilon - min_epsilon) / decay_eps
    global_step = 1

    try:
        for ep in tqdm(
            range(start + 1, episodes + 1), initial=start, total=episodes
        ):
            first = train_episode(
                env,
                model,
                fixed,
                optimizer,
                replay,
                batch_size,
                epsilon,
                discount,
                global_step,
                reset_steps,
                writer,
                log_steps,
            )

            with writer.as_default(), tf.name_scope("metrics"):
                first = tf.image.convert_image_dtype(first, tf.float32)
                pred = model(tf.expand_dims(first, axis=0))[0]  # not training
                tf.summary.scalar("max q", tf.reduce_max(pred), step=ep)

            if ep % save_eps == 0:
                saver(model, fixed, ep, epsilon, save_dir)

            if decay_wait <= ep <= decay_wait + decay_eps:
                epsilon -= epsilon_decay

    except KeyboardInterrupt:
        pass
    finally:
        saver(model, fixed, ep, epsilon, save_dir)


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
    if args.resume:
        model.load_weights(os.path.join(args.save_dir, MODEL_SAVE_NAME))
        fixed.load_weights(os.path.join(args.save_dir, FIXED_SAVE_NAME))
        with open(os.path.join(args.save_dir, DATA_SAVE_NAME), "wb") as dataf:
            start, epsilon = pickle.load(dataf)
        print("Loaded model and training data")
    else:
        fixed.set_weights(model.get_weights())
        start = 0
        epsilon = args.epsilon

    optimizer = tf.keras.optimizers.Adam(args.lr)
    writer = tf.summary.create_file_writer(args.log_dir)

    # Save each run into a directory by its timestamp.
    # Remove microseconds and convert to ISO 8601 YYYY-MM-DDThh:mm:ss format.
    time_stamp = datetime.now().replace(microsecond=0).isoformat()
    log_dir = os.path.join(args.log_dir, time_stamp)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save hyperparams in both log and save directories
    with open(os.path.join(args.log_dir, CONFIG_SAVE_NAME), "w") as conf:
        json.dump(vars(args), conf)
    with open(os.path.join(args.save_dir, CONFIG_SAVE_NAME), "w") as conf:
        json.dump(vars(args), conf)

    train(
        env,
        model,
        fixed,
        optimizer,
        replay,
        start=start,
        batch_size=args.batch_size,
        episodes=args.episodes,
        epsilon=epsilon,
        min_epsilon=args.min_epsilon,
        decay_wait=args.decay_wait,
        decay_eps=args.decay_eps,
        discount=args.discount,
        reset_steps=args.reset_steps,
        writer=writer,
        log_steps=args.log_steps,
        video_eps=args.video_eps,
        log_dir=log_dir,
        save_eps=args.save_eps,
        save_dir=args.save_dir,
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

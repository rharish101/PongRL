#!/usr/bin/env python3
"""Train the DQN for Pong."""
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque

import tensorflow as tf
from tqdm import tqdm

import gym
from gym.wrappers import Monitor  # gym.wrappers doesn't work
from model import get_model
from utils import IMG_SIZE, STATE_FRAMES, choose, preprocess, sample_replay

MODEL_SAVE_NAME = "model.ckpt"
FIXED_SAVE_NAME = "fixed.ckpt"
DATA_SAVE_NAME = "data.pkl"


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
    writer,
    global_step,
    log_steps,
    log_dir,
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
        writer (`tf.summary.SummaryWriter`): The summary writer for saving logs
        global_step (`tf.Variable`): The no. of frames processed so far
        log_steps (int): Steps after which model is to be logged
        log_dir (str): Path where to save logs

    """
    with tf.GradientTape() as tape:
        q_initial = model(inputs, training=True)
        q_final = fixed(outputs, training=True)

        # If final state is terminal, then target is only the reward
        mask = tf.cast(tf.logical_not(terminals), tf.float32)
        targets = rewards + mask * discount * tf.reduce_max(q_final, axis=1)

        # Choose q-values based on actions taken
        indices = tf.stack(
            [tf.range(actions.shape[0], dtype=tf.int64), actions], axis=1
        )
        pred = tf.gather_nd(q_initial, indices)

        loss = tf.reduce_mean((targets - pred) ** 2)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if global_step % log_steps == 0:
        with writer.as_default(), tf.name_scope("losses"):
            tf.summary.scalar("loss", loss, step=global_step)


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
    decay_eps,
    discount,
    frame_skips,
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
        replay (`collections.deque`): The experience replay buffer
        batch_size (int): The no. of states to sample from the replay buffer at
            one instance
        episodes (int): The max episodes to train the model
        epsilon (float): Initial value of epsilon for the epsilon-greedy policy
        min_epsilon (float): Lower bound for epsilon after decay
        decay_eps (int): No. of episodes for epsilon decay
        discount (float): Discount factor for reward
        frame_skips (int): How much frames to skip
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
        video_callable=lambda count: count % video_eps == 0,
    )

    # Epsilon is decayed linearly for a few episodes, then kept constant
    epsilon_decay = (epsilon - min_epsilon) / decay_eps
    # `tf.Variable` is used, as global step changes every frame, and thus the
    # graph will be re-traced if it were a python value. Also, int64 is
    # expected by the summary op.
    # This is used for saving logs.
    global_step = tf.Variable(1, dtype=tf.int64)

    try:
        for ep in tqdm(
            range(start + 1, episodes + 1), initial=start, total=episodes
        ):
            state = deque(maxlen=STATE_FRAMES)
            state.append(preprocess(env.reset()))  # initial state

            first = None

            while True:
                if len(state) < STATE_FRAMES or global_step % frame_skips != 0:
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

                # Sample from the replay buffer if not skipping frames
                if (
                    len(replay) >= batch_size
                    and global_step % frame_skips == 0
                ):
                    inputs, outputs, actions, rewards, terms = sample_replay(
                        replay, batch_size
                    )
                    prev_wts = model.get_weights()

                    exp_replay(
                        model,
                        fixed,
                        optimizer,
                        inputs,
                        outputs,
                        actions,
                        rewards,
                        terms,
                        discount,
                        writer,
                        global_step,
                        log_steps,
                        log_dir,
                    )

                    # Ensure that the fixed model weights are always one step
                    # behind the model's weights.
                    fixed.set_weights(prev_wts)

                global_step.assign_add(1)
                if done:
                    break

            with writer.as_default(), tf.name_scope("metrics"):
                first = tf.image.convert_image_dtype(first, tf.float32)
                pred = model(tf.expand_dims(first, axis=0))[0]  # not training
                tf.summary.scalar("max q", tf.reduce_max(pred), step=ep)

            if ep % save_eps == 0:
                saver(model, fixed, ep, epsilon, save_dir)

            if ep <= decay_eps:
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
    env = gym.make("Pong-v4")

    model = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )
    fixed = get_model(
        IMG_SIZE + (STATE_FRAMES,), output_dims=env.action_space.n
    )

    replay = deque(maxlen=args.replay_size)
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

    optimizer = tf.keras.optimizers.RMSprop(args.lr)
    writer = tf.summary.create_file_writer(args.log_dir)

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
        decay_eps=args.decay_eps,
        discount=args.discount,
        frame_skips=args.frame_skips,
        writer=writer,
        log_steps=args.log_steps,
        video_eps=args.video_eps,
        log_dir=args.log_dir,
        save_eps=args.save_eps,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train the DQN for Pong",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for RMSprop"
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
        default=0.1,
        help="lower bound for epsilon after decay",
    )
    parser.add_argument(
        "--decay-eps",
        type=int,
        default=int(1e6),
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
        default=int(1e6),
        help="max size of experience replay buffer",
    )
    parser.add_argument(
        "--frame-skips", type=int, default=4, help="how much frames to skip"
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

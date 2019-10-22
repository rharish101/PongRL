# RL Agent for Atari Pong

Implementation of [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).

[OpenAI Gym](https://github.com/openai/gym)'s interface for the Atari Pong environment is used in this project.

## Instructions

### Setup
Install all required Python libraries:
```
pip install -r requirements.txt
```

### Training
Run `train.py`:
```
./train.py
```
The trained model is saved in TensorFlow's ckpt format (to the directory given by the `--save-dir` argument).

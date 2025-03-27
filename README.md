# FlappyBird-RL

## Goal

This project uses Reinforcement Learning (RL) to train an AI agent capable of autonomously playing the game Flappy Bird. The goal is to demonstrate how RL algorithms can effectively solve complex decision-making tasks in dynamic environments.

## Repository Structure

- QAgent.py: Contains the Deep Q-Network (DQN) agent implementation.
- train_Q_base_learner.py: Script to train the baseline RL agent.
- train_DQN.py: Script to train the DQN agent.
- plot_moving_average.ipynb: Notebook to see the results

## Key Components

1. _Environment:_

   - Gymnasium Librairy :

2. _Reinforcement Learning Algorithm:_

   - Deep Q-Network (DQN)
   - Uses an epsilon-greedy policy for exploration and exploitation.

3. _State Representation:_

   - Horizontal and vertical distances to the nearest pipes for the baseline
   - The full 12 dimensions states for the fully connected layer

4. _Reward Function:_

   - Reward of +1 for successfully passing a pipe
   - Reward of 0.1 for not dying
   - Negative reward (-0.5) for touching the top screen
   - Negative reward (-1) for collisions or hitting the ground

## Dependencies

- Python 3.8 or newer
- PyTorch
- NumPy
- Flappy-bird-gymnasium
- Gymnasium

## Installation

bash
git clone https://github.com/tanguyLC01/FlappyBird-RL
cd FlappyBird-RL
pip install -r requirements.txt

## Training the Agent

Run the following command to begin training:
bash
python train_Q_base_learner.py

## Visualizing Agent Performance

See the jupyter plot_moving_average.ipynb

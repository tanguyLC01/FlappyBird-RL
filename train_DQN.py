import torch
from QAgent import DQNAgent
import gymnasium
import flappy_bird_gymnasium

hyperparameters = {
    'eps_decay': 0.9999,
    'eps_start': 0.8, 
    'eps_min': 1e-5,
    'action_space': 2,
    'memory_size': 50000,
    'learning_rate': 3e-5,
    'gamma': 0.99,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pre_train_steps': 100,
    'num_episodes': 50000,
    'batch_size': 48,
    #'update_freq': 1000,
    'img_width': 80,
    'img_height': 80,
    'state_size': 6400,
    'action_size': 2,
    'img_buffer_size': 50000
}


env = gymnasium.make("FlappyBird-v0", use_lidar=False)

dql = DQNAgent(parameters=hyperparameters)
q_values, score = dql.train(env)
# %%
import time
import flappy_bird_gymnasium
import gymnasium
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

#OPTIMISATION
torch.backends.cudnn.benchmark = True  # Déjà activé
torch.backends.cudnn.enabled = True  # Active toutes les optimisations cuDNN

class DQN(torch.nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, action_space)
        self.relu = nn.ReLU()

    def forward(self, observation):
        output = self.relu(self.conv1(observation))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy
import random
import numpy as np
from collections import deque
import torch.optim as optim
from datetime import datetime, timedelta


class DQNAgent:

    def __init__(self, parameters):
        self.eps_decay = parameters['eps_decay']
        self.eps_min = parameters['eps_min']
        self.eps = parameters['eps_start']
        self.action_space = parameters['action_space']
        self.memory = deque(maxlen=parameters['memory_size'])
        self.pre_trained_episode = parameters['pre_train_steps']
        self.device = parameters['device']
        self.batch_size = parameters['batch_size']
        self.gamma = parameters['gamma']
        self.num_episodes = parameters['num_episodes']
        self.Q_policy = DQN(self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_policy).to(self.device)
        self.optimizer = optim.AdamW(self.Q_policy.parameters(), lr=parameters['learning_rate'])
        self.lossFuc = nn.MSELoss()
        self.action_dict = {0: 0, 1: 1}
        self.scaler = torch.cuda.amp.GradScaler()
    
    def one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes) 
        return y[labels] 
    
    def img_process(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        return img

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        state = np.reshape(state, (1, 1, 80, 80))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.Q_policy(state_tensor)[0]

        if np.random.rand() < self.eps:
            max_q_index = torch.randint(self.action_space, (1,)).item()
        else:
            max_q_index = torch.argmax(q_values).item()
        max_q_one_hot = self.one_hot_embedding([max_q_index], self.action_space).to(self.device)
        return max_q_index, max_q_one_hot, q_values

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device) for s in state_batch]).reshape(-1, 1, 80, 80)
        action_batch = torch.stack(action_batch).reshape(-1, self.action_space)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).reshape(-1, 1)
        next_state_batch = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device) for s in next_state_batch]).reshape(-1, 1, 80, 80)

        current_predictions = self.Q_policy(state_batch)
        next_predictions = self.Q_target(next_state_batch).detach()
        y_batch = reward_batch + self.gamma * torch.max(next_predictions, dim=1, keepdim=True)[0] * (~torch.tensor(terminal_batch, dtype=torch.bool, device=self.device)).float().unsqueeze(1)

        q_value = torch.sum(current_predictions * action_batch, dim=1)
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss = self.lossFuc(q_value, y_batch)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.cuda.synchronize()

        self.eps = max(self.eps_min, self.eps_decay * self.eps)

    def save_model(self, file_name='dqn_model.pth'):
        checkpoint = {
            'model_state_dict': self.Q_policy.state_dict(),
            'target_model_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
        }
        torch.save(checkpoint, file_name)
        print(f"Model saved to {file_name}")
    
    def load_model(self, file_name='dqn_model.pth'):
        checkpoint = torch.load(file_name)
        self.Q_policy.load_state_dict(checkpoint['model_state_dict'])
        self.Q_target.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eps = checkpoint['eps']
        print(f"Model loaded from {file_name}")
    
    def train(self, env):
        run = 0
        score_values = []
        q_values = []
        rewards_total = []
        best_reward = float("-inf")
        with open('training_log.txt', 'w') as f:
            for _ in range(self.num_episodes):
                episode_reward = 0.0
                run += 1
                state = env.reset()
                state = env.render()
                state = self.img_process(state)
                step = 0

                if run % 5000 == 4999:
                    self.save_model('model_checkpoint.pth')

                done = False
                
                while not done:
                    step += 1
                    action, action_one_hot, p_values = self.predict(state)
                    next_state, reward, done, info, score = env.step(self.action_dict[action])
                    next_state = env.render()
                    next_state = self.img_process(next_state)
                    reward = torch.tensor([reward], device=self.device)
                    self.remember(state, action_one_hot, reward, next_state, done)
                    episode_reward += reward.detach().cpu().numpy()[0]
                    state = next_state

                    if done:
                        log = f"Run: {run}, exploration: {self.eps}, score: {score['score']}, Q_value: {torch.max(p_values)}, Reward : {episode_reward}"
                        print(log)
                        f.write(log + '\n')
                        q_values.append(torch.max(p_values))
                        score_values.append(score['score'])
                        rewards_total.append(episode_reward)
                        break

                    self.experience_replay()

                    if step % 50 == 0:
                        self.Q_target.load_state_dict(copy.deepcopy(self.Q_policy.state_dict()))
                        
                if episode_reward > best_reward:
                    torch.save(self.Q_target.state_dict(), 'model.pth')
                    best_reward = episode_reward

        return score_values, q_values


# %%
import torch


hyperparameters = {
    'eps_decay': 0.9999,
    'eps_start': 5e-1, 
    'eps_min': 1e-5,
    'action_space': 2,
    'memory_size': 50000,
    'learning_rate': 9e-4,
    'gamma': 0.99,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pre_train_steps': 100,
    'num_episodes': 50000,
    'batch_size': 500,
    #'update_freq': 1000,
    'img_width': 80,
    'img_height': 80,
    'state_size': 6400,
    'action_size': 2,
    'img_buffer_size': 50000
}


env = gymnasium.make("FlappyBird-v0", render_mode='rgb_array')

dql = DQNAgent(parameters=hyperparameters)
q_values, score = dql.train(env)



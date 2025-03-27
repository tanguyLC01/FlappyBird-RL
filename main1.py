# %%
import time
import flappy_bird_gymnasium as flappy_bird_gym
import gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# %%
env = gym.make("FlappyBird-v0")

obs = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()  # env.action_space.sample() for a random action

    # Processing:
    obs, reward, done, _, info = env.step(action)
    
    # Rendering the game:
    # (remove this two lines during training)
    # env.render()
    # time.sleep(1 / 30)  # FPS
    
    # Checking if the player is still alive
    if done:
        break
# env.close()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
import cv2
import copy

class Agent:
    
    def __init__(self, parameters):
        self.eps_decay = parameters['eps_decay']
        self.eps_min = parameters['eps_min']
        self.eps = parameters['eps_start']
        self.action_space = parameters['action_space']
        self.memory = deque(maxlen=parameters['memory_size'])
        self.device = parameters['device']
        self.batch_size = parameters['batch_size']
        self.gamma = parameters['gamma']
        self.Q_policy = DQN(self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_policy).to(self.device)
        self.optimizer = optim.Adam(self.Q_policy.parameters(), lr = parameters['learning_rate'])
        self.lossFuc = torch.nn.MSELoss()
        self.action_dict = {0:0, 1:1}
        
    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.
        Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes) 
        return y[labels] 
    
    def img_process(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img[:400,:]
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        retval, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        #img = torch.FloatTensor(img)
        return img

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        state = np.reshape(state, (1, 1, 80, 80))  # Reshape to (batch_size, channels, height, width)
        state_tensor = torch.FloatTensor(state)  
        q_values = self.Q_policy(state_tensor)[0]

        # best action
        if np.random.rand() < self.eps:
            max_q_index = [random.randrange(self.action_space)]
            max_q_one_hot =  self.one_hot_embedding(max_q_index, self.action_space)
        else:
            max_q_index =  [torch.max(q_values, 0).indices.cpu().numpy().tolist()]
            max_q_one_hot =  self.one_hot_embedding(max_q_index, self.action_space)
       # print(torch.max(q_values).detach().numpy())
        return max_q_index, max_q_one_hot, q_values
        
    def experince_replay(self):
            if len(self.memory) < self.batch_size:
                return
            batch = random.sample(self.memory, self.batch_size) 
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
            state_batch = torch.cat(tuple(torch.FloatTensor(state) for state in state_batch)).reshape(self.batch_size, 1, 80, 80)
            action_batch = torch.cat(tuple(torch.FloatTensor(action) for action in action_batch)).reshape(self.batch_size, self.action_space)
            reward_batch = torch.cat(tuple(torch.Tensor(reward) for reward in reward_batch)).reshape(self.batch_size, 1)
            next_state_batch = torch.cat(tuple(torch.Tensor(next_state) for next_state in next_state_batch)).reshape(self.batch_size, 1, 80, 80)
            current_prediction_batch = self.Q_policy(state_batch)
            next_prediction_batch = self.Q_target(next_state_batch)

            y_batch = torch.cat(
                tuple(reward if terminal else reward + self.gamma * torch.max(prediction) for reward, terminal, prediction in
                    zip(reward_batch, terminal_batch, next_prediction_batch)))

            q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
            self.optimizer.zero_grad()
            # y_batch = y_batch.detach()
            loss = self.lossFuc(q_value, y_batch)
            loss.backward()
            self.optimizer.step()

            self.exploration_rate  = max(self.eps_min, self.eps_decay * self.eps)
        
    def train(self, env):    
        reward = 0.0
        run = 0
        score = []
        q_values = []
        while True:
            run += 1
            state = env.reset()
            state = self.img_process(state)
            step = 0
            if run % 5000 == 4999:
                torch.save(self.Q_policy.state_dict(), 'model.pth')
            done =  False
            while not done:
                step += 1
                action, action_one_hot, p_values = self.predict(state)
                next_state, reward, done, info = env.step(self.action_dict[action[0]])
                state_next = self.img_process(next_state)
                
                reward = torch.tensor([reward])
                if torch.cuda.is_available():
                    reward = reward.cuda()
                    action_one_hot = action_one_hot.cuda()
                self.remember(state, action_one_hot, reward, state_next,
                                    done)
                state = state_next

                if done:
                    print ("Run: " + str(run) + ", exploration: " + str(self.eps) + ", score: " + str(step) )
                    print('Q_value :', torch.max(p_values))
                    q_values.append(torch.max(p_values))
                    score.append(step)
                    print('Score : ', step)
                    break
                self.experince_replay()
                if step % 50 == 0:
                    self.Q_target.load_state_dict(copy.deepcopy(self.Q_policy.state_dict()))
        return score, q_values
            

# %%
hyperparameters = {
    'eps_decay': 0.999,
    'eps_start': 0.1, 
    'eps_min': 0.01,
    'action_space': 2,
    'memory_size': 10000,
    'learning_rate': 1e-4,
    'gamma': 0.9,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pre_train_steps': 1000,
    'num_episodes': 10000,
    'batch_size': 32,
    'update_freq': 1000,
    'img_width': 80,
    'img_height': 80,
    'state_size': 6400,
    'action_size': 2,
    'img_buffer_size': 50000
}


env = gym.make("FlappyBird-v0")

dql = Agent(parameters=hyperparameters)
q_values, score = dql.train(env)



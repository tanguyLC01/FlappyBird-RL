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

class FlappyBirQdAgent():
    
    def __init__(self, init_config):
        """
        This is where we get things rolling when the experiment kicks off.
        
        Args:
        init_config (dict): A bundle of setup goodies, including:
        {
            grid_size (int): Number of possible states,
            moves_count (int): Number of potential moves,
            explore_rate (float): Chance of exploring randomly,
            learning_rate (float): How fast we adapt,
            future_discount (float): How much we value future rewards,
        }
        """
        
        # Unpack the setup bundle and stash the values for later.
        self.moves_count = init_config["moves_count"]
        self.grid_size = init_config["grid_size"]
        self.explore_rate = init_config['explore_start']
        self.explore_decay = init_config['explore_decay']  # How much less we explore after each round
        self.min_explore = init_config['explore_min']  # The lowest possible exploration rate
        self.learning_rate = init_config["learning_rate"]
        self.future_discount = init_config["future_discount"]
        self.random_gen = np.random.RandomState(init_config["random_seed"])
        
        # Initialize the Q-table for keeping track of action values (all zeros for now).
        self.q_table = np.zeros((self.grid_size, self.moves_count))  

        # Map game states (like (x, y) coordinates) to unique indices. This keeps things tidy.
        self.state_index_map = {}
        
    def get_state_index(self, state):
        """Map the state to an index, and dynamically resize Q-table and eligibility traces if necessary."""
        if state not in self.state_index_map:
            state_idx = len(self.state_index_map)
            self.state_index_map[state] = state_idx

            # Dynamically expand Q-table and eligibility trace arrays to handle the new state
            if state_idx >= self.q_table.shape[0]:
                self.q_table = np.vstack([self.q_table, np.zeros((1, self.moves_count))])
        else:
            state_idx = self.state_index_map[state]
        return state_idx

    def agent_start(self, game_state):
        """
        Called at the start of each game session.

        Args:
            game_state (tuple): The initial state from the game environment.

        Returns:
            chosen_move (int): The agent's first move.
        """
        
        # Slow down the exploring gradually with each new game.
        self.explore_rate = max(self.explore_rate * self.explore_decay, self.min_explore)

        state_tuple =  (int(game_state[3]*80), int((game_state[9] - (game_state[5] - game_state[6])//2)*80))

        # Assign or retrieve an index for the current state (x, y coordinates).
        state_index = self.get_state_index(state_tuple)
        
        # Time to choose a move! Either explore or exploit based on explore_rate.
        current_q_values = self.q_table[state_index, :]
        if self.random_gen.rand() < self.explore_rate:
            chosen_move = self.random_gen.randint(self.moves_count)  # Go rogue and explore!
        else:
            chosen_move = np.argmax(current_q_values)  # Play it smart.

        # Save the current state and action for later use.
        self.previous_state_index = state_index
        self.previous_move = chosen_move

        return chosen_move

    def agent_step(self, feedback, game_state):
        """
        Called every time the agent takes a step.

        Args:
            feedback (float): Reward for the last move.
            game_state (tuple): The new state the agent landed in.

        Returns:
            next_move (int): The next action the agent decides to take.
        """
        # Map (or add) the state and grab the corresponding index.
        state_tuple =  (int(game_state[3]*80), int((game_state[9] - (game_state[5] - game_state[6])//2)*80))

        # Assign or retrieve an index for the current state (x, y coordinates).
        state_index = self.get_state_index(state_tuple)

        # Pick the next move using our current understanding of the world.
        current_q_values = self.q_table[state_index, :]
        if self.random_gen.rand() < self.explore_rate:
            next_move = self.random_gen.randint(self.moves_count)  # Explore again!
        else:
            next_move = np.argmax(current_q_values)

        # Time to update the Q-value (the brain!) using the feedback and new state info.
        old_value = self.q_table[self.previous_state_index, self.previous_move]
        best_future_q = np.max(self.q_table[state_index, :])
        self.q_table[self.previous_state_index, self.previous_move] = old_value + self.learning_rate * (feedback + self.future_discount * best_future_q - old_value)

        # Update the state and action for the next step.
        self.previous_state_index = state_index
        self.previous_move = next_move

        return next_move

    def agent_end(self, feedback):
        """
        Called when the game session ends (e.g., the bird crashes!).

        Args:
            feedback (float): The final reward received at the terminal state.
        """
        # Update the Q-value for the last move since there are no future rewards left.
        self.q_table[self.previous_state_index, self.previous_move] += 1/(1 + self.learning_rate)  * feedback 
        
    def train(self, environment, episodes):
        """
        Train the agent by letting it play multiple episodes and learn from its experience.

        Args:
            environment (object): The game environment that provides states, rewards, and termination info.
            episodes (int): Number of episodes (games) to train on.
        """
        with open("training_log_base_Q.txt", "w") as log_file:
            log_file.write("Episode,Reward,Score,Max Q Value\n")

            for episode in range(episodes):
                state, _ = environment.reset()  # Start a fresh game
                action = self.agent_start(state)

                done = False
                total_reward = 0
                steps = 0

                while not done:
                    next_state, reward, done, info, score = environment.step(action)    
                    total_reward += reward
                    steps += 1
                    if not done:
                        action = self.agent_step(reward, next_state)
                    else:
                        self.agent_end(reward)  # Handle the terminal state

                max_q_value = np.max(self.q_table)  # Get the top Q value from the table
                log_file.write(f"{episode + 1},{total_reward},{score['score']},{max_q_value:.6f}\n")
                print(f"Episode {episode + 1}/{episodes} completed. Reward: {total_reward}, Score: {score['score']}, Max Q: {max_q_value:.6f}. Explore rate: {self.explore_rate:.6f}")
        self.explore_rate = 0 
        



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
        self.optimizer = optim.Adam(self.Q_policy.parameters(), lr=parameters['learning_rate'])
        self.lossFuc = nn.MSELoss()
        self.action_dict = {0: 0, 1: 1}
    
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
        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.Q_policy(state_tensor)

        if np.random.rand() < self.eps:
            max_q_index = [random.randrange(self.action_space)]
            max_q_one_hot = self.one_hot_embedding(max_q_index, self.action_space)
        else:
            max_q_index = [torch.max(q_values, 0).indices.cpu().numpy().tolist()]
            max_q_one_hot = self.one_hot_embedding(max_q_index, self.action_space)
        return max_q_index, max_q_one_hot.to(self.device), q_values

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat([torch.FloatTensor(s) for s in state_batch]).reshape(-1, 12).to(self.device)
        action_batch = torch.stack(action_batch).reshape(-1, self.action_space)
        reward_batch = torch.cat([torch.FloatTensor([r]) for r in reward_batch]).to(self.device).reshape(-1, 12)
        next_state_batch = torch.cat([torch.FloatTensor(s) for s in next_state_batch]).to(self.device).reshape(-1, 12)

        #terminal = torch.cat([torch.tensor(t, dtype=torch.float32) for t in terminal_batch])
        current_predictions = self.Q_policy(state_batch)
        next_predictions = self.Q_target(next_state_batch)
        y_batch = torch.cat([
            r if d else r + self.gamma * torch.max(p)
            for r, d, p in zip(reward_batch, terminal_batch, next_predictions)
        ]).to(self.device)

        q_value = torch.sum(current_predictions * action_batch, dim=1)
        self.optimizer.zero_grad()
        loss = self.lossFuc(q_value, y_batch)
        loss.backward()
        self.optimizer.step()

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
                state, _ = env.reset()
                step = 0

                if run % 5000 == 4999:
                    self.save_model('model_checkpoint.pth')

                done = False
                
                while not done:
                    step += 1
                    action, action_one_hot, p_values = self.predict(state)
                    next_state, reward, done, info, score = env.step(self.action_dict[action[0]])
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

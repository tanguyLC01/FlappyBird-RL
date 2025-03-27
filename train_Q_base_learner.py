from QAgent import FlappyBirQdAgent
import gymnasium as gym
import flappy_bird_gymnasium

# Create the Flappy Bird environment.
env = gym.make("FlappyBird-v0", use_lidar=False)

# Create the Q-learning agent.
init_config = {
    "moves_count": 2,
    "grid_size": 6400,
    "explore_start": 0.99,
    "explore_decay": 0.999,
    "explore_min": 1e-4,
    "learning_rate": 0.1,
    "future_discount": 0.925,
    "random_seed": 42
}

agent = FlappyBirQdAgent(init_config)
agent.train(env, episodes=150000)
print(agent.state_index_map)
# After training, play the game with the trained Q-agent.
print("Training complete! Now let's watch the agent play...")

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = agent.agent_start(obs)
    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        print(info['score'])
        break
print(obs)
env.close()

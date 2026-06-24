import os
import gymnasium as gym
from agent import TileCodingAgent

# Configuration Variables
WEIGHTS_FILE = "mountain_car_weights.npy"
NUM_TILINGS = 8
GRID_SIZE = 8
TRAIN_EPISODES = 500
TEST_EPISODES = 3


def get_configured_agent(env):
    """Helper to initialize an agent with consistent geometric constraints."""
    return TileCodingAgent(
        num_tilings=NUM_TILINGS, 
        grid_size=GRID_SIZE, 
        num_actions=env.action_space.n, 
        low=env.observation_space.low, 
        high=env.observation_space.high,
        alpha=0.2, 
        epsilon=0.1
    )


def run_training_pipeline(episodes: int = 500):
    """Trains an agent from scratch and exports its weights matrix."""
    env = gym.make("MountainCar-v0")
    agent = get_configured_agent(env)

    print(f"--> Saved weights not found. Starting training for {episodes} episodes...")

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            
        agent.decay_epsilon()
        
        if (episode + 1) % 50 == 0:
            print(f"    Episode {episode + 1:03d}/{episodes} processed.")

    env.close()
    print("--> Training Cycle Complete!")
    
    # Export the weights matrix locally
    agent.save_weights(WEIGHTS_FILE)
    return agent


def run_visual_simulation(agent: TileCodingAgent, test_episodes: int = 3):
    """Executes a visual post-training simulation using the provided agent."""
    print(f"\n--> Launching visual simulation for {test_episodes} evaluation episodes...")
    
    env = gym.make("MountainCar-v0", render_mode="human")
    
    # CRITICAL: Force optimal performance (turn off exploration)
    agent.epsilon = 0.0 

    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
        print(f"    Evaluation Episode {episode + 1} | Escaped in {int(-total_reward)} steps!")

    env.close()
    print("--> Simulation Pipeline Complete.")


if __name__ == "__main__":
    # Check if the intelligence matrix already exists
    if os.path.exists(WEIGHTS_FILE):
        print(f"--> Found existing weights file: '{WEIGHTS_FILE}'")
        
        # 1. Initialize a blank environment just to read shape constraints
        temp_env = gym.make("MountainCar-v0")
        active_agent = get_configured_agent(temp_env)
        temp_env.close()
        
        # 2. Inject the saved weights directly into the new agent instance
        active_agent.load_weights(WEIGHTS_FILE)
    else:
        # 1. No file found -> Run training and grab the returned trained agent
        active_agent = run_training_pipeline(episodes=TRAIN_EPISODES)
    
    # Run the visual simulation using the active_agent (whether loaded or just trained!)
    run_visual_simulation(active_agent, test_episodes=TEST_EPISODES)
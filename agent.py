import numpy as np
from tile_coder import TileCoder  # Assuming tile_coder.py is in the same directory

class TileCodingAgent:
    """
    Reinforcement Learning Agent utilizing linear function approximation 
    via Tile Coding and standard 1-step SARSA/Q-learning updates.
    """
    def __init__(self, num_tilings: int, grid_size: int, num_actions: int, 
                 low: list, high: list, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        
        self.tile_coder = TileCoder(num_tilings, grid_size, low, high)
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Normalize learning rate by dividing by the number of active features (tilings)
        # This keeps updates stable and prevents weights from exploding
        self.alpha = alpha / num_tilings 
        
        # Total tiles across all feature tracking layers
        total_tiles = num_tilings * (grid_size ** 2)
        
        # Weight table initialized to zeros: shape (total_tiles, num_actions)
        self.weights = np.zeros((total_tiles, num_actions))
        
    def get_q_values(self, active_tiles: list) -> np.ndarray:
        """Computes the action values by summing the weights of all active tiles."""
        return np.sum(self.weights[active_tiles], axis=0)
        
    def select_action(self, state: np.ndarray) -> int:
        """Selects an action using an epsilon-greedy exploration strategy."""
        active_tiles = self.tile_coder.get_active_tiles(state)
        q_values = self.get_q_values(active_tiles)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(q_values)
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Performs a linear TD(0) update on the active weights."""
        active_tiles = self.tile_coder.get_active_tiles(state)
        current_q = self.get_q_values(active_tiles)[action]
        
        if done:
            target = reward
        else:
            next_active_tiles = self.tile_coder.get_active_tiles(next_state)
            next_q = np.max(self.get_q_values(next_active_tiles))
            target = reward + self.gamma * next_q
            
        td_error = target - current_q
        
        # Vectorized weight update applied simultaneously across all currently active tile paths
        self.weights[active_tiles, action] += self.alpha * td_error
        
    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01):
        """Smoothly drops the exploration rate over training generations."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        
    def save_weights(self, filepath: str):
        """Saves the current weight matrix to a binary file."""
        np.save(filepath, self.weights)
        print(f"Successfully saved weights to {filepath}")

    def load_weights(self, filepath: str):
        """Loads a weight matrix from a binary file."""
        self.weights = np.load(filepath)
        print(f"Successfully loaded weights from {filepath}")
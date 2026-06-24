import numpy as np

class TileCoder:
    """
    Handles the multi-layer overlapping grid (tiling) geometry to convert
    continuous states into discrete, high-dimensional binary feature IDs.
    """
    def __init__(self, num_tilings: int, grid_size: int, low: list, high: list):
        self.num_tilings = num_tilings
        self.grid_size = grid_size
        self.low = np.array(low)
        self.high = np.array(high)
        
        # Calculate the size of a single tile in each dimension
        self.tile_widths = (self.high - self.low) / (grid_size - 1)
        
    def get_active_tiles(self, state: np.ndarray) -> list:
        """
        Accepts a continuous state and returns a list of unique 1D integers 
        representing the active tile indices across all layers.
        """
        active_tiles = []
        state = np.array(state)
        
        for i in range(self.num_tilings):
            # Apply systematic displacement offset to this specific tiling layer
            shifted_state = state + i * (self.tile_widths / self.num_tilings)
            
            # Map the continuous coordinate to a discrete grid index
            grid_coord = ((shifted_state - self.low) / self.tile_widths).astype(int)
            
            # Ensure indices stay strictly within our grid bounds
            grid_coord = np.clip(grid_coord, 0, self.grid_size - 1)
            
            # Convert multidimensional grid coordinates into a unique 1D ID for this layer
            # Formula scales dynamically for a 2D observation space (position, velocity)
            tile_id = i * (self.grid_size ** 2) + (grid_coord[0] * self.grid_size + grid_coord[1])
            active_tiles.append(tile_id)
            
        return active_tiles
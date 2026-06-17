import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GameOfLifeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=40, max_steps=200, render_mode=None):
        super().__init__()
        self.grid_size   = grid_size
        self.max_steps   = max_steps
        self.render_mode = render_mode
        self.grid        = np.zeros((grid_size, grid_size), dtype=np.float32)

        # float32 required by most RL libraries (SB3, RLlib, etc.)
        self.observation_space = spaces.Box(0.0, 1.0, (grid_size, grid_size), dtype=np.float32)

        # agent picks any cell by flat index: 0 to grid_size*grid_size-1
        self.action_space = spaces.Discrete(grid_size * grid_size)

    def _tick(self):
        """Advance the Game of Life by one step using vectorized numpy."""
        p = np.pad(self.grid, 1, mode='constant')
        n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
             p[1:-1, :-2] +                p[1:-1, 2:] +
             p[2:,   :-2] + p[2:,   1:-1] + p[2:,  2:])
        self.grid = (
            ((self.grid == 1) & ((n == 2) | (n == 3))) |
            ((self.grid == 0) &  (n == 3))
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Start a new episode with a random grid."""
        super().reset(seed=seed)
        self.grid       = self.np_random.integers(0, 2, (self.grid_size, self.grid_size)).astype(np.float32)
        self.step_count = 0
        self.prev_live  = int(self.grid.sum())
        return self.grid.copy(), {}

    def step(self, action):
        """
        Agent toggles one cell, then the game ticks forward.

        action : int — flat index of the cell to toggle (row * grid_size + col)
        returns: obs, reward, terminated, truncated, info
        reward : change in live cell count this step (positive = more cells alive)
        """
        # 1. toggle the chosen cell
        row, col = divmod(int(action), self.grid_size)
        self.grid[row, col] = 1.0 - self.grid[row, col]

        # 2. advance game of life
        self._tick()
        self.step_count += 1

        # 3. reward = delta in live cells
        live           = int(self.grid.sum())
        reward         = float(live - self.prev_live)
        self.prev_live = live

        # 4. episode end conditions
        terminated = live == 0                       # all cells dead
        truncated  = self.step_count >= self.max_steps

        return self.grid.copy(), reward, terminated, truncated, {"live_cells": live}

    def render(self):
        """Return an RGB image of the current grid (only when render_mode='rgb_array')."""
        if self.render_mode == "rgb_array":
            img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            img[self.grid == 1] = [255, 255, 0]    # yellow = alive
            img[self.grid == 0] = [128, 128, 128]  # grey   = dead
            return img

    def close(self):
        pass


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("=== gymnasium check_env ===")
    env = GameOfLifeEnv(grid_size=20, max_steps=50)
    check_env(env, skip_render_check=False)
    print("PASSED\n")

    print("=== functional test (random agent) ===")
    env = GameOfLifeEnv(grid_size=20, max_steps=50)
    obs, _ = env.reset()
    print(f"obs dtype : {obs.dtype}")
    print(f"obs shape : {obs.shape}")
    print(f"action space : {env.action_space}\n")

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={i+1:02d}  live={info['live_cells']:4d}  reward={reward:+.1f}")
        if terminated or truncated:
            reason = "terminated (all dead)" if terminated else "truncated (max steps)"
            print(f"\nEpisode ended: {reason}")
            break

    env.close()

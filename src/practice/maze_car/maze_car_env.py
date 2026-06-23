import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from collections import deque

class ContinuousMazeCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1
        self.max_steps = 1000  # Give it more time since mazes are dynamic and winding
        
        # Grid settings for Maze
        self.cell_size = 50
        # DFS maze generation prefers odd dimensions
        self.grid_h, self.grid_w = 11, 11
        self.width = self.grid_w * self.cell_size
        self.height = self.grid_h * self.cell_size
        
        # Action space: [acceleration, steering]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space:
        # [dx, dy, next_dx, next_dy, v_x, v_y, cos(angle), sin(angle)] + 24 lidar sensors (360 degrees)
        self.num_lidar_rays = 24
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8 + self.num_lidar_rays,), dtype=np.float32)
        
        self.car_radius = 12.0  # Slightly smaller to fit through maze paths better
        
        # Pygame setup
        self.window = None
        self.clock = None
        
        self.reset()

    def _generate_maze(self):
        # Initialize maze with all walls (1)
        maze = np.ones((self.grid_h, self.grid_w), dtype=np.int8)
        
        # Helper to check if a cell is valid to carve into
        def is_valid(y, x):
            return 0 < y < self.grid_h - 1 and 0 < x < self.grid_w - 1
            
        # DFS Carving
        start_y, start_x = 1, 1
        maze[start_y, start_x] = 0
        stack = [(start_y, start_x)]
        
        while stack:
            cy, cx = stack[-1]
            # Shuffle directions for randomness
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            self.np_random.shuffle(directions)
            
            carved = False
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if is_valid(ny, nx) and maze[ny, nx] == 1:
                    # Carve the path and the intermediate wall
                    maze[cy + dy//2, cx + dx//2] = 0
                    maze[ny, nx] = 0
                    stack.append((ny, nx))
                    carved = True
                    break
                    
            if not carved:
                stack.pop()
                
        return maze

    def _compute_bfs_distances(self, target_cell):
        distances = np.full((self.grid_h, self.grid_w), -1, dtype=np.int32)
        distances[target_cell[0], target_cell[1]] = 0
        
        queue = deque([tuple(target_cell)])
        
        while queue:
            cy, cx = queue.popleft()
            curr_dist = distances[cy, cx]
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                    if self.maze_map[ny, nx] == 0 and distances[ny, nx] == -1:
                        distances[ny, nx] = curr_dist + 1
                        queue.append((ny, nx))
        return distances

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. Generate a new random maze
        self.maze_map = self._generate_maze()
        
        # 2. Find all free spaces (0)
        free_spaces = np.argwhere(self.maze_map == 0)
        
        # 3. Randomly pick target_cell from free spaces
        target_idx = self.np_random.choice(len(free_spaces))
        target_cell = free_spaces[target_idx]
        self.target_pos = (target_cell[1] * self.cell_size + self.cell_size / 2.0, 
                           target_cell[0] * self.cell_size + self.cell_size / 2.0)
        
        # 4. Compute BFS distances from target_cell
        self.bfs_distances = self._compute_bfs_distances(target_cell)
        
        # 5. Find cell(s) with maximum BFS distance (maximal separation)
        max_dist = np.max(self.bfs_distances)
        max_dist_cells = np.argwhere(self.bfs_distances == max_dist)
        
        # Randomly choose one of the furthest cells as start_cell
        start_idx = self.np_random.choice(len(max_dist_cells))
        start_cell = max_dist_cells[start_idx]
        
        self.start_pos = (start_cell[1] * self.cell_size + self.cell_size / 2.0, 
                          start_cell[0] * self.cell_size + self.cell_size / 2.0)
        
        # Reset car states
        self.car_x, self.car_y = self.start_pos
        self.car_v = 0.0
        self.car_angle = self.np_random.uniform(-np.pi, np.pi)
        
        # Record initial distance
        self.prev_dist = self._get_bfs_distance_at(self.car_x, self.car_y)
            
        self._update_lidar()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def _get_bfs_distance_at(self, px, py):
        # Convert pixels to cell index
        gx = int(px // self.cell_size)
        gy = int(py // self.cell_size)
        
        # Clip to bounds
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        
        d = self.bfs_distances[gy, gx]
        
        # Fallback if in wall
        if d == -1:
            closest_valid_d = -1
            closest_gx, closest_gy = -1, -1
            min_cell_dist = 9999
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        nd = self.bfs_distances[ny, nx]
                        if nd != -1:
                            dist = abs(dy) + abs(dx)
                            if dist < min_cell_dist:
                                min_cell_dist = dist
                                closest_valid_d = nd
                                closest_gx, closest_gy = nx, ny
            if closest_valid_d != -1:
                gx, gy = closest_gx, closest_gy
                d = closest_valid_d
            else:
                dist_to_target = math.hypot(self.target_pos[0] - px, self.target_pos[1] - py)
                return (self.grid_w + self.grid_h) * self.cell_size + dist_to_target

        if d == 0:
            return math.hypot(self.target_pos[0] - px, self.target_pos[1] - py)
        
        # Find neighbor with d - 1
        next_cell = None
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                if self.bfs_distances[ny, nx] == d - 1:
                    next_cell = (ny, nx)
                    break
                    
        if next_cell is not None:
            ny, nx = next_cell
            target_pixel_x = nx * self.cell_size + self.cell_size / 2.0
            target_pixel_y = ny * self.cell_size + self.cell_size / 2.0
            return (d - 1) * self.cell_size + math.hypot(target_pixel_x - px, target_pixel_y - py)
        else:
            return d * self.cell_size + math.hypot(self.target_pos[0] - px, self.target_pos[1] - py)

    def _get_obs(self):
        # Relative position to target
        dx = (self.target_pos[0] - self.car_x) / self.width
        dy = (self.target_pos[1] - self.car_y) / self.height
        
        # Relative position to the center of the next cell in shortest path
        gx = int(self.car_x // self.cell_size)
        gy = int(self.car_y // self.cell_size)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        
        d = self.bfs_distances[gy, gx]
        
        if d > 0:
            next_cell = None
            for dy_idx, dx_idx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = gy + dy_idx, gx + dx_idx
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                    if self.bfs_distances[ny, nx] == d - 1:
                        next_cell = (ny, nx)
                        break
            if next_cell is not None:
                ny, nx = next_cell
                next_cell_x = nx * self.cell_size + self.cell_size / 2.0
                next_cell_y = ny * self.cell_size + self.cell_size / 2.0
                next_dx = (next_cell_x - self.car_x) / self.width
                next_dy = (next_cell_y - self.car_y) / self.height
            else:
                next_dx, next_dy = dx, dy
        else:
            next_dx, next_dy = dx, dy
        
        v_x = (self.car_v * math.cos(self.car_angle)) / 50.0
        v_y = (self.car_v * math.sin(self.car_angle)) / 50.0
        
        c = math.cos(self.car_angle)
        s = math.sin(self.car_angle)
        
        # Lidar measurements normalized
        lidar_norm = [d / max(self.width, self.height) for d in self.lidar_distances]
        
        obs = [dx, dy, next_dx, next_dy, v_x, v_y, c, s] + lidar_norm
        return np.array(obs, dtype=np.float32)

    def _update_lidar(self):
        # 360 degree coverage
        angles = np.linspace(-np.pi, np.pi, self.num_lidar_rays, endpoint=False)
        self.lidar_distances = []
        max_dist = max(self.width, self.height)
        
        for a in angles:
            ray_angle = self.car_angle + a
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            dist = 0.0
            
            x, y = self.car_x, self.car_y
            while dist < max_dist:
                x += dx * 2
                y += dy * 2
                dist += 2
                
                # Check collision with wall or out of bounds
                gx = int(x // self.cell_size)
                gy = int(y // self.cell_size)
                if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h or self.maze_map[gy, gx] == 1:
                    break
            self.lidar_distances.append(dist)

    def step(self, action):
        self.current_step += 1
        
        acc_input = np.clip(action[0], -1.0, 1.0)
        steer_input = np.clip(action[1], -1.0, 1.0)
        
        # Physics update
        acc = acc_input * 20.0
        steer = steer_input * 3.0 # steer limit
        
        self.car_v += acc * self.dt
        self.car_v *= 0.95 # friction
        self.car_v = np.clip(self.car_v, -50.0, 50.0)
        
        # Steer only works when moving
        self.car_angle += steer * (self.car_v / 50.0) * self.dt
        
        new_x = self.car_x + self.car_v * math.cos(self.car_angle) * self.dt
        new_y = self.car_y + self.car_v * math.sin(self.car_angle) * self.dt
        
        # Collision detection
        margin = self.car_radius
        points_to_check = [
            (new_x, new_y),
            (new_x + margin, new_y), (new_x - margin, new_y),
            (new_x, new_y + margin), (new_x, new_y - margin)
        ]
        
        collided = False
        for px, py in points_to_check:
            gx = int(px // self.cell_size)
            gy = int(py // self.cell_size)
            if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h or self.maze_map[gy, gx] == 1:
                collided = True
                break
                
        if not collided:
            self.car_x = new_x
            self.car_y = new_y
            
        self._update_lidar()
        
        # Target logic
        dist_to_target = math.hypot(self.target_pos[0] - self.car_x, self.target_pos[1] - self.car_y)
        reached = dist_to_target < (self.car_radius + 15)
        
        # Dense Potential Reward
        curr_dist = self._get_bfs_distance_at(self.car_x, self.car_y)
        reward_potential = (self.prev_dist - curr_dist) * 0.1
        self.prev_dist = curr_dist
        
        reward = reward_potential - 0.1 # Constant time penalty
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if collided:
            reward -= 10.0
            terminated = True
            
        if reached:
            reward += 100.0
            terminated = True
            
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((18, 18, 20)) # Dark mode background
        
        # Draw maze walls
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.maze_map[y, x] == 1:
                    # Draw wall rect
                    pygame.draw.rect(
                        canvas,
                        (30, 30, 36),
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    # Subtle border
                    pygame.draw.rect(
                        canvas,
                        (45, 45, 55),
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1
                    )
                    
        # Draw shortest path (glowing blue line)
        path_cells = []
        gx = int(self.car_x // self.cell_size)
        gy = int(self.car_y // self.cell_size)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        
        curr_d = self.bfs_distances[gy, gx]
        if curr_d != -1:
            cx, cy = gx, gy
            path_cells.append((cx, cy))
            while curr_d > 0:
                next_cell = None
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        if self.bfs_distances[ny, nx] == curr_d - 1:
                            next_cell = (nx, ny)
                            break
                if next_cell is None:
                    break
                cx, cy = next_cell
                path_cells.append((cx, cy))
                curr_d -= 1
                
        if len(path_cells) > 1:
            points = []
            for cx, cy in path_cells:
                px = cx * self.cell_size + self.cell_size / 2
                py = cy * self.cell_size + self.cell_size / 2
                points.append((int(px), int(py)))
            # Glowing neon path
            pygame.draw.lines(canvas, (0, 80, 200), False, points, 8)
            pygame.draw.lines(canvas, (100, 200, 255), False, points, 3)

        # Draw target with glowing effect
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        pygame.draw.circle(canvas, (0, 100, 50), (tx, ty), 20, 0)
        pygame.draw.circle(canvas, (0, 200, 100), (tx, ty), 12, 0)
        pygame.draw.circle(canvas, (255, 255, 255), (tx, ty), 5, 0)
        
        # Draw Lidar rays (subtle blue-gray)
        angles = np.linspace(-np.pi, np.pi, self.num_lidar_rays, endpoint=False)
        for i, a in enumerate(angles):
            ray_angle = self.car_angle + a
            dist = self.lidar_distances[i]
            end_x = self.car_x + dist * math.cos(ray_angle)
            end_y = self.car_y + dist * math.sin(ray_angle)
            pygame.draw.line(canvas, (55, 55, 75), (int(self.car_x), int(self.car_y)), (int(end_x), int(end_y)), 1)
        
        # Draw car
        cx, cy = int(self.car_x), int(self.car_y)
        pygame.draw.circle(canvas, (255, 80, 80), (cx, cy), int(self.car_radius))
        
        # Draw direction line
        end_x = int(self.car_x + self.car_radius * math.cos(self.car_angle))
        end_y = int(self.car_y + self.car_radius * math.sin(self.car_angle))
        pygame.draw.line(canvas, (255, 255, 255), (cx, cy), (end_x, end_y), 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

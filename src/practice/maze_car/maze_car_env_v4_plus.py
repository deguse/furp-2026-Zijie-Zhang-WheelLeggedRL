import math
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ContinuousMazeCarEnvV4Plus(gym.Env):
    """
    V4 Plus continuous-control maze car environment.

    Improvements over V4:
    - Curriculum-friendly config: grid size, fixed/random maze, start difficulty.
    - Non-terminal collisions by default, so the agent can recover.
    - Strong dense BFS potential reward.
    - Heading, speed, and smoothness shaping rewards.
    - Info dict exposes success/collision/bfs distance for evaluation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        grid_size=7,
        max_steps=500,
        random_maze=False,
        start_mode="easy",  # easy | medium | far | random
        collision_ends_episode=False,
        seed=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.dt = 0.1
        self.max_steps = int(max_steps)

        # DFS maze generation works best with odd dimensions.
        grid_size = int(grid_size)
        if grid_size < 5:
            grid_size = 5
        if grid_size % 2 == 0:
            grid_size += 1
        self.grid_h = self.grid_w = grid_size

        self.cell_size = 50
        self.width = self.grid_w * self.cell_size
        self.height = self.grid_h * self.cell_size

        self.random_maze = bool(random_maze)
        self.start_mode = start_mode
        self.collision_ends_episode = bool(collision_ends_episode)
        self.fixed_maze_map = None

        # Action: [acceleration, steering], both in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation:
        # [target_dx, target_dy, next_wp_dx, next_wp_dy,
        #  vx, vy, cos(angle), sin(angle),
        #  prev_acc, prev_steer, last_collision] + lidar rays
        self.num_lidar_rays = 24
        self.base_obs_dim = 11 + self.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.base_obs_dim,),
            dtype=np.float32,
        )

        self.car_radius = 12.0
        self.window = None
        self.clock = None

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_collision = 0.0
        self.lidar_distances = [0.0] * self.num_lidar_rays

        self.reset(seed=seed)

    # -----------------------------
    # Maze utilities
    # -----------------------------
    def _generate_maze(self):
        maze = np.ones((self.grid_h, self.grid_w), dtype=np.int8)

        def is_valid(y, x):
            return 0 < y < self.grid_h - 1 and 0 < x < self.grid_w - 1

        start_y, start_x = 1, 1
        maze[start_y, start_x] = 0
        stack = [(start_y, start_x)]

        while stack:
            cy, cx = stack[-1]
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            self.np_random.shuffle(directions)
            carved = False

            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if is_valid(ny, nx) and maze[ny, nx] == 1:
                    maze[cy + dy // 2, cx + dx // 2] = 0
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

    def _cell_center(self, cell):
        y, x = int(cell[0]), int(cell[1])
        return (
            x * self.cell_size + self.cell_size / 2.0,
            y * self.cell_size + self.cell_size / 2.0,
        )

    def _choose_start_cell(self):
        valid = np.argwhere(self.bfs_distances > 1)
        if len(valid) == 0:
            valid = np.argwhere(self.maze_map == 0)

        max_d = int(np.max(self.bfs_distances))

        if self.start_mode == "far":
            candidates = np.argwhere(self.bfs_distances == max_d)
        elif self.start_mode == "medium":
            lo = max(2, int(max_d * 0.35))
            hi = max(lo + 1, int(max_d * 0.75))
            candidates = np.argwhere((self.bfs_distances >= lo) & (self.bfs_distances <= hi))
        elif self.start_mode == "random":
            candidates = valid
        else:  # easy
            hi = max(3, int(max_d * 0.55))
            candidates = np.argwhere((self.bfs_distances >= 2) & (self.bfs_distances <= hi))

        if len(candidates) == 0:
            candidates = valid

        idx = self.np_random.choice(len(candidates))
        return candidates[idx]

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_collision = 0.0

        if self.random_maze or self.fixed_maze_map is None:
            self.maze_map = self._generate_maze()
            if not self.random_maze:
                self.fixed_maze_map = self.maze_map.copy()
        else:
            self.maze_map = self.fixed_maze_map.copy()

        free_spaces = np.argwhere(self.maze_map == 0)
        target_cell = free_spaces[self.np_random.choice(len(free_spaces))]
        self.target_pos = self._cell_center(target_cell)

        self.bfs_distances = self._compute_bfs_distances(target_cell)

        start_cell = self._choose_start_cell()
        self.start_pos = self._cell_center(start_cell)

        self.car_x, self.car_y = self.start_pos
        self.car_v = 0.0
        self.car_angle = float(self.np_random.uniform(-np.pi, np.pi))

        self.prev_dist = self._get_bfs_distance_at(self.car_x, self.car_y)
        self._update_lidar()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        acc_input = float(np.clip(action[0], -1.0, 1.0))
        steer_input = float(np.clip(action[1], -1.0, 1.0))
        clipped_action = np.array([acc_input, steer_input], dtype=np.float32)

        acc = acc_input * 20.0
        steer = steer_input * 3.0

        self.car_v += acc * self.dt
        self.car_v *= 0.95
        self.car_v = float(np.clip(self.car_v, -50.0, 50.0))

        # Steering effectiveness scales with speed magnitude.
        self.car_angle += steer * (self.car_v / 50.0) * self.dt
        self.car_angle = math.atan2(math.sin(self.car_angle), math.cos(self.car_angle))

        new_x = self.car_x + self.car_v * math.cos(self.car_angle) * self.dt
        new_y = self.car_y + self.car_v * math.sin(self.car_angle) * self.dt

        collided = self._check_collision(new_x, new_y)
        if not collided:
            self.car_x = new_x
            self.car_y = new_y
            self.last_collision = 0.0
        else:
            # Let the agent recover instead of terminating most early episodes.
            self.car_v = 0.0
            self.last_collision = 1.0

        self._update_lidar()

        dist_to_target = math.hypot(self.target_pos[0] - self.car_x, self.target_pos[1] - self.car_y)
        reached = dist_to_target < (self.car_radius + 15)

        curr_dist = self._get_bfs_distance_at(self.car_x, self.car_y)
        reward_potential = (self.prev_dist - curr_dist) * 1.0
        self.prev_dist = curr_dist

        # Heading reward: encourage the car to point toward the next BFS waypoint.
        wp_dx, wp_dy = self._next_waypoint_delta()
        desired_angle = math.atan2(wp_dy, wp_dx)
        angle_diff = math.atan2(math.sin(desired_angle - self.car_angle), math.cos(desired_angle - self.car_angle))
        heading_reward = 0.03 * math.cos(angle_diff)

        # Encourage moving, but keep this small to avoid reckless crashing.
        speed_reward = 0.01 * min(abs(self.car_v) / 50.0, 1.0)

        # Encourage smoother controls.
        action_change = float(np.linalg.norm(clipped_action - self.prev_action))
        smoothness_penalty = 0.005 * action_change

        reward = reward_potential + heading_reward + speed_reward - smoothness_penalty - 0.005

        terminated = False
        truncated = self.current_step >= self.max_steps

        if collided:
            reward -= 1.0
            if self.collision_ends_episode:
                terminated = True

        if reached:
            reward += 100.0
            terminated = True

        self.prev_action = clipped_action

        if self.render_mode == "human":
            self._render_frame()

        info = {
            "is_success": bool(reached),
            "collided": bool(collided),
            "bfs_distance": float(curr_dist),
            "dist_to_target": float(dist_to_target),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # -----------------------------
    # Observation and physics helpers
    # -----------------------------
    def _get_next_cell(self, gy, gx):
        d = self.bfs_distances[gy, gx]
        if d <= 0:
            return None

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                if self.bfs_distances[ny, nx] == d - 1:
                    return ny, nx
        return None

    def _get_bfs_distance_at(self, px, py):
        gx = int(px // self.cell_size)
        gy = int(py // self.cell_size)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))

        d = self.bfs_distances[gy, gx]

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
                return (self.grid_w + self.grid_h) * self.cell_size

        if d == 0:
            return math.hypot(self.target_pos[0] - px, self.target_pos[1] - py)

        next_cell = self._get_next_cell(gy, gx)
        if next_cell is not None:
            target_pixel_x, target_pixel_y = self._cell_center(next_cell)
            return (d - 1) * self.cell_size + math.hypot(target_pixel_x - px, target_pixel_y - py)

        return d * self.cell_size + math.hypot(self.target_pos[0] - px, self.target_pos[1] - py)

    def _next_waypoint_delta(self):
        gx = int(self.car_x // self.cell_size)
        gy = int(self.car_y // self.cell_size)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))

        next_cell = self._get_next_cell(gy, gx)
        if next_cell is None:
            wx, wy = self.target_pos
        else:
            wx, wy = self._cell_center(next_cell)

        return wx - self.car_x, wy - self.car_y

    def _get_obs(self):
        target_dx = (self.target_pos[0] - self.car_x) / self.width
        target_dy = (self.target_pos[1] - self.car_y) / self.height

        next_dx_raw, next_dy_raw = self._next_waypoint_delta()
        next_dx = next_dx_raw / self.width
        next_dy = next_dy_raw / self.height

        v_x = (self.car_v * math.cos(self.car_angle)) / 50.0
        v_y = (self.car_v * math.sin(self.car_angle)) / 50.0
        c = math.cos(self.car_angle)
        s = math.sin(self.car_angle)

        lidar_norm = [d / max(self.width, self.height) for d in self.lidar_distances]

        obs = [
            target_dx,
            target_dy,
            next_dx,
            next_dy,
            v_x,
            v_y,
            c,
            s,
            float(self.prev_action[0]),
            float(self.prev_action[1]),
            float(self.last_collision),
        ] + lidar_norm

        return np.array(obs, dtype=np.float32)

    def _update_lidar(self):
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
                gx = int(x // self.cell_size)
                gy = int(y // self.cell_size)
                if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h or self.maze_map[gy, gx] == 1:
                    break

            self.lidar_distances.append(dist)

    def _check_collision(self, new_x, new_y):
        margin = self.car_radius
        points_to_check = [
            (new_x, new_y),
            (new_x + margin, new_y),
            (new_x - margin, new_y),
            (new_x, new_y + margin),
            (new_x, new_y - margin),
        ]

        for px, py in points_to_check:
            gx = int(px // self.cell_size)
            gy = int(py // self.cell_size)
            if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h or self.maze_map[gy, gx] == 1:
                return True
        return False

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((18, 18, 20))

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.maze_map[y, x] == 1:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(canvas, (30, 30, 36), rect)
                    pygame.draw.rect(canvas, (45, 45, 55), rect, 1)

        # Draw current shortest path.
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
            points = [
                (int(cx * self.cell_size + self.cell_size / 2), int(cy * self.cell_size + self.cell_size / 2))
                for cx, cy in path_cells
            ]
            pygame.draw.lines(canvas, (0, 80, 200), False, points, 8)
            pygame.draw.lines(canvas, (100, 200, 255), False, points, 3)

        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        pygame.draw.circle(canvas, (0, 100, 50), (tx, ty), 20, 0)
        pygame.draw.circle(canvas, (0, 200, 100), (tx, ty), 12, 0)
        pygame.draw.circle(canvas, (255, 255, 255), (tx, ty), 5, 0)

        angles = np.linspace(-np.pi, np.pi, self.num_lidar_rays, endpoint=False)
        for i, a in enumerate(angles):
            ray_angle = self.car_angle + a
            dist = self.lidar_distances[i]
            end_x = self.car_x + dist * math.cos(ray_angle)
            end_y = self.car_y + dist * math.sin(ray_angle)
            pygame.draw.line(canvas, (55, 55, 75), (int(self.car_x), int(self.car_y)), (int(end_x), int(end_y)), 1)

        cx, cy = int(self.car_x), int(self.car_y)
        pygame.draw.circle(canvas, (255, 80, 80), (cx, cy), int(self.car_radius))
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
            self.window = None
            self.clock = None


# Backward-compatible alias if needed.
ContinuousMazeCarEnv = ContinuousMazeCarEnvV4Plus

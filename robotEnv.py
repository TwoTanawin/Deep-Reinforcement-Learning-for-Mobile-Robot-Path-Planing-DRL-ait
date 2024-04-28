import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2 as cv
import random
import time

class CustomEnv(gym.Env):
    """Custom Environment for Mobile Robot Path Planning."""

    metadata = {"render.modes": ["human"], "render.fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(8)  # 8 discrete actions
        self.observation_space = spaces.Box(low=-1, high=2, shape=(4, 172, 300), dtype=np.float32)

        # Initialize parameters
        self.mapW = 40
        self.mapH = 24
        self.span_x = 37
        self.span_y = 21
        self.gridW = self.mapW + self.span_x * 2
        self.gridH = self.mapH + self.span_y * 2
        self.robot_rad = 1
        self.group_rad = self.robot_rad * 3
        self.n_stacks = 4
        self.n_actions = 8
        self.n_obstacles = 3
        self.obs_size_min = 2
        self.obs_size_max = 4
        self.image_ratio = 20  # scale up image when render
        self.map_ratio = 4
        self.input_dims = (self.n_stacks, (self.span_y * 2 + 1) * self.map_ratio,
                           (self.span_x * 2 + 1) * self.map_ratio)

        # Colors
        self.floor_color = [255, 255, 255]
        self.obstacle_color = [100, 100, 100]
        self.goal_color = [0, 255, 0]
        self.robot_color = [255, 0, 0]
        self.path_color = [0, 128, 255]

        self.reset()

    def reset(self, seed=None):
        # Reset environment
        self.rad = random.choice([self.robot_rad, self.group_rad])
        self.create_obstacles()
        self.get_grid()
        self.create_goal()
        self.create_start()
        self.robot_pos = self.start.copy()
        self.path = [self.robot_pos.copy()]
        state = self.get_state()
        self.stack = [state] * self.n_stacks
        info = {}  # Placeholder for additional information, if needed
        
        # Cast the observation to float32
        obs = np.array(self.stack, dtype=np.float32)
        return obs, info


    def create_obstacles(self):
        while True:
            self.obstacles = []
            for _ in range(100):
                w = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                h = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                x = random.randrange(self.span_x, self.gridW - self.span_x - w + 1)
                y = random.randrange(self.span_y, self.gridH - self.span_y - h + 1)
                if not self.overlap_obstacle((x, y, w, h)):
                    self.obstacles.append([x, y, w, h])
                if len(self.obstacles) == 3:
                    break
            if len(self.obstacles) == 3:
                break

    def overlap_obstacle(self, obs):
        x, y, w, h = obs
        for x_, y_, w_, h_  in self.obstacles:
            if x in range(x_ - w - self.rad*2, x_ + w_ + self.rad*2 + 1) and \
                y in range(y_ - h - self.rad*2, y_ + h_  + self.rad*2 + 1):
                return True
        return False

    def get_grid(self):
        self.grid = np.zeros((self.gridH, self.gridW))
        for i in range(self.gridW):
            for j in range(self.gridH):
                if i not in range(self.span_x + self.rad, self.gridW - self.span_x - self.rad) or \
                    j not in range(self.span_y + self.rad, self.gridH - self.span_y - self.rad):
                    self.grid[j, i] = -1
                else:           
                    for x, y, w, h in self.obstacles:
                        if i in range(x - self.rad, x + w + self.rad) and \
                            j in range(y - self.rad, y + h + self.rad):
                            self.grid[j, i] = -1

    def create_goal(self):
        self.goal = []
        while True:
            x = random.randrange(self.span_x + self.rad, self.gridW - self.span_x - self.rad)
            y = random.randrange(self.span_y + self.rad, self.gridH - self.span_y - self.rad)
            if self.grid[y, x] == 0:
                self.goal = [x, y]
                self.grid[y, x] = 1
                break

    def create_start(self):
        self.start = []
        while True:
            x = random.randrange(self.span_x + self.rad, self.gridW - self.span_x - self.rad)
            y = random.randrange(self.span_y + self.rad, self.gridH - self.span_y - self.rad)
            if self.grid[y, x] == 0 and self.cal_distance([x,y], self.goal) >= self.mapH/2:
                self.start = [x, y]
                self.grid[y, x] = 2
                break

    def get_state(self):
        state = self.grid[self.robot_pos[1] - self.span_y: self.robot_pos[1] + self.span_y + 1,
                            self.robot_pos[0] - self.span_x: self.robot_pos[0] + self.span_x + 1]
        state_resized = cv.resize(state, ((self.span_x*2 + 1)*self.map_ratio, (self.span_y*2 + 1)*self.map_ratio), 
                                    interpolation=cv.INTER_AREA)
        return state_resized
            
    def step(self, action):
        new_position = self.robot_pos.copy()
        if action == 0: # move up-left
            new_position[0] -= 1
            new_position[1] -= 1
        elif action == 1: # move up
            new_position[1] -= 1
        elif action == 2: # move up-right
            new_position[0] += 1
            new_position[1] -= 1
        elif action == 3: # move left
            new_position[0] -= 1
        elif action == 4: # move right
            new_position[0] += 1
        elif action == 5: # move down-left
            new_position[0] -= 1
            new_position[1] += 1
        elif action == 6: # move down
            new_position[1] += 1
        elif action == 7: # move down-right
            new_position[0] += 1
            new_position[1] += 1

        # reward = closer distance - transverse distance
        dist = self.cal_distance(self.robot_pos, self.goal) - self.cal_distance(new_position, self.goal)
        line = self.cal_distance(new_position, self.robot_pos)
        reward = dist - line
        terminal = False
        info = {}

        if self.reachGoal(new_position):
            reward = 0
            terminal = True
            info = {'result': 'reach goal'}
        elif self.HitObstacle(new_position):
            reward = -3
            new_position = self.robot_pos.copy()
            info = {'result': 'hit obstacle'}

        self.update_grid(new_position)
        self.path.append(self.robot_pos.copy())
        state = self.get_state()
        self.stack.pop(0)
        self.stack.append(state)

        # Placeholder for truncated value, set to False for now
        truncated = False

        # Cast observation to float32
        obs = np.array(self.stack, dtype=np.float32)

        return obs, reward, terminal, truncated, info


    def reachGoal(self, position):
        if position == self.goal:
            return True
        else:
            return False

    def HitObstacle(self, position):
        if self.grid[position[1], position[0]] == -1:
            return True
        else:
            return False

    def update_grid(self, new_position):
        self.grid[self.robot_pos[1], self.robot_pos[0]] = 0
        self.grid[new_position[1], new_position[0]] = 2
        self.robot_pos = new_position

    def cal_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def render(self, mode='human'):
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                if self.grid[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                elif self.grid[j, i] == 1:
                    img[j, i] = np.array(self.goal_color)
                elif self.grid[j, i] == 2:
                    img[j, i] = np.array(self.robot_color)
                else:
                    img[j, i] = np.array(self.floor_color)

        # scale up image with image ratio
        img_resized = cv.resize(img, (self.gridW*self.image_ratio, self.gridH*self.image_ratio), interpolation=cv.INTER_AREA)

        # draw path
        for i, [x,y] in enumerate(self.path):
            x_point = x*self.image_ratio + self.image_ratio//2
            y_point = y*self.image_ratio + self.image_ratio//2
            cv.circle(img_resized, (x_point, y_point), 3, self.path_color, thickness = -1)
            if i > 0:
                cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.path_color, thickness = 1)
            x_prev = x_point
            y_prev = y_point

        img_croped = img_resized[self.span_y*self.image_ratio: (self.gridH - self.span_y)*self.image_ratio,
                                    self.span_x*self.image_ratio: (self.gridW - self.span_x)*self.image_ratio]

        if mode == 'rgb_array':
            return img_croped  # Return the image array directly
        elif mode == 'human':
            cv.imshow('Mobile Robot Path Planning', img_croped)
            cv.waitKey(1)
        else:
            super(CustomEnv, self).render(mode=mode)  # Call superclass render method with the specified mode




    def close(self):
        cv.destroyAllWindows()

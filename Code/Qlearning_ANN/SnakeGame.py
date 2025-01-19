import numpy as np
import pygame
import random

# Constants
WINDOW_SIZE = 400
GRID_SIZE = 20
FPS = 100

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE * 5, GRID_SIZE * 5)]
        self.food = self._place_food()
        self.direction = (0, -GRID_SIZE)
        self.score = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, (WINDOW_SIZE // GRID_SIZE) - 1) * GRID_SIZE,
                    random.randint(0, (WINDOW_SIZE // GRID_SIZE) - 1) * GRID_SIZE)
            if food not in self.snake:
                return food

    def _get_state(self):
        head = self.snake[0]
        # Check for obstacles in all directions
        danger_up = (head[0], head[1] - GRID_SIZE) in self.snake or head[1] - GRID_SIZE < 0
        danger_down = (head[0], head[1] + GRID_SIZE) in self.snake or head[1] + GRID_SIZE >= WINDOW_SIZE
        danger_left = (head[0] - GRID_SIZE, head[1]) in self.snake or head[0] - GRID_SIZE < 0
        danger_right = (head[0] + GRID_SIZE, head[1]) in self.snake or head[0] + GRID_SIZE >= WINDOW_SIZE

        state = [
            danger_up,            # Danger up
            danger_down,          # Danger down
            danger_left,          # Danger left
            danger_right,         # Danger right
            self.direction == (0, -GRID_SIZE),  # Moving up
            self.direction == (0, GRID_SIZE),   # Moving down
            self.direction == (-GRID_SIZE, 0), # Moving left
            self.direction == (GRID_SIZE, 0),  # Moving right
            head[0] < self.food[0],            # Food is to the right
            head[0] > self.food[0],            # Food is to the left
            head[1] < self.food[1],            # Food is below
            head[1] > self.food[1],            # Food is above
        ]
        return np.array(state, dtype=int)

    def _is_near_body(self, position):
        # Check if the given position is in front, to the right, or to the left of the snake's head.
        head = self.snake[0]
        # front = (head[0] + self.direction[0], head[1] + self.direction[1])
        right = (head[0] - self.direction[1], head[1] + self.direction[0])
        left = (head[0] + self.direction[1], head[1] - self.direction[0])

        return right in self.snake[1:] or left in self.snake[1:]
        # return front in self.snake[1:] or right in self.snake[1:] or left in self.snake[1:]

    def step(self, action):
        # Define possible actions: [UP, DOWN, LEFT, RIGHT]
        directions = [(0, -GRID_SIZE), (0, GRID_SIZE), (-GRID_SIZE, 0), (GRID_SIZE, 0)]
        if action < len(directions) and directions[action] != (-self.direction[0], -self.direction[1]):
            self.direction = directions[action]

        # Move the snake
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        # Check collision
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= WINDOW_SIZE or
            new_head[1] < 0 or new_head[1] >= WINDOW_SIZE):
            return self._get_state(), -100, True

        # Initialize reward
        reward = 0

        # Check if food is eaten
        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            self.score += 1
            reward += 20
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward -= 1

        # Penalize proximity to body
        if self._is_near_body(new_head):
            reward += 1
            
        return self._get_state(), reward, False

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (*segment, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (*self.food, GRID_SIZE, GRID_SIZE))
        pygame.display.flip()

    def launch_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        print("Starting to play ! ")

    def get_state_index(self, state):
        return int("".join(map(str, state)), 2)
import random
import pygame
import numpy as np
import csv
import os
import time

# Constants
WINDOW_SIZE = 800
GRID_SIZE = 20
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

class Qlearning:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = self._place_food()
        self.direction = (0, -1)
        self.score = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head = self.snake[0]
        # Check for obstacles in all directions
        danger_up = (head[0], head[1] - 1) in self.snake or head[1] - 1 < 0
        danger_down = (head[0], head[1] + 1) in self.snake or head[1] + 1 >= GRID_SIZE
        danger_left = (head[0] - 1, head[1]) in self.snake or head[0] - 1 < 0
        danger_right = (head[0] + 1, head[1]) in self.snake or head[0] + 1 >= GRID_SIZE

        state = [
            danger_up,            # Danger up
            danger_down,          # Danger down
            danger_left,          # Danger left
            danger_right,         # Danger right
            self.direction == (0, -1),  # Moving up
            self.direction == (0, 1),   # Moving down
            self.direction == (-1, 0), # Moving left
            self.direction == (1, 0),  # Moving right
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
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        if action < len(directions) and directions[action] != (-self.direction[0], -self.direction[1]):
            self.direction = directions[action]

        # Move the snake
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        # Check collision
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
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
    
    def get_state_index(self, state):
        return int("".join(map(str, state)), 2)
    
    def QlearningTraining(self):
        global alpha, gamma, epsilon, epsilon_decay, epsilon_min, num_episodes
        # Q-table
        state_size = 12
        action_size = 4
        q_table = np.zeros((2 ** state_size, action_size))

        # Training the model
        for episode in range(num_episodes):
            state = self.reset()
            state_index = self.get_state_index(state)
            total_reward = 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, action_size - 1)
                else:
                    action = np.argmax(q_table[state_index])

                next_state, reward, done = self.step(action)
                next_state_index = self.get_state_index(next_state)

                # Update Q-value
                q_table[state_index, action] = (q_table[state_index, action] +
                                                alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action]))

                state_index = next_state_index
                total_reward += reward

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}: Total Score = {self.score}")

        os.makedirs('Code/Qlearning_ANN/Qlearning_data', exist_ok=True)
        with open('Code/Qlearning_ANN/Qlearning_data/Qlearning.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(q_table)

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (*tuple(x * WINDOW_SIZE/GRID_SIZE for x in segment), WINDOW_SIZE/GRID_SIZE, WINDOW_SIZE/GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (*tuple(x * WINDOW_SIZE/GRID_SIZE for x in self.food), WINDOW_SIZE/GRID_SIZE, WINDOW_SIZE/GRID_SIZE))
        pygame.display.flip()

    def openQtableFile(self):
        q_table = []
        with open('Code/Qlearning_ANN/Qlearning_data/Qlearning.csv', 'r') as fichier:
            reader = csv.reader(fichier)
            for row in reader:
                q_table.append(list(map(float, row)))  # Convertit les valeurs en float
        return q_table
    
    def startGameQlearning(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()

        # Run the trained model
        state = self.reset()
        state_index = self.get_state_index(state)
        total_reward = 0
        done = False
        
        q_table = self.openQtableFile()

        while not done:
            action = np.argmax(q_table[state_index])
            state, reward, done = self.step(action)
            state_index = self.get_state_index(state)
            total_reward += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            self.render()
            self.clock.tick(FPS)

        print(f"Game Over! Total Score: {self.score}")
        
        time.sleep(0.5)

        pygame.quit()
import numpy as np
import random

class QLearning:
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001
        self.num_episodes = 1000

        # Q-table
        self.state_size = 12
        self.action_size = 4
        self.q_table = np.zeros((2 ** self.state_size, self.action_size))
        

    def fill_qtable(self, game):

        for episode in range(self.num_episodes):
            self.state = game.reset()
            state_index = game.get_state_index(self.state)
            total_reward = 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.action_size - 1)
                else:
                    action = np.argmax(self.q_table[state_index])

                next_state, reward, done = game.step(action)
                next_state_index = game.get_state_index(next_state)

                # Update Q-value
                self.q_table[state_index, action] = (self.q_table[state_index, action] +
                                                self.alpha * (reward + self.gamma * np.max(self.q_table[next_state_index]) - self.q_table[state_index, action]))

                state_index = next_state_index
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}: Total Score = {game.score}")

    def get_best_action(self, state_index):
        return np.argmax(self.q_table[state_index])
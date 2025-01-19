import pygame
import numpy as np

import SnakeGame as sg
import QLearning as ql
import UserGame as ug
import ANNTraining as ann

import csv


################################### For data collection ###################################
offset = 0
def data_collection(file_index):
    in_path = 'C:/Users/kulbi/OneDrive - Université Nice Sophia Antipolis/polytech_lessons/ROBO5/RL/Projet/ANN_datasets/testlabels'
    out_path = 'C:/Users/kulbi/OneDrive - Université Nice Sophia Antipolis/polytech_lessons/ROBO5/RL/Projet/ANN_datasets/testoutfiles'
    offset_path_in = in_path + str(offset+file_index) +'.csv'
    offset_path_out = out_path + str(offset+file_index) +'.csv'
    return [offset_path_in, offset_path_out]

################################### Custom functions to try out our models ###################################
def play_with_ql_model(file_index):

    state = game.reset()
    state_index = game.get_state_index(state)
    total_reward = 0
    done = False

    # Recording objects
    in_path, out_path = data_collection(file_index)
    user = ug.UserGame(in_path, out_path)
    file = open(in_path, 'w', newline='\n')
    writer = csv.writer(file)

    while not done:

        # Registering Q-learning
        action = ql_agent.get_best_action(state_index)

        # Record the data if wanted
        user.save_state(game.snake, game.food) # Saves an environment picture
        writer.writerow([action / 4])

        # Take the action 
        state, reward, done = game.step(action)
        state_index = game.get_state_index(state)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        game.render()
        game.clock.tick(sg.FPS)


    print(f"Game Over! Total Score: {game.score}")
    file.close()
    user.close_files()

def play_with_ann_model():

    total_reward = 0
    done = False

    in_path, out_path = data_collection(0)
    user = ug.UserGame(in_path, out_path)

    while not done:

        # Infering with ann
        env_image = user.save_state(game.snake, game.food).flatten()
        input_vector = np.expand_dims(env_image, axis=0) # To make the image readable 
        action = ann_agent.infer_from_state(input_vector)

        # Take the action 
        state, reward, done = game.step(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        game.render()
        game.clock.tick(sg.FPS)

    print(f"Game Over! Total Score: {game.score}")

def play_with_user():

    state = game.reset()
    total_reward = 0
    done = False

    in_path, out_path = data_collection(0)
    user = ug.UserGame(in_path, out_path)

    while not done:

        # Gather user data
        action = user.capture_user_input()
        user.save_state(game.snake, game.food)

        # Take the action 
        state, reward, done = game.step(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        game.render()
        game.clock.tick(sg.FPS)

    print(f"Game Over! Total Score: {game.score}")
    user.close_files()


################################### Our objects ###################################
game = sg.SnakeGame()
# ql_agent = ql.QLearning()
ann_agent = ann.ANNTraining()

################################### Filling the Qtable ###################################
# ql_agent.fill_qtable(game)

################################### Setting up ANN agent ###################################
duplicate_path_outputs = 'C:/Users/kulbi/OneDrive - Université Nice Sophia Antipolis/polytech_lessons/ROBO5/RL/Projet/ANN_datasets/outfiles.csv'
duplicate_path_labels = 'C:/Users/kulbi/OneDrive - Université Nice Sophia Antipolis/polytech_lessons/ROBO5/RL/Projet/ANN_datasets/labels.csv'

ann_agent.convert_ds(duplicate_path_outputs, duplicate_path_labels)
ann_agent.train_model(10, 32) # Number of epochs and batch-size

print("Modeled trained ! ")


################################### The game loop ###################################
for i in range(1): # We can loop if we want to save several games from the qtable
    game.launch_game()
    play_with_ann_model()
    pygame.quit()


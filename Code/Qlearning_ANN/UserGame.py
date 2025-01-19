import pygame
import csv
import numpy as np 

"""
Captures the state of the game and the user to export it to 
datasets. These sets will be used to train an ANN model. 
"""
class UserGame:

    def __init__(self, in_file, out_file):
        self.grid_size = 20
        self.output_size = self.grid_size ** 2

        self.previous_action = (0.5,0) # Contains the previous action in case the user doesn't press anything

        self.out_csv = open(out_file,'w', newline='\n')
        self.out_writer = csv.writer(self.out_csv)

        self.in_csv = open(in_file,'w', newline='\n')
        self.in_writer = csv.writer(self.in_csv)

    def __del__(self):
        self.out_csv.close()
        self.in_csv.close()
    
    """
    Gets the user's input and returns it.
    The actions must be in [up, down, right, left]
    Returns the previous action if the user hasn't pressed anything.
    """
    def capture_user_input(self):
        key = pygame.key.get_pressed()

        if key[pygame.K_DOWN]:
            self.in_writer.writerow([0.25]) # [0,0.25] -> down 
            self.in_csv.flush()
            self.previous_action = (0.25,1)
            return 1
            
        elif key[pygame.K_UP] :
            self.in_writer.writerow([0.5])  # [0.25 , 0.5] - up
            self.in_csv.flush()
            self.previous_action = (0.5,0)
            return 0
            
        elif key[pygame.K_LEFT] :
            self.in_writer.writerow([0.75])
            self.in_csv.flush()
            self.previous_action = (0.75,2)
            return 2
            
        elif key[pygame.K_RIGHT] :
            self.in_writer.writerow([1])
            self.in_csv.flush()
            self.previous_action = (1,3)
            return 3
            
        else:
            self.in_writer.writerow([self.previous_action[0]])
            self.in_csv.flush()
            return self.previous_action[1]


    """
    Saves the state to the correspondant csv.
    The state is a picture of the grid in the shape of an array. 
    The values of the array are set as :
        -> 0 : empty 
        -> 400 : the apple
        -> ]0,400[ : the body elements of the snake

        Output : [400 x 1]
    """
    def save_state(self, snake, food):
        current_state = np.zeros([self.output_size, 1])
        body_count = 1

        for body_elem in snake: #body_elem is a tuple (x,y) (0-19, 0-19) 

            index = body_elem[0] + body_elem[1] // self.grid_size # Get the index of the elem on the grid [0 -> 400]
            current_state[index] = body_count / self.output_size # We normalize the output to [0,1] body_count / 400
            body_count+=1

        food_index = food[0] + food[1] // self.grid_size
        current_state[food_index] = 1 # Maximum value 

        self.out_writer.writerow(current_state)
        self.out_csv.flush()

        return current_state
    
    def close_files(self):
        self.out_csv.close()
        self.in_csv.close()
        return 0


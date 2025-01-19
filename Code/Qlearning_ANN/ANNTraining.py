import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import csv
import numpy as np

"""
A class to train, test and infere from an artificial neural network.
This class is intended for the snake game.
The inputs of the class are the training parameters and the datasets.
The outputs are the loss and absolute mean error, in addition to the infered action. 
"""
class ANNTraining :

    def __init__(self):
        self.num_inputs = 400
        self.num_output = 1

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.loss = 0
        self.mae = 0

        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.num_inputs,)),
            # Dense(32, activation='relu', input_shape=(self.num_inputs,)),
            Dense(16, activation='relu'),
            Dense(4, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            Dense(self.num_output, activation='sigmoid')
        ])

        self.model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy', #'mean_squared_error', #
            metrics = ['mean_absolute_error']
        )

    """
    Specifies a dataset (a collection of vectors) to use for training the model.

    """
    def train_model(self, epochs, batch_size):
        # Training sequence
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        # Testing sequence
        self.loss, self.mae = self.model.evaluate(self.x_test, self.y_test)
        return 0

    """
    Converts a dataset (collection of vector) into training and testing sets.
    Partionning is [80-20].
    """
    def convert_ds(self, x_path, y_path):
        x_res = self.csv_to_vector(x_path, self.x_train, self.x_test, self.num_inputs)
        y_res = self.csv_to_vector(y_path, self.y_train, self.y_test, self.num_output)
        self.x_train = x_res[0]
        self.x_test = x_res[1]
        self.y_train = y_res[0]
        self.y_test = y_res[1]

        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()

        return 1
    
    """
    Utility function.
    """
    def csv_to_vector(self, path, training, testing, size):
        file = open(path, 'r', newline='\n')
        reader = csv.reader(file)

        # Create two containers with all the proper sample values
        samples = []
        count = 0
        for row in reader:
            temp = []
            temp.extend([float(value.strip("[]")) for value in row])
            samples.append(temp)
            count+=1

        # Assign portions of the samples to the training and testing data
        unread_samples = 50 # Imposes at least 50 samples in the data
        training_size = int(np.floor(count * 0.8))
        testing_size = count - training_size - unread_samples # We remove the last samples
        training = np.zeros([training_size, size])
        testing = np.zeros([testing_size, size])

        for i in range(count - unread_samples):
            if(i < training_size):
                training[i] = samples[i]
            else:
                testing[i-training_size] = samples[i]

        return (training, testing)

    """
    Predicts the output for a given input. 
    """
    def infer_from_state(self, state):
        prediction = self.model.predict(state)

        # # Debug
        # print(prediction)

        return int(prediction * 4)
    

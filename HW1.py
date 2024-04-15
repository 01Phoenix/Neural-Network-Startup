#Presenting Author's Inro

import time

def animate_text(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.05)  
    print()

text = """
CS3806 Homework #2 - Neural Network
Author: Santosh Bogati
Student ID: 306486
Last Updated: 4/14/2024

Implementing  Neural Network Program.......


"""

animate_text(text)



#Starting the Neural Network Program

import numpy as np

class cs3806_nn:
    
    # initializing the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
       
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
       
        self.lr = learningrate
        
        
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    # Training the Model
    def train(self, inputs_list, targets_list):
       
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
  
        hidden_inputs = np.dot(self.wih, inputs)
        
        hidden_outputs = self.activation_function(hidden_inputs)
        
        
        final_inputs = np.dot(self.who, hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
      
        output_errors = targets - final_outputs
       
        hidden_errors = np.dot(self.who.T, output_errors)
        
       
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
       
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
    #Quering The Neural Network:
    def query(self, inputs_list):
       
        inputs = np.array(inputs_list, ndmin=2).T
        
      
        hidden_inputs = np.dot(self.wih, inputs)
       
        hidden_outputs = self.activation_function(hidden_inputs)
        
      
        final_inputs = np.dot(self.who, hidden_outputs)
       
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

class SOM:
    def __init__ (self, n_inputs, n_outputs, init_learn_rate, init_neighborhood_sigma=0, n_train_cycles):
         """
        Creates a symmetric nxn self organizing map using n_outputs as the dimensions
        All parametrs are set on startup such as epoch, learning rate, neighborhood sigma, prior to the
        initial training on the input data. 

        The weight matrix for this map is initialized to small random values on startup 

        Parameters
        ----------
        n_inputs: int
            number of input nodes
        n_outputs_x: int
            number output nodes in X and Y direction
        init_learn_rate: float
            the initial learning rate parameter which tapers off per epoch
        init_neighborhood_sigma : init (default 0)
            the initial Neighborhood value used to shrink update between epochs. 
            Default = 0 (Winner take all strategy) 
             =/= 0 (Cooperative Strategy)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs_x = n_outputs_x
        self.n_outputs_y = n_outputs_y
        self.init_learn_rate = init_learn_rate
        self.init_neighborhood_sigma = init_neighborhood_sigma
        self.n_train_cycles = n_train_cycles

        #intialize weights and other params that get adjusted during training
        weight_init_range = 0.5
        self.weights = np.random.uniform(-weight_init_range, weight_init_range, (n_hidden, n_outputs))   
        #Keep a count of epoch as thats used to decay our neighborhood and learning rate 
        self.current_epoch = 0

        #Set the current state to the initial learning rate
        self.current_learn_rate = init_learn_rate
        self.current_neighborhood_sigma = init_neighborhood_sigma


        def updateSigmaNeighborhood(self):
            """
            Updated the value of the spread (sigma) in the gaussian neighborhood around a winner node.
            Uses the internal state of the Self organizing maps current nodee as well as 
            """
            new_sigma = self.init_sigma * np.exp(-(self.current_epoch / self.n_train_cycles))

            return new_sigma 
    
        def get_distance(self, a, b):
            """
            Parameters
            -------------
            a,b: 2d numpy array
                 target nodes in the output map specified as a 2-d numpy array
            
            returns
            -----------
            distance: float
                Euclidean distance between node a and b in the map (2-Norm)
            """
            return np.linalg.norm(a, b)

        def updateLearningRate(self):
            """
            Update the learning rate using a time varying decaying exponential using the current epoch
            and the max amount of epochs 
            """
            self.current_learn_rate = self.init_learn_rate * np.exp(- self.current_epoch / self.n_train_cycles)          


        def updateNeighborHood(self, winning_node, target_node):
            """
            Updated the neighborhood around the winning node and target nodes around it. This applies the update
            procedure by invoking the following chain  

            Parameters
            ----------
            winning_node : np.array(x,y) 
                2x1 vector of position of the winning node in the output map
            target_node:  np.array(x,y)
                2x1 vector of position of the target node in the output map

            """
            epoch = self.current_epoch
            self.current_neighborhood = self.init_neighborhood  * np.exp(- self.get_distance(winning_node, target_node) / (2 * self.updateSigmaNeighborhood() ^2))
            


       def train()

#Create the self organizing map given the following parameters of our data we wish to train and output
# Use parameters supplied by the assignment 4 to initialize it
n_input = 3 # 3 inputs since each value represents an RGB encoded valuea
n_output = 100

init_learn_rate = 0.8 
init_neighborhood = 0.1
max_epochs = 1000

sigma_list = [1, 10 ,30, 50, 70]

init_sigma = 1
#Generate an n_output x n_output (square) feature map
som_map = SOM(n_input, n_output, init_learn_rate, init_sigma, max_epochs)




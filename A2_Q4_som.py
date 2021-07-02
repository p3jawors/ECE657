import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

class SOM:
    def __init__(self, n_inputs, n_outputs, init_learn_rate, init_neighborhood_sigma=0, n_epochs=1000):
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

        n_epochs: int
        the max amoutn of training epochs used for the output map.
        prev_state: class SOM
        Push the previous state of another similar sized mapped to this instance.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.init_learn_rate = init_learn_rate
        self.init_neighborhood_sigma = init_neighborhood_sigma
        self.n_epochs = n_epochs

        #intialize weights and other params that get adjusted during training should have a n_input x (n_output x n_output) tensor for our matrix
        weight_init_range = 0.1
        self.weights = np.random.uniform(-weight_init_range, weight_init_range, (self.n_inputs, self.n_outputs, self.n_outputs))


        # Keep a count of epoch as thats used to decay our neighborhood and learning rate
        # current_epoch as well as n_epochs is used to determine time with respect to epoch in the SOM
        self.current_epoch = 0

        #Set the current state to the initial learning rate
        self.current_learn_rate = init_learn_rate
        self.current_neighborhood_sigma = init_neighborhood_sigma


    def updateSigmaNeighborhood(self):
        """
        Updated the value of the spread (sigma) in the gaussian neighborhood around a winner node.
        Uses the internal state of the Self organizing maps current nodee as well as
        """
        new_sigma = self.init_sigma * np.exp(-(self.current_epoch / self.n_epochs))
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
        return np.linalg.norm(a[:, None, None] - b, axis=0)

    def updateLearningRate(self):
        """
        Update the learning rate using a time varying decaying exponential using the current epoch
        and the max amount of epochs
        """
        self.current_learn_rate = self.init_learn_rate * np.exp(- self.current_epoch / self.n_epochs)


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
        self.current_neighborhood = (
                self.init_neighborhood
                * np.exp(
                    - self.get_distance(winning_node, target_node)
                    / (2 * self.updateSigmaNeighborhood() ^2)
                    )
                )


    def fit(self, training_data, n_epochs=None):
        """
        Train the map over the given set of cycles

        Parameters
        -------------
        training_data: np.array (r,g,b)
            A 3x1 input array with the data values to train the map

        number_epochs: int
            The number of desired training epochs to progress over the current map
        """
        if n_epochs is None:
            n_epochs = self.n_epochs

        for data_point in training_data:
            # Step 3: Get distances and select winning index
            dist = self.get_distance(data_point, self.weights)

            # argmin returns the index on the flatten array, unravel_index returns the 2d value
            min_index = np.unravel_index(np.argmin(dist), dist.shape)
            print('min index: ', min_index)
            print(dist[min_index])

            # Step 4: Update weight matrix

            # Step 5: Update learning rate and neighbourhood
            raise Exception


    def get_state(self):
        """
        Get the current state of the weight matrix as a n_intput x n_output x n_output numpy array
        Used to see the resulting RGB value or the resultant output node
        """
        return self.weights


#Generate input data for the color scheme of 24 colours
# save data to dict with the color name for simpler plotting later on
colours = {
        'red': [255, 0, 0],
        'maroon': [128, 0, 0],
        'tomato': [255, 99, 71],
        'crimson': [220, 20, 60],

        'green': [0, 128, 0],
        'lime': [0, 255, 0],
        'pale_green': [152, 251, 152],
        'sea_green': [46, 139, 87],

        'blue': [0, 0, 255],
        'royal_blue': [65, 105, 225],
        'midnight_blue': [25, 25, 112],
        'sky_blue': [135, 206, 235],


        'yellow': [255, 255, 0],
        'gold': [255, 215, 0],
        'golden_rod': [218, 165, 32],
        'khaki': [240, 230, 140],


        'teal': [0, 128, 128],
        'turquoise': [64, 224, 208],
        'cyan': [0, 255, 255],
        'aqua_marine': [127, 255, 212],

        'pink': [255, 192, 203],
        'hot_pink': [255, 105, 180],
        'deep_pink': [255, 20, 147],
        'magenta': [255, 0, 255]
}

training_data = []
for key in colours:
    training_data.append(np.asarray(colours[key])/255)
training_data = np.asarray(training_data)

#Create the self organizing map given the following parameters of our data we wish to train and output
# Use parameters supplied by the assignment 4 to initialize it
n_input = 3 # 3 inputs since each value represents an RGB encoded valuea
n_output = 100

init_learn_rate = 0.8
init_neighborhood = 0.1
max_epochs = 1000

sigma_list = [1, 10 ,30, 50, 70]
epoch_states = [20, 40, 100, 1000]

# Get a series of output maps for each sigma_list value and append the map to a resultant output weight map we store
# For later output and viewing.
# for sigma in sigma_list:
sigma = 0.8
    #Generate an n_output x n_output (square) feature map
som_map = SOM(n_input, n_output, init_learn_rate, sigma, max_epochs)
# prev_epoch_count = 0 #Prev num epochs completed

#Train the Self organizing map on the data using the unsupervised model over a number of epochs
# for epoch_count in epoch_states:
# epochs_left_to_train = epoch_count - prev_epoch_count

som_map.fit(training_data)
# current_train_state.append(som.get_state())
#
# prev_epoch_count = prev_epoch_count + epochs_left_to_train


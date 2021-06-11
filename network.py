import numpy as np
import csv
from sklearn.model_selection import train_test_split

class Neuron:
    """
    A place to store activation functions to pass Network on init
    """
    def sigmoid(x):
        """
        Takes a 1D float as input and returns a 1D float
        Parameters
        ----------
        x: 1D float
        """
        y = 1/(1 + np.exp(-x))
        return y


    def tanh(x):
        """
        Takes a 1D float as input and returns a 1D float
        Parameters
        ----------
        x: 1D float
        """
        y = (np.exp(x) - np.exp(x)) / (np.exp(x) + np.exp(-x))
        return y

class Helper:
    """
    Useful stuff we dont want to pack in the network itself
    """

    def extract_csv(csv_data_path, delim):
        """
        extract CSV data given a specific delimiter and return it as a list
        """    
        output_data_blob = []

        with open(csv_data_path, newline=delim) as fp:
            reader = cvs.reader(fp, delimiter=delim)
            for II, row in enumerate(reader):
                data = list(map(float, row))
                output_data_blob.append(data)

        return output_data_blob

class Layer:
    """
    Handles initializing weights and biases, and calculating feedforward layer output
    """
    def __init__(self, n_neurons, activation_fn, weights=None, init_range=0.1):
        """

        Parameters
        ----------
        n_neurons: int
            number of nodes in the layer
        activation_fn: function
            The activations function of the node, set to be the same for all nodes in the layer.
            The function should take a 1D float as input and return a 1D float
        weights: (2, n_neurons) float array, Optional (Default: None)
            The multiplicative weights are row 0 and the biases are row 1.
            Optional to pass in trained weights, if None will initialize weights and biases to
            values between +init_range and -init_range.
        init_range: float, Optional (Default: 0.1)
            Used as positive and negative value range in np.random.uniform when initializing
            weights and biases (only if weights are None)
        """
        self.n_neurons = n_neurons
        self.activation_fn = activation_fn
        if weights is None:
            weights = np.random.uniform(-init_range, init_range, (2, n_neurons))

        self.weights = weights

    def forward(self, layer_input):
        """
        Takes the output of the previous layer as input and sums it, then scales
        by weights and adds biases for each neuron in the layer. Returns an output
        of the pre-summed layer output (shape (n_neurons,))

        Parameters
        ----------
        layer_input: np.array of floats, shape (n_neurons_prev_layer,)
        """
        # apply the linear and nonlinear transforms on our input
        # return the array of outputs len n_neurons
        layer_out = []
        sum_input = sum(layer_input)
        for ii in range(0, self.n_neurons):
            layer_out.append(
                    self.activation_fn(
                        sum_input * self.weights[0, ii] + self.weights[1, ii])
            )

        return layer_out

    def modify_weights(self, updates):
        pass
        
    

class Network:
    """
    Creates our network structure assuming the shape of n_inputs > n_hidden > n_outputs

    Parameters
    ----------
    n_inputs: int
        Length of input
    n_hidden: int
        Number of neurons in the hidden layer
    n_outputs: int
        Number of prediction categories
    activation_fn: function
        activation function for the neurons of the hidden layer and output
    learning_rate: float
            value between 0-1 to adjust learning between training sequences
    weights: tuple of float arrays shape ((2, n_hidden), (2, n_outputs))
    """
    #TODO have not setup weight loading, can have it as a (2, n_hidden + n_outputs) and slice, but this may be more legible
    def __init__(self, n_inputs, n_hidden, n_outputs, activation_fn, learning_rate=0.1, weights=None):
        # Instantiate our layers and initialize weights & biases
        self.layers = []
        self.layers.append(Layer(
            n_neurons=n_hidden,
            activation_fn=activation_fn,
            weights=weights)
        )
        self.layers.append(Layer(
            n_neurons=n_outputs,
            activation_fn=activation_fn,
            weights=weights)
        )

        self.learning_rate = learning_rate



    def feedforward(self, layer_input):
        # propagate data forward
        for layer in self.layers:
            # overwrite our input with the output of the last layer
            layer_input = layer.forward(layer_input)

        return layer_input


    def backprop():
        
        pass


    def calc_error():
        pass


    def run_inference(self, data_csv):
        predictions = []
        with open(data_csv, newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for ii, row in enumerate(reader):
                #NOTE this is a hack to test without reading all 24k vals
                if ii > 5:
                    break
                data = list(map(float, row))
                predictions.append(self.feedforward(data))

        return predictions


    def train(self, train_data, target_data, num_epochs = 5, batch_size = 100, verbose = True):
        """
        take the current training data and run an inferance on the input then perform a backpropogation
        on the network and upate the weights 

        train-data: a list of values used for training, must be same size as target data
        target_data: target data matching the input data for desired target output must be same size as train
        num_epochs: The number of training cycles for the input training data used
        batch size: the number of samples from the training data to use 
        verbose: Show the information of the system per training epoch 
        
        """
        pass


    def export_weights():
        """
        export the weights of the network as a numpy array
        """
        pass

if __name__ == "__main__":
    #NOTE number of neurons set arbitrarily atm

    #Get training data and targets
    train_raw_data = Helper.extract_csv('train_data.csv', "'")
    target_raw_data  = Helper.extract_csv('train_labels.csb', "'")

    net = Network(
            n_inputs=784,
            n_hidden=100,
            n_outputs=4,
            activation_fn=Neuron.sigmoid
    )

    ## Perform a 90 - 5 - 5   Train - Validation - Test set of data for training and validating neural network
    #Get a 90% - 10% split of train_val data
    train_data, y_data, train_labels, y_labels = train_test_split(train_raw_data, target_raw_data, testsize = 0.1, random_state = 42)

    #Get 50% 50% split of remaining val data 
    test_data, val_data , test_labels, val_labels = train_test_split(y_data, y_labels, testsize= 0.5 , random_state = 42)


    predictions = net.run_inference('train_data.csv')
    print('Predictions: ', predictions)
    #TODO load data
    #TODO split train/val/test
    #TODO call train
    #TODO call val
    #TODO call test

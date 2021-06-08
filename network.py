import numpy as np
class Neuron:
    """
    A place to store activation functions to pass Network on init
    """
    def sigmoid(x):
        y = 1/(1 + np.exp(-x))
        return y

class Layer:
    """
    Handles initializing weights and biases, and calculating feedforward layer output
    """
    def __init__(n_neurons, activation_fn, weights=None):
        self.n_neurons = n_neurons
        self.activation_fn = activation_fn
        if weights is None:
            #TODO add function to initialize weights and biases
            weights = init_weights(n_neurons)

        #NOTE should we pass biases in as well or pack them with 'weights'?
        self.weights = weights
        self.biases = biases


    def forward(layer_input):
        # apply the linear and nonlinear transforms on our input
        # return the array of outputs len n_neurons
        layer_out = []
        sum_input = sum(layer_input)
        for ii in range(0, self.n_neurons):
            layer_out.append(
                    self.activation_fn(
                        sum_input * self.weights[ii] + self.biases[ii])
            )

        return layer_out


class Network:
    def __init__(self, n_layers, n_neurons, activation_fn, weights=None):
        # Instantiate our layers and initialize weights & biases
        self.layers = []
        for layer in n_layers:
            self.layers.append(Layer(
                n_neurons=n_neurons,
                activation_fn=activation_fn,
                weights=weights)
            )


    def feedforward(self, layer_input):
        # propagate data forward
        for layer in self.layers:
            # overwrite our input with the output of the last layer
            layer_input = layer.forward(layer_input)

        self.output = layer_input


    def backprop():
        pass


    def calc_error():
        pass


if __name__ == "__main__":
    #NOTE number of neurons set arbitrarily atm
    net = Network(
            n_layers=1,
            n_neurons=10,
            activation_fn=Neuron.sigmoid
    )


    #TODO load data
    #TODO split train/val/test
    #TODO call train
    #TODO call val
    #TODO call test

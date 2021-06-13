import numpy as np
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from acc_calc import accuracy
import timeit
import matplotlib.pyplot as plt


class Helper:
    """
    Useful stuff we dont want to pack in the network itself
    """

    def extract_csv(csv_data_path, delim=',', newline=''):
        """
        extract CSV data given a specific delimiter and return it as a list
        """
        data = []
        with open(csv_data_path, newline=newline) as fp:
            reader = csv.reader(fp, delimiter=delim)
            for II, row in enumerate(reader):
                readline = list(map(float, row))
                data.append(readline)

        return data

    def search_params(hidden_num_list, activation_list=None):
        #Perform a search of the values accross number of nodes and check using the val dataset to find
        #the best set of accuracy, Trying differnt activation as well as hidden number of nodes
        #Omitting k fold cross validation for  the dataset as well.
        result = []
        net_weights = []
        net_activation = []
        net_hidden_cnt = []

        if activation_list is None:
            #TODO have to account for this being the same length as hidden num list if we want to vary this
            activation_list = np.array([Nonlinarities.sigmoid, Nonlinearities.pd_sigmoid])

        for item in hidden_num_list:
            # for active_fn in activation_list:

                #Create a MLP network with teh given parameters we'd like to try
                net = Network(
                    n_inputs=784,
                    n_hidden=item,
                    n_outputs=4,
                    activation_fn=activation_list
                )

                #Perform a training sequence on the input data and the given labels
                net.train(train_data, train_labels)

                #Log accuracy from the given run after training and run a feedforward test using the validation data for this given
                #run.
                result.append(net.feedforward(val_data, val_labels))

                #snapshot the configuration of the weights and activation of the network
                net_weights.append(net.export_weights())
                net_activation.append(active_fn)
                net_hidden_cnt.append(item)


        best_run_idx = 0
        index = 0
        #Find the max accuracy
        for run in result:
            if run >= result[best_run_idx]:
                best_run_idx = index
            index = index + 1


        #Use the best result accuracy on the data itself
        net = Network(
            n_inputs=784,
            n_hidden=net_hidden_cnt[best_run_idx],
            n_outputs=4,
            activation_fn=net_activation[best_run_idx],
            weights=net_weights[best_run_idx]
        )


        predictions = net.inference(test_data)
        print('Predictions: ', predictions)

    def load_weights(path):
        # work around for possible numpy savez
        # https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        print('Loading network weights')
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        trained_weights = np.load(path)
        weights = trained_weights['weights']
        biases = trained_weights['biases']
        return weights, biases


class Nonlinearities():

    def sigmoid(x):
        """
        Takes a 1D float as input and returns a 1D float
        Parameters
        ----------
        x: 1D float
        """
        y = 1/(1 + np.exp(-x))
        return y

    def pd_sigmoid(y):
        return y * (1-y)



    def tanh(x):
        """
        Takes a 1D float as input and returns a 1D float
        Parameters
        ----------
        x: 1D float
        """
        y = (np.exp(x) - np.exp(x)) / (np.exp(x) + np.exp(-x))
        return y


class Neuron:
    """
    A place to store activation functions to pass Network on init
    """
    def __init__(self, weights, bias, nonlinearity=None):
        self.weights = weights
        self.bias = bias
        if nonlinearity is None:
            self.nonlinearity = Nonlinearities.sigmoid
            self.pd_nonlinearity = Nonlinearities.pd_sigmoid
        else:
            self.nonlinearity = nonlinearity[0]
            self.pd_nonlinearity = nonlinearity[1]

    def linear_ops(self, layer_input):
        output = np.dot(np.asarray(layer_input), np.asarray(self.weights)) + self.bias
        # print('neuron linear ops: ', output)
        return output

    def forward(self, layer_input):
        self.layer_input = layer_input
        self.neuron_output = self.nonlinearity(self.linear_ops(layer_input))
        # print('neuron output: ', self.neuron_output)
        return self.neuron_output

    def pd_error_wrt_network_input(self, target):
        # a = self.pd_error_wrt_output(target=target)
        # print('a: ', a)
        # b = self.pd_nonlinearity(self.neuron_output)
        # print('b: ', b)
        return self.pd_error_wrt_output(target=target) * self.pd_nonlinearity(self.neuron_output)

    def pd_error_wrt_output(self, target):
        """
        Partial derivative of the error wrt the desired output
        """
        return -(target - self.neuron_output)

    def pd_input_wrt_weight(self, conn_index):
        return self.layer_input[conn_index]


class Layer:
    """
    Handles initializing weights and biases, and calculating feedforward layer output
    """
    def __init__(self, nonlinearity, weights, biases):
        """

        Parameters
        ----------
        nonlinearity: function
            The activations function of the node, set to be the same for all nodes in the layer.
            The function should take a 1D float as input and return a 1D float
        weights: (n_neurons, n_inputs) float array
            pretrained or initialized weights for each neuron in the layer
        biases: (n_neurons, ) float array
            pretrained or initialized biases for each neuron in the layer
        """
        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.nonlinearity = nonlinearity
        self.n_neurons = self.weights.shape[0]
        self.neurons = []
        for ii in range(0, self.n_neurons):
            self.neurons.append(Neuron(self.weights[ii], self.biases[ii], self.nonlinearity))

    def forward(self, layer_input):
        """
        Takes the output of the previous layer as input and sums it, then scales
        by weights and adds biases for each neuron in the layer. Returns an output
        of the pre-summed layer output (shape (n_neurons,))

        Parameters
        ----------
        layer_input: np.array of floats, shape (n_neurons_prev_layer,)
        """
        layer_output = []
        for neuron in self.neurons:
            layer_output.append(neuron.forward(layer_input))

        return np.asarray(layer_output)

    def get_weights(self):
        weights = []
        biases = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
            biases.append(neuron.bias)

        return (weights, biases)


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
    init_range: float, Optional (Default: 0.1)
        Used as positive and negative value range in np.random.uniform when initializing
        weights and biases (only if weights are None)
    """
    def __init__(
            self,
            n_inputs,
            n_hidden,
            n_outputs,
            nonlinearity_hidden=None,
            nonlinearity_output=None,
            learning_rate=0.1,
            weights_hidden=None,
            weights_output=None,
            biases_hidden=None,
            biases_output=None,
            weight_init_range=0.1,
            bias_init_range=0.1):

        if weights_hidden is None:
            weights_hidden = np.random.uniform(-weight_init_range, weight_init_range, (n_hidden, n_inputs))

        if weights_output is None:
            weights_output = np.random.uniform(-weight_init_range, weight_init_range, (n_outputs, n_hidden))

        if biases_hidden is None:
            biases_hidden = np.random.uniform(-bias_init_range, bias_init_range, n_hidden)

        if biases_output is None:
            biases_output = np.random.uniform(-bias_init_range, bias_init_range, n_outputs)

        # Instantiate our layers and initialize weights & biases
        self.layers = []
        self.layers.append(Layer(
            nonlinearity=nonlinearity_hidden,
            weights=weights_hidden,
            biases=biases_hidden
            )
        )

        self.layers.append(Layer(
            nonlinearity=nonlinearity_output,
            weights=weights_output,
            biases=biases_output
            )
        )

        self.learning_rate = learning_rate
        self.network_output = None
        self.error = None


    def feedforward(self, layer_input, target=None):
        # propagate data forward
        for layer in self.layers:
            # overwrite our input with the output of the last layer
            layer_input = layer.forward(layer_input)

        self.network_output = layer_input

        # If given a target, calculate the error
        if target is not None:
            self.error = self.calc_error(target=target, network_output=self.network_output)
        else:
            self.error = None

        return (self.network_output, self.error)


    def train(self, training_inputs, training_targets, batch_size=1):
        """
        take the current training data and run an inferance on the input then perform a backpropogation
        on the network and upate the weights

        train-data: a list of values used for training, must be same size as target data
        target_data: target data matching the input data for desired target output must be same size as train
        num_epochs: The number of training cycles for the input training data used
        batch size: the number of samples from the training data to use
        verbose: Show the information of the system per training epoch
        """

        training_error = []

        print('Starting training, sit tight!')
        for ii in tqdm(range(0, len(training_inputs))):
            _, err = self.feedforward(training_inputs[ii], training_targets[ii])
            training_error.append(err)


            # STEP 1: Calculate weight updates for output layer
            pd_error_wrt_output_neuron = []
            for nn, neuron in enumerate(self.layers[-1].neurons):
                dw = neuron.pd_error_wrt_network_input(target=training_targets[ii][nn])
                # print('dw: ', dw)
                pd_error_wrt_output_neuron.append(dw)

            # STEP 2: Calculate weight updates for hidden layer
            pd_error_wrt_hidden_neuron = []
            for hh, hidden_neuron in enumerate(self.layers[0].neurons):
                error_wrt_hidden_neuron_sum = 0

                for oo, output_neuron in enumerate(self.layers[-1].neurons):
                    error_wrt_hidden_neuron_sum += pd_error_wrt_output_neuron[oo] * output_neuron.weights[hh]

                pd_error_wrt_hidden_neuron.append(
                        error_wrt_hidden_neuron_sum * hidden_neuron.pd_nonlinearity(hidden_neuron.neuron_output)
                )

            # Update weights for output neurons
            for oo, output_neuron in enumerate(self.layers[-1].neurons):
                for wj, weight in enumerate(output_neuron.weights):
                    dw = pd_error_wrt_output_neuron[oo] * output_neuron.pd_input_wrt_weight(wj)
                    output_neuron.weights[wj] -= self.learning_rate * dw

            # Update weights for hidden neurons
            for hh, hidden_neuron in enumerate(self.layers[0].neurons):
                for wi, weight in enumerate(hidden_neuron.weights):
                    dw = pd_error_wrt_hidden_neuron[hh] * hidden_neuron.pd_input_wrt_weight(wi)
                    hidden_neuron.weights[wi] -= self.learning_rate * dw

        return training_error


    def calc_error(self, target, network_output):
        error = 0
        for ii, output_dim in enumerate(network_output):
            error += 0.5 * (target[ii] - output_dim)**2

        return error


    def inference(self, inference_inputs, inference_targets):
        inference_error = []
        for ii in tqdm(range(0, len(inference_inputs))):
            _, err = self.feedforward(inference_inputs[ii], inference_targets[ii])
            inference_error.append(err)

        return np.asarray(inference_error)


    def export_weights(self):
        """
        export the weights of the network as a numpy array
        """
        weights = []
        biases = []
        for layer in self.layers:
            weight_bias = layer.get_weights()
            weights.append(weight_bias[0])
            biases.append(weight_bias[1])

        return (weights, biases)


if __name__ == "__main__":
    #NOTE number of neurons set arbitrarily atm

    #Get training data and targets
    print('Loading data from csv file...')
    train_raw_data = Helper.extract_csv('train_data.csv')
    target_raw_data  = Helper.extract_csv('train_labels.csv')

    print('Splitting data into test/val/train')
    ## Perform a 70 - 15 - 15   Train - Validation - Test set of data for training and validating neural network
    #Get a 70% - 30% split of train_val data
    train_data, y_data, train_labels, y_labels = train_test_split(
            train_raw_data, target_raw_data, test_size=0.3, random_state=42)

    #Get 50% 50% split of remaining val data
    test_data, val_data , test_labels, val_labels = train_test_split(
            y_data, y_labels, test_size=0.5 , random_state=42)

    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    val_data = np.asarray(val_data)

    print('raw data: ', np.asarray(train_raw_data).shape)
    print('train data: ', train_data.shape)
    print('val data: ', val_data.shape)
    print('test data: ', test_data.shape)

    # Set your options for hidden neurons and learning rates
    n_hidden_options = [75]
    # learning_rates = [5e-4, 5e-3, 5e-2, 0.1, 0.25, 0.5]
    # learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    learning_rates = [1e-4]

    # these are used for the +/- limits of the random distribution drawn from for initialization of weights and biases
    weight_init_range = 0.1
    bias_init_range = 0.3

    for learning_rate in learning_rates:
        fig = plt.figure(figsize=(8, 12))
        a1 = plt.subplot(211)
        a1.set_title('Training Error Learning Rate %.6f' % learning_rate)
        a2 = plt.subplot(212)
        a2.set_title('Validation Error Learning Rate %.6f' % learning_rate)

        for n_hidden in n_hidden_options:
            print('Instantiating network with %i hidden neurons and learing rate of %.6f...' % (n_hidden, learning_rate))
            # Instantiate our network

            weights_loc = 'first_pass-Pawel_and_Ted_weights.npz'
            weights, biases = Helper.load_weights(weights_loc)

            net = Network(
                    n_inputs=784,
                    n_hidden=n_hidden,
                    n_outputs=4,
                    nonlinearity_hidden=[Nonlinearities.sigmoid, Nonlinearities.pd_sigmoid],
                    nonlinearity_output=[Nonlinearities.sigmoid, Nonlinearities.pd_sigmoid],
                    learning_rate=learning_rate,
                    weights_hidden=weights[0],
                    weights_output=weights[1],
                    biases_hidden=biases[0],
                    biases_output=biases[1],
                    # weights_hidden=None,
                    # weights_output=None,
                    # biases_hidden=None,
                    # biases_output=None,
                    weight_init_range=weight_init_range,
                    bias_init_range=bias_init_range
            )

            # Run Training
            start = timeit.default_timer()
            train_errors = net.train(
                    training_inputs=train_data,
                    training_targets=train_labels
            )
            runtime = timeit.default_timer() - start
            print('Training took %.2f min' % (runtime/60))

            a1.plot(train_errors, label='n_hidden:%i\nmin train_err: %.4f' % (n_hidden, min(train_errors)))
            plt.legend()

            # Run inference on trained network, using validation set
            start = timeit.default_timer()
            val_errors = net.inference(
                    inference_inputs=val_data,
                    inference_targets=val_labels
            )
            runtime = timeit.default_timer() - start
            print('Validation Inference took %.2f min' % (runtime/60))

            a2.plot(val_errors, label='n_hidden:%i\nmin val_err: %.4f' % (n_hidden, min(val_errors)))
            plt.legend()

            weights, biases = net.export_weights()
            np.savez_compressed('Pawel_and_Ted_weights.npz', weights=weights, biases=biases)

        plt.savefig('learning_rate%.7f.png' % learning_rate)
        # plt.show()

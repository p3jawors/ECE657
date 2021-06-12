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


    def train(self, training_inputs, training_targets):
        training_error = []
        #TODO we'll add batching later
        print('Starting training, sit tight!')
        for ii in tqdm(range(0, len(training_inputs))):
            _, err = self.feedforward(training_inputs[ii], training_targets[ii])
            training_error.append(err)


            # STEP 1: Calculate weight updates for output layer
            # print('1: calc output weights')
            pd_error_wrt_output_neuron = []
            for nn, neuron in enumerate(self.layers[-1].neurons):
                dw = neuron.pd_error_wrt_network_input(target=training_targets[ii][nn])
                # print('dw: ', dw)
                pd_error_wrt_output_neuron.append(dw)

            # STEP 2: Calculate weight updates for hidden layer
            # print('2: calc hidden weights')
            pd_error_wrt_hidden_neuron = []
            for hh, hidden_neuron in enumerate(self.layers[0].neurons):
                error_wrt_hidden_neuron_sum = 0

                for oo, output_neuron in enumerate(self.layers[-1].neurons):
                    error_wrt_hidden_neuron_sum += pd_error_wrt_output_neuron[oo] * output_neuron.weights[hh]
                    # print('1: ', pd_error_wrt_output_neuron[hh])
                    # print('2: ', output_neuron.weights[oo])
                    # print('prod: ', error_wrt_hidden_neuron_sum)

                pd_error_wrt_hidden_neuron.append(
                        error_wrt_hidden_neuron_sum * hidden_neuron.pd_nonlinearity(hidden_neuron.neuron_output)
                )
                # print('3: ', error_wrt_hidden_neuron_sum)
                # print('4: ', hidden_neuron.pd_nonlinearity(hidden_neuron.neuron_output))
                # print('prod: ', pd_error_wrt_hidden_neuron)

                # print('a: ', pd_error_wrt_hidden_neuron)
                # print('b: ', error_wrt_hidden_neuron_sum)
                # print('c: ', hidden_neuron.pd_nonlinearity(hidden_neuron.neuron_output))
            # Update weights for output neurons
            for oo, output_neuron in enumerate(self.layers[-1].neurons):
                for wj, weight in enumerate(output_neuron.weights):
                    # print('should be one thing: ', pd_error_wrt_output_neuron[oo])
                    # print(output_neuron.pd_input_wrt_weight(wj))
                    dw = pd_error_wrt_output_neuron[oo] * output_neuron.pd_input_wrt_weight(wj)
                    # print(output_neuron.weights[wj])
                    # print(output_neuron.weights)
                    # print(self.learning_rate)
                    # print(dw)
                    output_neuron.weights[wj] -= self.learning_rate * dw
                    # print('FINAL OUTPUT: ', output_neuron.weights[wj])

            # Update weights for hidden neurons
            for hh, hidden_neuron in enumerate(self.layers[0].neurons):
                for wi, weight in enumerate(hidden_neuron.weights):
                    dw = pd_error_wrt_hidden_neuron[hh] * hidden_neuron.pd_input_wrt_weight(wi)
                    hidden_neuron.weights[wi] -= self.learning_rate * dw
                    # print('FINAL HIDDEN: ', hidden_neuron.weights[wi])

        return training_error


    def calc_error(self, target, network_output):
        error = 0
        for ii, output_dim in enumerate(network_output):
            error += 0.5 * (target[ii] - output_dim)**2

        return error


    # def run_inference(self, data_csv):
    #     predictions = []
    #     with open(data_csv, newline='') as fp:
    #         reader = csv.reader(fp, delimiter=',')
    #         for ii, row in enumerate(reader):
    #             #NOTE this is a hack to test without reading all 24k vals
    #             if ii > 5:
    #                 break
    #             data = list(map(float, row))
    #             predictions.append(self.feedforward(data))
    #
    #     return predictions
    #
    #
    # def train(self, train_data, target_data, num_epochs = 5, batch_size = 100, verbose = True):
    #     """
    #     take the current training data and run an inferance on the input then perform a backpropogation
    #     on the network and upate the weights
    #
    #     train-data: a list of values used for training, must be same size as target data
    #     target_data: target data matching the input data for desired target output must be same size as train
    #     num_epochs: The number of training cycles for the input training data used
    #     batch size: the number of samples from the training data to use
    #     verbose: Show the information of the system per training epoch
    #
    #     """
    #     pass
    #
    #
    # def export_weights():
    #     """
    #     export the weights of the network as a numpy array
    #     """
    #     pass

if __name__ == "__main__":
    #NOTE number of neurons set arbitrarily atm

    #Get training data and targets
    print('Loading data from csv file...')
    train_raw_data = Helper.extract_csv('train_data.csv')
    target_raw_data  = Helper.extract_csv('train_labels.csv')

    # #NOTE HACK TO CUTE DOWN DATASIZE WHILE TESTING
    # print('HACK ADDED TO CUT DOWN DATA SIZE WHILE TESTING HERE')
    # train_raw_data = np.asarray(train_raw_data)[:1000, :]
    # target_raw_data = np.asarray(target_raw_data)[:1000, :]

    print('Splitting data into test/val/train')
    ## Perform a 90 - 5 - 5   Train - Validation - Test set of data for training and validating neural network
    #Get a 90% - 10% split of train_val data
    train_data, y_data, train_labels, y_labels = train_test_split(
            train_raw_data, target_raw_data, test_size=0.1, random_state=42)

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
    n_hidden_options = [10, 20, 60]#30, 40, 50, 60]
    learning_rates = [0.1, 0.5]

    # these are used for the +/- limits of the random distribution drawn from for initialization of weights and biases
    weight_init_range = 0.1
    bias_init_range = 0.3

    for learning_rate in learning_rates:
        plt.figure()
        plt.title('Training Error Learning Rate %.3f' % learning_rate)

        for n_hidden in n_hidden_options:
            print('Instantiating network with %i hidden neurons and learing rate of %.3f...' % (n_hidden, learning_rate))
            # Instantiate our network
            net = Network(
                    n_inputs=784,
                    n_hidden=n_hidden,
                    n_outputs=4,
                    nonlinearity_hidden=[Nonlinearities.sigmoid, Nonlinearities.pd_sigmoid],
                    nonlinearity_output=[Nonlinearities.sigmoid, Nonlinearities.pd_sigmoid],
                    learning_rate=learning_rate,
                    weights_hidden=None,
                    weights_output=None,
                    biases_hidden=None,
                    biases_output=None,
                    weight_init_range=weight_init_range,
                    bias_init_range=bias_init_range):
            )

            start = timeit.default_timer()
            errors = net.train(
                    training_inputs=train_data,
                    training_targets=train_labels
            )
            runtime = timeit.default_timer() - start
            print('Training took %.2f min' % (runtime/60))

            plt.plot(errors, label='n_hidden:%i\nFinal Error: %.4f' % (n_hidden, errors[-1]))

        plt.legend()
        plt.savefig('learning_rate%.3f.png' % learning_rate)
        plt.show()

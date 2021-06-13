import numpy as np
from network import Helper, Nonlinearities, Neuron, Layer, Network
from sklearn.model_selection import train_test_split

STUDENT_NAME = 'Ted_Themistokleous-Pawel_Jaworski'
STUDENT_ID = '20302981-20392961'

def test_mlp(data_file=None):

    # work around for possible numpy savez
    # https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    print('Loading network weights')
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    trained_weights = np.load('Pawel_and_Ted_weights.npz')
    weights = trained_weights['weights']
    biases = trained_weights['biases']

    if data_file is None:
        print('No data file was passed in, running with test split from training...')

        #Get training data and targets
        print('Loading data from csv file...')
        train_raw_data = Helper.extract_csv('train_data.csv')
        target_raw_data  = Helper.extract_csv('train_labels.csv')

        print('Splitting data into test/val/train')
        ## Perform a 70 - 15 - 15   Train - Validation - Test set of data for training and validating neural network
        #Get a 70% - 30% split of train_val data
        train_data, y_data, train_labels, y_labels = train_test_split(
                train_raw_data, target_raw_data, test_size=0.1, random_state=42)

        #Get 50% 50% split of remaining val data
        test_data, val_data , test_labels, val_labels = train_test_split(
                y_data, y_labels, test_size=0.5 , random_state=42)

        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

    else:
        print("Received data file '%s'" % data_file)
        #Get training data and targets
        test_data = Helper.extract_csv(data_file)
        test_data = np.asarray(test_data)
        test_labels = None
        print('Data loaded with shape: ', test_data.shape)

    # these are used for the +/- limits of the random distribution drawn from for initialization of weights and biases
    weight_init_range = 0.1
    bias_init_range = 0.3
    n_hidden = 5
    learning_rate = 1e-4

    print('Instantiating network with %i hidden neurons and learing rate of %.6f...' % (n_hidden, learning_rate))
    # Instantiate our network
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
            weight_init_range=weight_init_range,
            bias_init_range=bias_init_range
    )

    # Run inference on trained network, using validation set
    predictions = []

    if test_labels is None:
        for ii, test_input in enumerate(test_data):
            prediction, _ = net.feedforward(
                    layer_input=test_data[ii],
                    target=None
            )

            predictions.append(prediction)

    else:
        errors = []
        for ii, test_input in enumerate(test_data):
            prediction, error = net.feedforward(
                    layer_input=test_data[ii],
                    target=test_labels[ii]
            )

            predictions.append(prediction)
            errors.append(error)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(errors)
        plt.show()

    print('Returning predictions of shape ', np.asarray(predictions).shape)
    return predictions


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
if __name__ == '__main__':
    test_mlp('train_data.csv')

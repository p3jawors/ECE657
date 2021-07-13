# import required packages
import utils
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras



# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    verbose=True
    # 1. Load your saved model
    model = keras.models.load_model('models/RNN_model')

    # 2. Load your testing data
    test_data, test_labels = utils.load_data('data/test_data_RNN.csv', verbose=verbose)
    test_data = utils.dataset_2d_to_3d(test_data, verbose=verbose)

    # 3. Run prediction on the test data and output required plot and loss
    score = model.evaluate(test_data, test_labels, batch_size=1, verbose=True)

    scores = []
    # cycle through each test point to get loss at each point
    for ii in range(0, test_data.shape[0]):
        # reshape to maintain dimensionality
        dat = np.expand_dims(test_data[ii], 0)
        lab = np.expand_dims(np.asarray(test_labels[ii]), 0)
        scores.append(model.evaluate(dat, lab, verbose=verbose))
    plt.figure()
    plt.title('RNN Test Loss')
    plt.plot(scores)
    plt.show()

    #TODO need to extract predictions and plot them against GT
    # probably should just use the predit instead of the evaluate function

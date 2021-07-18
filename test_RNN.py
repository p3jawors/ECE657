# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

from tensorflow import keras



# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    verbose=True
    # 1. Load your saved model
    model = keras.models.load_model('models/RNN_model')

    # 2. Load your testing data
    test_data, test_labels = utils.load_RNN_data('data/test_data_RNN.csv', verbose=verbose)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    test_data = scaler.fit_transform(test_data)
    test_data = utils.dataset_2d_to_3d(test_data, verbose=verbose)
    test_data = utils.dataset_2d_to_3d(test_data, verbose=verbose)

    # 3. Run prediction on the test data and output required plot and loss
    score = model.evaluate(test_data, test_labels, batch_size=1, verbose=True)

    scores = []
    predictions = []
    # cycle through each test point to get loss at each point
    for ii in range(0, test_data.shape[0]):
        # reshape to maintain dimensionality
        dat = np.expand_dims(test_data[ii], 0)
        lab = np.expand_dims(np.asarray(test_labels[ii]), 0)
        # scores.append(model.evaluate(dat, lab, verbose=verbose))
        predictions.append(model.predict(dat, verbose=verbose))
        scores.append(predictions[ii]-lab)
    plt.figure()
    plt.subplot(211)
    plt.title('RNN Test Predictions')
    plt.plot(np.squeeze(predictions), label='predicitons')
    plt.plot(test_labels, label='ground truth')
    plt.legend()
    plt.subplot(212)
    plt.title('RNN Prediction Difference')
    plt.plot(np.squeeze(scores))
    plt.legend()
    plt.savefig('Q2_RNN-test.png')
    plt.show()

    #TODO need to extract predictions and plot them against GT
    # probably should just use the predit instead of the evaluate function

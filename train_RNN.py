# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
# TODO
def load_model():
    pass


if __name__ == "__main__":
        # uncomment to generate train/test split
        utils.generate_train_test_split()

	# 1. load your training data
        train_data, train_labels = utils.load_data('data/train_data_RNN.csv')

        # TODO normalize our data

        # load our model
        model = load_model()

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model

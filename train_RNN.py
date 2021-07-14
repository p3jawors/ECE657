# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM
from keras.optimizers import Adam
from tensorflow import keras


def load_model(model_name, verbose=True, **kwargs):
    #=================DEFINE MODEL STRUCTURE HERE====================
    if verbose:
        print(f"Loading model: {model_name}")

    if model_name == 'vanilla_lstm':
        model = Sequential()
        model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape'])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    if model_name == 'vanilla_batch_norm_lstm':
        model = Sequential()
        tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
        )
        model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape'])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    else:
        raise Exception(f"{model_name} is not a valid model")

    return model


if __name__ == "__main__":
    verbose = True
    # NOTE: uncomment to generate train/test split
    # utils.generate_RNN_train_test_split(verbose=verbose)

    # 1. load your training data
    train_data, train_labels = utils.load_RNN_data('data/train_data_RNN.csv', verbose=verbose)

    # Preprocess data
    # TODO look into what preprocessing we should be doing
    train_data = utils.dataset_2d_to_3d(train_data, verbose=verbose)

    # load our model
    # define the model name and the arguments that go along with it here
    # model_name = 'vanilla_lstm'
    # model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 50}

    model_name = 'vanilla_batch_norm_lstm'
    model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 100}

    model = load_model(model_name, verbose=verbose, **model_args)

    # 2. Train your network
    n_epochs = 10
    batch_size = 8
    validation_split = 0.2
    if verbose:
        print('training data shape: ', train_data.shape)

    history = model.fit(
            train_data,
            train_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=True)

    utils.plot_training_results(
            histories=[history],
            cols=['r'],
            labels=[model_name],
            title='RNN Model Training')

    # 		Make sure to print your training loss within training to show progress
    # 		Make sure you print the final training loss

    # 3. Save your model
    model.save('models/RNN_model')

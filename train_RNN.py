# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam
import keras_lmu
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

    elif model_name == 'vanilla_batch_norm_lstm':
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

    elif model_name == 'layered_batch_norm_lstm':
        model = Sequential()
        # tf.keras.layers.BatchNormalization(
        #     axis=-1,
        #     momentum=0.99,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer="zeros",
        #     gamma_initializer="ones",
        #     moving_mean_initializer="zeros",
        #     moving_variance_initializer="ones",
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        # )
        for ii in range(kwargs['n_layers']):
            model.add(BatchNormalization())
            if ii == kwargs['n_layers']-1:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape'])))
            else:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape']), return_sequences=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    elif model_name == 'layered_dropout_lstm':
        model = Sequential()
        for ii in range(kwargs['n_layers']):
            if ii == kwargs['n_layers']-1:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape'])))
            else:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape']), return_sequences=True))
            model.add(Dropout(kwargs['dropout_rate']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    elif model_name == 'layered_dropout_batchnorm_lstm':
        model = Sequential()
        for ii in range(kwargs['n_layers']):
            model.add(BatchNormalization())
            if ii == kwargs['n_layers']-1:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape'])))
            else:
                model.add(LSTM(kwargs['n_neurons'], activation='relu', input_shape=(kwargs['input_shape']), return_sequences=True))
            model.add(Dropout(kwargs['dropout_rate']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')



    elif model_name == 'vanilla_lmu':
        model = Sequential()
        lmu_layer = keras_lmu.LMU(
            memory_d=kwargs['memory_d'],
            order=kwargs['order'],
            theta=kwargs['theta'],
            hidden_cell=kwargs['hidden_cell'],
            hidden_to_memory=kwargs['hidden_to_memory'],
            memory_to_memory=kwargs['memory_to_memory'],
            input_to_hidden=kwargs['input_to_hidden'],
            dropout=kwargs['dropout']
        )
        model.add(lmu_layer)
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    elif model_name == 'layered_lmu':
        model = Sequential()
        for ii in range(kwargs['n_layers']):
            lmu_layer = keras_lmu.LMU(
                memory_d=kwargs['memory_d'],
                order=kwargs['order'],
                theta=kwargs['theta'],
                hidden_cell=kwargs['hidden_cell'],
                # hidden_cell=tf.keras.layers.SimpleRNNCell(units=500),
                hidden_to_memory=kwargs['hidden_to_memory'],
                memory_to_memory=kwargs['memory_to_memory'],
                input_to_hidden=kwargs['input_to_hidden'],
                dropout=kwargs['dropout']
            )
            model.add(lmu_layer)
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
    train_data = train_data.astype('float32')
    train_labels = train_labels.astype('float32')

    # Preprocess data
    # TODO look into what preprocessing we should be doing
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    # #TODO is this scaling as expected? not suire since we have to flatten our features first
    train_data = scaler.fit_transform(train_data)
    train_data = utils.dataset_2d_to_3d(train_data, verbose=verbose)
    print('SCALED DATA: ', train_data)
    # raise Exception

    # load our model
    # define the model name and the arguments that go along with it here
    # model_name = 'vanilla_lstm'
    # model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 50}

    # model_name = 'vanilla_batch_norm_lstm'
    # model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 500}

    # model_name = 'layered_batch_norm_lstm'
    # model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 500, 'n_layers': 2}

    # model_name = 'layered_dropout_lstm'
    # model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 50, 'n_layers': 4, 'dropout_rate': 0.2}

    model_name = 'layered_dropout_batchnorm_lstm'
    model_args = {'input_shape': train_data.shape[1:], 'n_neurons': 50, 'n_layers': 4, 'dropout_rate': 0.2}

    # model_name = 'vanilla_lmu'
    # model_args = {
    #         'memory_d': 16,
    #         'order': 256,
    #         'theta': np.prod(train_data.shape),
    #         # 'hidden_cell': tf.keras.layers.SimpleRNNCell(units=10),
    #         'hidden_cell': tf.keras.layers.SimpleRNNCell(units=50),
    #         'hidden_to_memory': True,
    #         'memory_to_memory': True,
    #         'input_to_hidden': False,
    #         'dropout': 0.2
    #         }


    # model_name = 'layered_lmu'
    # model_args = {
    #         'memory_d': 4,
    #         'order': 256,
    #         'theta': np.prod(train_data.shape),
    #         # 'hidden_cell': tf.keras.layers.SimpleRNNCell(units=10),
    #         'hidden_cell': tf.keras.layers.SimpleRNNCell(units=500),
    #         'hidden_to_memory': True,
    #         'memory_to_memory': False,
    #         'input_to_hidden': False,
    #         'dropout': 0.2,
    #         'n_layers': 2
    #         }

    model = load_model(model_name, verbose=verbose, **model_args)

    # 2. Train your network
    n_epochs = 1000
    batch_size = 32
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
            title='RNN Model Training: %s' % model_name)

    # 		Make sure to print your training loss within training to show progress
    # 		Make sure you print the final training loss

    # 3. Save your model
    model.save('models/RNN_model')

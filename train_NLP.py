# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
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
    # utils.generate_NLP_train_test_split(verbose=verbose)


    try:
        print("attempting to find previously trained preprocessing")
        train_data = pd.read_csv(os.path.join(os.getcwd(), 'data/NLP_Preproc.csv'))
        print("DATA FOUND YEEEE BOII")
        print(train_data)
    except IOError:
        print("No preproced data, raw load and preproc commencing")
        # 1. load your training data - stupid folder structure with this dataset
        raw_train_data = utils.load_NLP_data('data/aclImdb/train/', verbose=False)

        # Preprocess data - I gotchu boo
        train_data = utils.preprocess_NLP_data(raw_train_data, verbose=False)

        #Save preprocessed data to save time between runs/tuning (~2 min per run)
        train_data.to_csv(os.path.join(os.getcwd(), 'data/NLP_Preproc.csv'))
    # 2. Train your network

   # history = model.fit(
   #         train_data,
   #         train_labels,
   #         epochs=n_epochs,
   #         batch_size=batch_size,
   #         validation_split=validation_split,
   #         verbose=True)

   # utils.plot_training_results(
   #         histories=[history],
   #         cols=['r'],
   #         labels=[model_name],
   #         title='RNN Model Training')

   # 		Make sure to print your training loss within training to show progress
   # 		Make sure you print the final training loss

   # 3. Save your model
   # model.save('models/nlp_embedding_model')

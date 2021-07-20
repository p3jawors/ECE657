# import required packages
import utils
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import gzip

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import Doc2Vec


from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences



# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    verbose=True
    # algorithm = "sg"
    algorithm = "cbow"

    # 1. Load your saved model
    # try:
    # sentiment_model.read_pickle(os.path.join(os.getcwd(), 'models/NLP_sentiment_classifier_'+algorithm+'.pickle'), compression='gzip')
    # sentiment_model = pickle.load(open(
    #     os.path.join(os.getcwd(), 'models/NLP_sentiment_classifier_'+algorithm+'.pickle'), 'rb'
    #     )
    # )

    sentiment_model = keras.models.load_model('models/NLP_model')

        # # 2. Load your testing data
        # try:
    print("attempting to find previously preproceed test set")
    test_data = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))
    print("Previous test data found")
    print(test_data)
        # except IOError:
        #     raw_test_data = utils.load_NLP_test_data('data/aclImdb/test/', verbose=False)
        #
        #     # Preprocess data - I gotchu boo
        #     test_data = utils.preprocess_NLP_data(raw_test_data, verbose=False)
        #
        #     #Save preprocessed data to save time between runs/tuning (~2 min per run)
        #     test_data.to_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))
        #     del raw_test_data

       # 3 Genterate proper embedded dataset using the new model prior to output training
        #  This allows us to reuse previous iterations
        # try:
    print("Attempting to find previous trained "+ algorithm +" model")
    model = KeyedVectors.load(os.path.join(os.getcwd(), 'models/' + algorithm + '_model.blob')) #uncomment for production


    # print("IS THIS WORKING? ", os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle'))
    g_data = gzip.open(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'embedd.pickle.gz'))
    test_embedded_df = pd.read_pickle(g_data)
    g_data.close()
    # test_embedded_df = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle'))
    print("Test Data")
    print(test_embedded_df)

    del test_data

        # except IOError:
        #     print("Generating vectorized training from dataframe")
        #     print(test_data)
        #     test_embedded_df = utils.embedd_dataset(test_data, model)
        #
        #     if test_embedded_df is not None:
        #         test_embedded_df.to_pickle(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle'), compression='gzip')
        #         print("Reesult stored:" + str(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle')))
        #         del test_data




    # pad test data with zeros
    print(test_embedded_df)
    print(type(test_embedded_df))
    x_test = test_embedded_df['vector_sentence']
    # print('shape1: ', x_test.shape)
    x_test = pad_sequences(x_test, padding='post')
    # print('shape2: ', x_test.shape)
    x_test.resize(x_test.shape[0], 1383, x_test.shape[2])
    x_test = np.asarray(x_test).astype('float32')
    print('shape3: ', x_test.shape)
    # x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
    # print('shape4: ', x_test.shape)
    y_test = test_embedded_df['sentiment']
    y_test = y_test.to_numpy()
    new_y = []
    for y in y_test:
        new_y.append(y)
    y_test = np.asarray(new_y)
    y_test = np.asarray(y_test).astype('float32')
    # 3. Run prediction on the test data and print the test accuracy

    # scores = sentiment_model.score(x_test, y_test)
    predictions = sentiment_model.predict(x_test)
    # scores = []
    # predictions = []
    # # cycle through each test point to get loss at each point
    # for ii in range(0, x_test.shape[0]):
    #     predictions.append(sentiment_model.predict(x_test[ii]))
    #     scores.append(sentiment_model.score(x_test[ii], y_test[ii]))
    plt.figure()
    # plt.subplot(211)
    plt.title('NLP Test Predictions')
    # plt.scatter(np.arange(0, len(y_test)), y_test[0], label='ground truth')
    # plt.scatter(np.arange(0, len(y_test)), y_test[1], label='ground truth')
    plt.plot(y_test)
    plt.plot(np.squeeze(predictions), label='predicitons')
    plt.legend()
    # plt.subplot(212)
    # plt.title('NLP Prediction Difference')
    # plt.plot(np.squeeze(scores))
    # plt.legend()
    plt.show()

    # except IOError:
    #     print("Loading model failed for testing")
    #     print("run train_NLP.py to generate, data/models")
    #

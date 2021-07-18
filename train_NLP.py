# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import gzip

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import Doc2Vec

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM
from keras.optimizers import Adam
from tensorflow import keras



"""
    Takes a vectorized dataframe and trains a classifier to determine sentiment.
    return a classifier thats trained to store for sentiment classification
"""
def train_classifier(dataframe, random_seed):

    x_train = dataframe['vector_sentence'].to_list()
    y_train = dataframe['sentiment'].to_list()
    logRClassifier = SGDClassifier(loss='log', penalty='l1')
    logRClassifier.fit(x_train, y_train)

    return logRClassifier



if __name__ == "__main__":
    verbose = True
    #Best random seed to make things more consistent :P
    random_seed = 1337

    # NOTE: uncomment to generate train/test split
    # utils.generate_NLP_train_test_split(verbose=verbose)

    #Set this to either CBOW/SG for continous bag of words or Skip-o-gram for generating model
    # as well as our test and train sets before going training
    algorithm = "cbow"

    try:
        print("Attempting to find previously trained preprocessing")
        #train_data = pd.read_pickle(os.path.join(os.getcwd(), 'Does not exist')) #Use for testing

        train_data = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_train_Preproc.pickle'))
        #train_data.LABELS = train_data.LABELS.apply(ast.literal_eval)
        print("DATA FOUND YEEEE BOII")
        print(train_data)

        test_data = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))
        #test_data.LABELS = test_data.LABELS.apply(ast.literal_eval)
        print("Boyakasha! Test Data too bro!")
        print(test_data)

    except IOError:
        print("No preproced data, raw load and preproc commencing")
        # 1. load your training data - stupid folder structure with this dataset
             #you need to load and embedd all this data train/test/usup to make proper embeddings
             # before we make a network to train on sentiment
        raw_train_data, raw_test_data = utils.load_all_NLP_data('data/aclImdb/', verbose=False)

        # Preprocess data - I gotchu boo
        train_data, test_data = utils.preprocess_NLP_data(raw_train_data, raw_test_data, verbose=False)

        #Save preprocessed data to save time between runs/tuning (~2 min per run)
        train_data.to_pickle(os.path.join(os.getcwd(), 'data/NLP_train_Preproc.pickle'))
        test_data.to_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))

    # 2. Train your emedding using word2vec
    # Continous bag of words model
    try:
        print("Attempting to find previous trained "+ algorithm +" model")
        model = KeyedVectors.load(os.path.join(os.getcwd(), 'models/' + algorithm + '_model.blob')) #uncomment for production

        #model = Doc2Vec.load(os.path.join(os.getcwd(), 'always fail'))
        print("Bag of Words found!")
        #model = model.wv
    except IOError:
        #Using params from tutorial will fine tune with grid search after
        min_count = 7
        learning_rate = 0.03
        min_learning_rate = 0.0007
        window = 2
        vec_size = 0 #Setting this to zero/negative, train_NLP_vectors will use average word counts for vector size
        sample = 6e-5
        neg_samples = 20

        model = utils.train_NLP_vectors(train_data, test_data, vec_size, window, learning_rate, min_learning_rate, min_count, neg_samples, algorithm,
                                              random_seed, verbose)

        print("No existing model found: Generating "+algorithm+ " embedding model")
        model.save(os.path.join(os.getcwd(), 'models/'+ algorithm +'_model.blob'))

    #Do some visualizations if we want
    print("top 10 words")
    utils.visualize_embeddings(model)

    #3 Genterate proper embedded dataset using the new model prior to output training
    #  This allows us to reuse previous iterations
    try:
        print("Looking for previously vectorized training data")
        g_data = gzip.open(os.path.join(os.getcwd(), 'data/NLP_train_'+algorithm+'embedd.pickle.gz'))
        train_embedded_df = pd.read_pickle(g_data)
        g_data.close()
        print("Train Data. FOUND YEEEE BOII")
        print(train_embedded_df)

    except IOError:
        print("Load failed")
        print("Generating vectorized training from dataframe")
        print(train_data)
        train_embedded_df = utils.embedd_dataset(train_data, model)

        if train_embedded_df is not None:
            train_embedded_df.to_pickle(os.path.join(os.getcwd(), 'data/NLP_train_'+algorithm+'embedd.pickle.gz'), compression='gzip')
            print("Reesult stored:" + str(os.path.join(os.getcwd(), 'data/NLP_train_'+algorithm+'embedd.pickle')))


    #5 Train the output classifier
    # Use a set of featature vectors applied to the original training set as well as the sentiment, and potentially other features
    # (rating/Word count) to determine the potential output of the resulting sentiment of the review
    sentiment_model = train_classifier(train_embedded_df, random_seed)
    sentiment_model.to_pickle(os.path.join(os.getcwd(), 'models/NLP_sentiment_classifier_'+algorithm+'.pickle'), compression='gzip')


    #6 Visualize result of the training/performance of the final network



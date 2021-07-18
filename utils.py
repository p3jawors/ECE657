# import required packages
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import time

from ast import literal_eval
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import multiprocessing
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

import nltk

import string


def create_dirs(folders):
    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Creating missing directory {folder}")
            os.makedirs(folder)

def plot_training_results(histories, cols, labels, title):
    plt.figure()
    plt.title(title)
    print('len histories: ', len(histories))
    for ii, hist in enumerate(histories):
        # plot loss
        plt.subplot(211)
        plt.title('Loss')
        if 'loss' in hist.history:
            plt.plot(hist.history['loss'], color=cols[ii], label='train %s' % labels[ii])
        if 'val_loss' in hist.history:
            plt.plot(hist.history['val_loss'], color=cols[ii], label='val %s' % labels[ii], linestyle='--')
        plt.legend()
        # plot accuracy
        plt.subplot(212)
        plt.title('Accuracy')
        if 'accuracy' in hist.history:
            plt.plot(hist.history['accuracy'], color=cols[ii], label='train %s' % labels[ii])
        if 'val_accuracy' in hist.history:
            plt.plot(hist.history['val_accuracy'], color=cols[ii], label='val %s' % labels[ii], linestyle='--')
        plt.legend()
    plt.savefig('Q1_%s.png' % title)
    plt.show()


# saved dataset to data folder
def generate_RNN_train_test_split(verbose=True):
    if verbose:
        print('--Generating train/test split--')
    # read the csv file
    with open('data/q2_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        xi = []
        y = []
        for ii, row in enumerate(reader):
            # columns are [Date, Close/Last, Volume, Open, High, Low]
            # we want to predict day 4 close using days 1-3 Vol, Op, Hi, and Low
            if ii == 0:
                # skip the header row
                continue
            if ii%4 == 0:
                # 4th day in this set, save close as target
                # cast to float
                y.append(np.copy(float(row[3])))
                # append our 3 day list to our feature list
                x.append(np.copy(xi))
                # reset our 3 day list
                xi = []
            else:
                # append features for days 1-3
                # cast to float
                xi.append([float(i) for i in row[2:]])
    if verbose:
        print('raw x: ', np.asarray(x).shape)
        print('raw y: ', np.asarray(y).shape)
    x = np.array(x)
    #reshape our input vectors from 2D to 1D
    x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    if verbose:
        print('reshape x: ', x.shape)

    # get our test / train split and randomize
    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.3, random_state=0)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    test_data = np.array(test_data)
    if verbose:
        print('train data: ', train_data.shape)
        print('train labels: ', train_labels.shape)
        print('test data: ', test_data.shape)
        print('test labels: ', test_labels.shape)

    # stack our data and labels to save to csv
    # NOTE that the GT is saved to column 0
    train = np.hstack((train_labels[:, None], train_data))
    test = np.hstack((test_labels[:, None], test_data))
    if verbose:
        print('stacked train: ', train.shape)
        print('stacked test: ', test.shape)
    # save to csv
    np.savetxt("data/train_data_RNN.csv", train, delimiter=",")
    np.savetxt("data/test_data_RNN.csv", test, delimiter=",")

# load the dataset we generated above
def load_RNN_data(filename, verbose=True):
    if verbose:
        print(f"--Loading dataset from {filename}--")
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        for ii, row in enumerate(reader):
            x.append([float(val) for val in row[1:]])
            y.append(float(row[0]))
        x = np.asarray(x)
        y = np.asarray(y)
        if verbose:
            print('loaded data: ', x.shape)
            print('loaded labels: ', y.shape)
    return (x, y)

# def normalize_data(data, verbose=True):
#     if verbose:
#         print('--Normalizing Data--')
#         print('data shape: ', data.shape)

def mean_sub_var_div(dataset, binary, verbose=True):
    # performs a mean subtraction and scales by the variance
    # this shift the majority of data to -1 to 1
    # if binary is false we scale once more to go to 0 to 1
    # TODO
    return 0


def dataset_2d_to_3d(dataset, verbose=True):
    # our dataset has 3 days of 4 features, reshape
    # for 3d layers to be in shape (n_samples, n_timesteps_per_sample, n_features)
    reshaped = np.reshape(dataset, (-1, 3, 4))
    if verbose:
        print(f"Reshaping dataset to be in shape of (n_sample, n_timesteps_per_sample, n_features): {reshaped.shape}")

    return reshaped



def preprocess_NLP_data(train_dataframe,
                        test_dataframe=None,
                    options=['lowercase',
                             'breaks',
                             'punctuation',
                             'tokenize',
                             'stopwords',
                             'stem'],
                    stemmer='porter',
                    language='english',
                    verbose = True):

  nltk.download('punkt')
  nltk.download('stopwords')

  if test_dataframe is not None:
    print("Preprocessing both train and test sets")
    data_to_proc = [train_dataframe, test_dataframe]

    row_range = len(train_dataframe)
    print("Num Items to preproc:" +str(row_range))
  else:
    print("Preprocessing train set")
    data_to_proc = [train_dataframe]
    row_range = len(train_dataframe)



  if verbose is True:
    print(preprocessed_df)

   #Try different methods for stemming our words as part of prepcoessing
  if stemmer == 'porter':
    stemmer_obj = PorterStemmer()
  elif stemmer == 'snowball':
    stemmer_obj = SnowballStemmer()
  elif stemmer == 'lancaster':
    stemmer_obj = LancasterStemmer()

  for preprocessed_df in data_to_proc:

      print("Ingesting frame")
      print(preprocessed_df)
      with tqdm(total= row_range, disable = verbose) as pbar:
        for i in preprocessed_df.itertuples():
            pbar.update(1)
            index = i.Index
            column = 2
            col = 'review'

            if 'lowercase' in options:
              preprocessed_df.at[index, col] = i[column].lower()
              if verbose is True:
                print("\nlowercase: " + preprocessed_df.at[index, col]+ "\n")

             #Remove breaks,
            if 'breaks' in options:
              soup = BeautifulSoup(preprocessed_df.at[index, col], features='lxml')
              for breaks in soup.findAll("<br>"):
                    breaks.extract()
              for breaks in soup.findAll('</br>'):
                    breaks.extract()
              preprocessed_df.at[index, col] = soup.get_text()

              if verbose is True:
                print("\nbreaks: " + preprocessed_df.at[index, col]+ "\n")


            if 'punctuation' in options:
              preprocessed_df.at[index, col] = \
              "".join([char for char in preprocessed_df.at[index, col]
                       .encode('ascii','ignore')
                       .decode() if char not in string.punctuation])
              if verbose is True:
                print("punctuation: " + preprocessed_df.at[index, col] + "\n")


            if 'tokenize' in options:
              preprocessed_df.at[index, col] = \
              nltk.word_tokenize(preprocessed_df.at[index, col])

              preprocessed_df.at[index, 'raw_word_count'] = len(preprocessed_df.at[index, col])
              if verbose is True:
                print("tokenized: " + str(preprocessed_df.at[index, col]) + "\n" )


            if 'stopwords' in options:
              stop_words = stopwords.words(language)
              preprocessed_df.at[index, col] = \
              [word for word in preprocessed_df.at[index, col] if word not in stop_words]

              #track how many words we removed
              preprocessed_df.at[index, 'pp_word_count'] = len(preprocessed_df.at[index, col])
              if verbose is True:
                print("stopwords: " + str(preprocessed_df.at[index, col]) + "\n")


            if 'stem' in options:
              preprocessed_df.at[index, col] = \
              [stemmer_obj.stem(word) for word in preprocessed_df.at[index, col]]
              if verbose is True:
                print("stemming: " + str(preprocessed_df.at[index, col]) + "\n")


      if verbose is True:
        print(preprocessed_df)

      pbar.close()

  if test_dataframe is not None:
      print("Train Dataframe")
      print(train_dataframe)
      print("Test Dataframe")
      print(train_dataframe)
      return train_dataframe, test_dataframe
  else:
      print(train_dataframe)
      return train_dataframe


#File data is distributed into folders in a gross way.

# data/adlImdb/train/pos   - positive data strings
# data/adlImdb/train/neg   - negative data strings
# data/adlImdb/train/unsup       - untaged datastrings usef for unsupervised learning

# Each filename contains the ID of the movie and the rating as follows
# ID_stars.txt

"""
   Given the input path, run through the bonkos folder structure used
   spit out loaded train/test before processing
"""

def load_NLP_data(path_to_data, verbose=True):

    train_data_raw_dict = {'id':[], 'review':[], 'rating':[], 'raw_word_count':[], 'pp_word_count':[], 'word_vectors':[] ,'sentiment':[]}
    test_data_raw_dict = {'id':[], 'review':[], 'rating':[], 'raw_word_count':[], 'pp_word_count':[], 'word_vectors':[] ,'sentiment':[]}

    path_to_train_folder =  "train/"
    path_to_test_folder =  "test/"

    #Load in the positive data values first
    path_to_pos =  'pos'
    path_to_neg =  'neg'

    print("Reading data from folders")

    #Sentiment encoded as positive = 1 , negative = 0
    #Done so its already encoded into some binary pattern lazily.
    #First iteration should go throu with positive values, and

    folder_paths = [[path_to_data+path_to_train_folder, train_data_raw_dict] , [path_to_data+path_to_test_folder, test_data_raw_dict]]

    #Churn through all our datasamples to make this blob for embedding
    with tqdm(total= 50000, disable = verbose) as pbar_top:

        list_of_paths = [[path_to_pos, 1], [path_to_neg, 0]]
        for folder in folder_paths:

            if verbose is True:
               print("Target Folder: "+ folder[0])

            with tqdm(total= 25000, disable=verbose) as pbar:
              for target_path in list_of_paths:
                  abs_path = os.path.join(os.getcwd(), folder[0]+target_path[0])

                  if verbose is True:
                     print("path: "+ target_path[0])

                  for filename in os.listdir(abs_path):
                      split_filename = re.split('_|\.txt', filename)

                      film_id = split_filename[0]
                      rating = split_filename[1]

                      folder[1]['id'].append(film_id)
                      folder[1]['rating'].append(rating)
                      #encode sentiment based on data using
                      folder[1]['sentiment'].append(target_path[1])

                      #pre/post processing statistics placeholders
                      folder[1]['raw_word_count'].append(0)
                      folder[1]['pp_word_count'].append(0)
                      folder[1]['word_vectors'].append(0)

                      with open(os.path.join(os.getcwd() + "/" + folder[0] + target_path[0] + "/", filename), 'r') as f:
                          text_string = f.readline()
                          folder[1]['review'].append(text_string)
                          if verbose is True:
                              print("Extracted " + folder[0] + target_path[0] + filename)
                          pbar.update(1)
                          pbar_top.update(1)

            dataframe = pd.DataFrame(data=folder[1])
            folder.append(dataframe)
            tqdm.pandas(desc="Dataframe")
            if verbose is True:
                print("Data Frame generated")
                print(folder[2])

    pbar_top.close()
    pbar.close()
    return folder_paths[0][2], folder_paths[1][2]



"""
  Ingest the preprocessed dataframe into word2vec and perform CBOW and Skip-Agram to generate
  two embedding models, then use that for training our final classifier/network

  Good resoource for this: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

  Direct doc since alot of tutorials are outdate. size->vector_size. RTFM I suppose
  https://radimrehurek.com/gensim/models/word2vec.html


  More useful tutorials:
  https://www.districtdatalabs.com/modern-methods-for-sentiment-analysis

  Using classifiers as the final decision maker from word embeddings
  https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca

  Extensive and broken down - deals with different symbols since twitter has links and 140 word limit
  https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-11-cnn-word2vec-41f5e28eda74

  Using Keras as the backend, but interesting frequentist approach. using LSTM + a few more output layers
  https://medium.datadriveninvestor.com/sentiment-analysis-using-embeddings-f3dd99aeaade

  Another example with the imdb dataset and where to go once we've created the embedding models
  https://thedatafrog.com/en/articles/word-embedding-sentiment-analysis/

"""
def train_NLP_vectors(dataframe,
                        test_dataframe,
                        feature_size,
                        window,
                        learning_rate,
                        min_learn_rate,
                        min_count,
                        negative_sample_rate,
                        algorithm,
                        rnd_seed,
                        verbose):
  #cbow - continous bag of words
  #sg - skip -o -gram
  dict_algo = {"sg":1, "cbow":0}


  if test_dataframe is not None:
    embedding_dataframe = dataframe.copy()
    test_frame = test_dataframe.copy()
    result = pd.concat([embedding_dataframe, test_frame], ignore_index =True)

    if verbose is True:
        print("Taking Training Data Frame \n")
        print(dataframe)
        print("Taking Test Data Frame \n")
        print(test_frame)
        print("Concat to the following: Result \n")
        print(result)
  else:
    result = dataframe.copy()


  if verbose is True:
      print("\nAlgo used: " + algorithm)
      print("Params: VecSize: "+ str(feature_size) + " Window:" + str(window)+ " Learn Rate:"+ str(learning_rate) + " Min learn Rate:"+ str(min_learn_rate))
      print("        Min_count:"+ str(min_count) +  " Neg Sample Rate: "+ str(negative_sample_rate) + "\n")


  #We need all the help we can get
  cores = multiprocessing.cpu_count()
  preproc_data = result['review'].values.tolist()

  if verbose is True:
     print("This is a sanity check")
     print(preproc_data[:2])

  #Initialize model
  start_time = time.time()

  w2v_model = Word2Vec(sentences = preproc_data,
                    min_count=min_count,
                   vector_size=feature_size,
                   window=window,
                   alpha=learning_rate,
                   min_alpha=min_learn_rate,
                   negative=negative_sample_rate,
                   sg= dict_algo[algorithm],
                   workers=cores-1)

  model = w2v_model.wv
  #build vocab of the model
  if verbose is True:
    print("Time taken to train model: " + str(time.time() - start_time))
    print("Total words in model: "+ str(len(model.key_to_index)))
  return model


"""
    Take in a model and then a dataframe, convert sentences in dataframe
    to a vectorized set of data that can be used for training with sentiment

    dataframe - pandas dataframe
    model     - model (represented by gensims KeyedVectors)
    verbise   - Turn on verbose commands

    return a dataframe with a vectorized sentence and its overall sentiment
"""
def embedd_dataset(dataframe, model, verbose=True):

  start_time = time.time()
  # Embed training dataset words.
  #embedded_dict = {'word':[], 'embedding vector':[]}
  #for i in dataframe.itertuples():
  #  column = 3
  #  col = 'review'
  #  index = i.Index
  #  print(i[column])
#
 #   for word in :
 #     if model.key_to_index[word] is not None:
 #       if word not in train_embedded_dict['word']:
 #         train_embedded_dict['word'].append(word)
 #         train_embedded_dict['embedding vector'].append(model.get_vector(word, norm=True))


  #embedded_df = pd.DataFrame(data=train_embedded_dict)

  print("Time taken to apply embedding to dataset: " + str(time.time() - start_time))

  #if verbose is True:
  #  print("Time taken to perform embedd vectors: " + str(time.time() - start_time))
  #  print(embedded_df)

"""
 Look at stuff close to the given words
 Also visualize stuff
 https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
"""
def visualize_embeddings(model):
    print("\nTop 10 related to bad")
    print(model.most_similar("bad"))

    print("\nTop 10 related to great\n")
    print(model.most_similar("great"))

    print("\nTop 10 related to ok\n")
    print(model.most_similar("ok"))


#

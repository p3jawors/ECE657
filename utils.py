# import required packages
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import time

from tqdm.auto import tqdm

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
nltk.download('punkt')
nltk.download('stopwords')

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



def preprocess_NLP_data(dataframe,
                    options=['lowercase',
                             'punctuation',
                             'tokenize',
                             'stopwords',
                             'stem'],
                    stemmer='porter',
                    language='english',
                    verbose = True):

  preprocessed_df = dataframe

  row_range = len(preprocessed_df['review'])


  print("Preprocessing Dataframe")
  if verbose is True:
    print(preprocessed_df)

   #Try different methods for stemming our words as part of prepcoessing
  if stemmer == 'porter':
    stemmer_obj = PorterStemmer()
  elif stemmer == 'snowball':
    stemmer_obj = SnowballStemmer()
  elif stemmer == 'lancaster':
    stemmer_obj = LancasterStemmer()

  row_count = 0
  with tqdm(total= len(preprocessed_df['review'])) as pbar:
    for i in preprocessed_df.itertuples():
        pbar.update(1)
        index = i.Index
        column = 2
        col = 'review'

        if 'lowercase' in options:
          preprocessed_df.at[index, col] = i[column].lower()
          if verbose is True:
            print("\nlowercase: " + preprocessed_df.at[index, col]+ "\n")


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
  return preprocessed_df

#File data is distributed into folders in a gross way.

# data/adlImdb/train/pos   - positive data strings
# data/adlImdb/train/neg   - negative data strings
# data/adlImdb/train/unsup       - untaged datastrings usef for unsupervised learning

# Each filename contains the ID of the movie and the rating as follows
# ID_stars.txt

"""
   Given the input path, run through the bonkos folder structure used
"""

def load_NLP_data(path_to_data, verbose=True):

    train_data_raw_dict = {'id':[], 'review':[], 'rating':[], 'raw_word_count':[], 'pp_word_count':[], 'WordVectors':[] ,'sentiment':[]}
    #Load in the positive data values first
    path_to_pos =  path_to_data + 'pos'
    path_to_neg =  path_to_data + 'neg'

    #Sentiment encoded as positive = 1 , negative = 0
    #Done so its already encoded into some binary pattern lazily.
    #First iteration should go throu with positive values, and
    list_of_paths = [ [path_to_pos, 1], [path_to_neg, 0]]

    print("Reading data from folders")
    with tqdm(total= 25000 ) as pbar:
      for target_path in list_of_paths:
          abs_path = os.path.join(os.getcwd(), target_path[0])
          for filename in os.listdir(abs_path):

              split_filename = re.split('_|\.txt', filename)

              film_id = split_filename[0]
              rating = split_filename[1]

              train_data_raw_dict['id'].append(film_id)
              train_data_raw_dict['rating'].append(rating)
              #encode sentiment based on data using
              train_data_raw_dict['sentiment'].append(target_path[1])

              #pre/post processing statistics placeholders
              train_data_raw_dict['raw_word_count'].append(0)
              train_data_raw_dict['pp_word_count'].append(0)

              with open(os.path.join(os.getcwd() + "/" + target_path[0] + "/", filename), 'r') as f:
                  text_string = f.readline()
                  train_data_raw_dict['review'].append(text_string)
                  if verbose is True:
                      print("Extracted " + target_path[0] + filename)
                  pbar.update(1)

    train_data_raw_df = pd.DataFrame(data=train_data_raw_dict, )
    tqdm.pandas(desc="Dataframe")

    print("Data Frame generated")
    print(train_data_raw_df)

    pbar.close()
    return train_data_raw_df


"""
  Ingest the preprocessed dataframe into word2vec and perform CBOW and Skip-Agram to generate
  two embedding models, then use that for training our final classifier

  Good resoource for this: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

  Direct doc since alot of tutorials are outdate. size->vector_size. RTFM I suppose
  https://radimrehurek.com/gensim/models/word2vec.html

"""
def train_NLP_embedding(dataframe,
                        feature_size,
                        window,
                        learning_rate,
                        min_learn_rate,
                        min_count,
                        negative_sample_rate,
                        algorithm,
                        rnd_seed,
                        verbose):

  dict_algo = {"Skip-o-gram":1, "CBOW":0}

  if verbose is True:
      print("\nAlgo used: " + algorithm)
      print("Params: VecSize: "+ str(feature_size) + " Window:" + str(window)+ " Learn Rate:"+ str(learning_rate) + " Min learn Rate:"+ str(min_learn_rate))
      print("        Min_count:"+ str(min_count) +  " Neg Sample Rate: "+ str(negative_sample_rate) + "\n")

  cores = multiprocessing.cpu_count()

  #Initialize model
  start_time = time.time()
  model = Word2Vec(sentences = dataframe['review'],
                    min_count=min_count,
                   vector_size=feature_size,
                   window=window,
                   alpha=learning_rate,
                   min_alpha=min_learn_rate,
                   negative=negative_sample_rate,
                   sg= dict_algo[algorithm],
                   workers=cores-1)

  #build vocab of the model
  print("Time taken to train model: " + str(time.time() - start_time))

  #with tqdm(total= 25000 ) as pbar:
  #  row_count = 0
  #  for row in dataframe['review#']:
      #if verbose is True:
      #  print("Training" + str(row)+ "\n" )

#      row_count += 1
#      pbar.update(1)
#      model.build_vocab(row)
#      #Testing
 #     if row_count >= 3:
 #           break

 # pbar.close()
  return model



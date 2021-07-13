# import required packages
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def create_dirs(folders):
    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Creating missing directory {folder}")
            os.makedirs(folder)

# saved dataset to data folder
def generate_train_test_split(verbose=True):
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
def load_data(filename, verbose=True):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        for ii, row in enumerate(reader):
            x.append(row[1:])
            y.append(row[0])
        x = np.asarray(x)
        y = np.asarray(y)
        if verbose:
            print('loaded data: ', x.shape)
            print('loaded labels: ', y.shape)
    return (x, y)



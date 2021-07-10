# import required packages
import csv
import numpy as np
from sklearn.model_selection import train_test_split

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
            # append our target and reset our feature list
            # cast to float
            y.append(np.copy(float(row[3])))
            x.append(np.copy(xi))
            xi = []
        else:
            # append 3 days of features
            # cast to float
            xi.append([float(i) for i in row[2:]])
print('x: ', np.asarray(x).shape)
print('y: ', np.asarray(y).shape)
x = np.array(x)
#reshape our input vectors from 2D to 1D
x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
print('reshape x: ', x.shape)

train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.3, random_state=0)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
test_data = np.array(test_data)
print('train data: ', train_data.shape)
print('train labels: ', train_labels.shape)
print('test data: ', test_data.shape)
print('test labels: ', test_labels.shape)
# stack our data and labels
train = np.hstack((train_labels[:, None], train_data))
test = np.hstack((test_labels[:, None], test_data))
print('stacked train: ', train.shape)
print('stacked test: ', test.shape)
np.savetxt("data/train_data_RNN.csv", train, delimiter=",")
np.savetxt("data/test_data_RNN.csv", test, delimiter=",")
# saved dataset to data folder

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


# if __name__ == "__main__": 
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model

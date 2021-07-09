# import required packages
import csv
import numpy as np

with open('data/q2_dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    x = []
    xi = []
    y = []
    for ii, row in enumerate(reader):
        if ii == 0:
            continue
        if ii%4 == 0:
            y.append(np.copy(row))
            x.append(np.copy(xi))
            xi = []
        else:
            xi.append(row)
        if ii == 10:
            break
print('x: ', x)
print('y: ', y)


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


# if __name__ == "__main__": 
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split


class RBF:
    def __init__(self, n_inputs, n_hidden, n_outputs, centers, widths):
        """
        Parameters
        ----------
        n_inputs: int
            number of input nodes
        n_hidden: int
            number of rbf nodes
        n_outputs: int
            number output nodes
        centers: (1xn_hidden) array
            rbf function centers
        widths: (1xn_hidden) array
            rbf widths / std_dev
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.centers = centers
        self.widths = widths
        weight_init_range = 0.5
        self.weights = np.random.uniform(-weight_init_range, weight_init_range, (n_hidden, n_outputs))

    def gauss_kernel_func(self, x):
        """
        Applies the gauss kernel with the instantiated widths and centers for all rbf nodes to the single input.
        Returns the output of each rbf node on the single input.

        Parameters
        ----------
        x: np.array (n_inputs, )
            an input from the series of inputs
        """
        return np.exp((-1*(np.linalg.norm(x-self.centers, axis=1)**2)/(2*self.widths**2)))

    def inference(self, x):
        """
        Takes a set of input data and applies the gauss kernel function on them, then does the weighted summation
        to obtain network outputs of shape (n_input_data, n_output)

        Parameters
        ----------
        x: np.array (n_input_data, n_inputs)
        """
        print('====INFERENCE====')
        activities = []
        for input_val in x:
            activity = self.gauss_kernel_func(input_val)
            activities.append(activity)
        activities = np.asarray(activities)

        # outputs = np.sum(self.weights * activities, axis=1)
        # outputs = np.asarray(outputs)
        outputs = activities @ self.weights
        return outputs

    def forward(self, x):
        """
        Takes a set of input data and applies the gauss kernel function on them, then does the weighted summation
        to obtain network outputs of shape (n_input_data, n_output).
        Differs from self.inference because here we store the inputs, activities, and output to self variables that
        are used during inference.

        Parameters
        ----------
        x: np.array (n_input_data, n_inputs)
        """

        print('====FORWARD=====')
        self.input = x

        self.activities = []
        for input_val in self.input:
            activity = self.gauss_kernel_func(input_val)
            self.activities.append(activity)
        self.activities = np.asarray(self.activities)

        self.output = self.activities @ self.weights

        print('weights| (n_hiddenxn_output): ', self.weights.shape)
        print('input| (n_samplesxn_input): ', self.input.shape)
        print('activities| (n_samplesxn_hidden): ', self.activities.shape)
        print('output: | (n_hiddenx1)', self.output.shape)
        return self.output

    def backward(self, target_output):
        """
        Run this after running self.forward where we determine and store our series of activities.
        Accepts the matching target outputs for the inputs used during the forward pass. Performs
        the matrix (pseudo)inverse using the series of activities and target output obtain the output weights.
        Parameters
        ----------
        target_output: np.array (n_input_data, n_outputs)
            the target outputs of the inputs used during self.forward

        """
        if target_output.ndim == 1:
            target_output = np.expand_dims(target_output, 1)
        print('====BACKWARD===')
        print('activities series: ', self.activities.shape)
        print('target_output series: ', target_output.shape)
        print('weights: ', self.weights.shape)
        # normal inverse
        # if self.n_hidden >= len(self.input):
        if np.linalg.matrix_rank(self.activities) >= target_output.shape[0]:
            print('Inverse')
            # np.savez_compressed('test_data.npz', activities=self.activities, weights=self.weights, targets=target_output)
            self.weights = np.linalg.inv(self.activities) @ target_output

        # pseudo inverse
        else:
            print('Pseudo-inverse')
            self.weights = np.linalg.pinv(self.activities) @ target_output

        # print('activities: ', self.activities[:10, :10])
        # sanity_check = self.activities @ self.weights
        # print('target_out: ', target_output[:10])
        # print('sanity check: ', sanity_check[:10])

def get_kernel_centers(name=None, n=None):
    if name == "kmeans":
        pass
        # NOTE add https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        # applied to default vals
    if name == "random":
        # NOTE randomly select n from default
        pass

    return x

# def get_kernel_widths(name, n=None, val=None):
#     if name == 'const':
#         widths = np.ones((n, n)) * val
#         return widths

def ground_truth_func(x):
    if x[0]**2 + x[1]**2 <= 1:
        return 1
    else:
        return -1



#=========================================SETUP
# Generate our input and target data
x = np.ones((21, 21, 2))
for ii in range(0, x.shape[0]):
    for jj in range(0, x.shape[1]):
        xi = -2 + 0.2*ii
        xj = -2 + 0.2*jj
        x[ii, jj] = [xi, xj]

print('kernel centers original shape: ', np.asarray(x).shape)
x = x.reshape(np.prod(x.shape[:2]), x.shape[-1])
print('kernel centers flattened shape: ', np.asarray(x).shape)

# calculate the ground truth using the known function we're modelling
y = np.zeros(x.shape[0])
for ii, input_val in enumerate(x):
    y[ii] = ground_truth_func(input_val)

print('target output shape: ', np.asarray(y).shape)

# ========================== P1
# get our train/test split
# set test_size to 0.659 for 150 rbf nodes
train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=0)
# train_data = x
# train_labels = y
print('train_data shape: ', train_data.shape)
# print('test_data shape: ', test_data.shape)
print('train labels: ', train_labels.shape)
# print('test labels: ', test_labels.shape)

# Create our rbf network
# set our layer sizes
n_input = 2
n_hidden = np.prod(train_data.shape[0])
n_output = 1

# returns the widths of our rbfs
width = 0.2
widths = np.ones(n_hidden) * width
centers = train_data
print('widths: ', widths.shape)
print('centers: ', centers.shape)

# Using training data as rbf centers
print('Creating RBFN with %i input, %i hidden, and %i output' % (n_input, n_hidden, n_output))
rbf = RBF(n_input, n_hidden, n_output, centers=centers, widths=widths)
# run forward and backward pass to get weights
train_out = rbf.forward(train_data)
print('train output shape: ', train_out.shape)
rbf.backward(train_labels)
# sanity check, this should match our data exactly if activities is invertable
test_out = rbf.inference(train_data)
print('test output shape: ', test_out.shape)

plt.figure()
a1 = plt.subplot(111, projection='3d')
a1.scatter(x[:, 0], x[:, 1], y, label='GT')
a1.scatter(train_data[:, 0], train_data[:, 1], np.squeeze(train_out), label='TRAIN')

# c1 = a1.plot_trisurf(x[:, 0], x[:, 1], y, label='GT')
# c1._facecolors2d=c1._facecolors3d
# c1._edgecolors2d=c1._edgecolors3d
# # c2 = a1.plot_trisurf(test_data[:, 0], test_data[:, 1], test_out, label='TEST')
# c2 = a1.plot_trisurf(train_data[:, 0], train_data[:, 1], np.squeeze(train_out), label='TRAIN')
# # c2 = a1.plot_trisurf(train_data[:, 0], train_data[:, 1], test_out, label='TEST')
# c2._facecolors2d=c2._facecolors3d
# c2._edgecolors2d=c2._edgecolors3d

plt.legend()
plt.show()

# ========================== P2a Using 150RBFs - using training data as centers
# get our train/test split
train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.659, random_state=0)
# train_data = x
# train_labels = y
print('train_data shape: ', train_data.shape)
# print('test_data shape: ', test_data.shape)
print('train labels: ', train_labels.shape)
# print('test labels: ', test_labels.shape)

# Create our rbf network
# set our layer sizes
n_input = 2
n_hidden = np.prod(train_data.shape[0])
n_output = 1

# returns the widths of our rbfs
width = 0.2
widths = np.ones(n_hidden) * width
centers = train_data
print('widths: ', widths.shape)
print('centers: ', centers.shape)

# Using training data as rbf centers
print('Creating RBFN with %i input, %i hidden, and %i output' % (n_input, n_hidden, n_output))
rbf = RBF(n_input, n_hidden, n_output, centers=centers, widths=widths)
# run forward and backward pass to get weights
train_out = rbf.forward(train_data)
print('train output shape: ', train_out.shape)
rbf.backward(train_labels)
# sanity check, this should match our data exactly if activities is invertable
test_out = rbf.inference(train_data)
print('test output shape: ', test_out.shape)

plt.figure()
a1 = plt.subplot(111, projection='3d')
a1.scatter(x[:, 0], x[:, 1], y, label='GT')
a1.scatter(train_data[:, 0], train_data[:, 1], np.squeeze(train_out), label='TRAIN')

# c1 = a1.plot_trisurf(x[:, 0], x[:, 1], y, label='GT')
# c1._facecolors2d=c1._facecolors3d
# c1._edgecolors2d=c1._edgecolors3d
# # c2 = a1.plot_trisurf(test_data[:, 0], test_data[:, 1], test_out, label='TEST')
# c2 = a1.plot_trisurf(train_data[:, 0], train_data[:, 1], np.squeeze(train_out), label='TRAIN')
# # c2 = a1.plot_trisurf(train_data[:, 0], train_data[:, 1], test_out, label='TEST')
# c2._facecolors2d=c2._facecolors3d
# c2._edgecolors2d=c2._edgecolors3d

plt.legend()
plt.show()

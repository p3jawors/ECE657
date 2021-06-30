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
        print('WEIGHT MATRIX: ', self.weights.shape)

    def gauss_kernel_func(self, x):
        # for ii in range(0, len(self.centers)):
            # b = np.exp(
            #         (-1*
            #             ((x[0]-self.centers[ii][0])**2 + x[1]-self.centers[ii][1]**2)**2
            #             /(2*self.widths[ii]**2)))
        #     out.append(b)
        # out = np.asarray(out)
        # return out
        return np.exp((-1*(np.linalg.norm(x-self.centers, axis=1)**2)/(2*self.widths**2)))

    def inference(self, x):
        print('====INFERENCE====')
        activities = []
        for input_val in x:
            activity = self.gauss_kernel_func(input_val)
            activities.append(activity)
            # print('act: ', activity.shape)
            # print('wei: ', self.weights.shape)
            # out = np.sum(self.weights * activity)
            # print('out: ', out.shape)
            # outputs.append(out)
            # raise Exception
        activities = np.asarray(activities)
        outputs = np.sum(self.weights * activities, axis=1)
        outputs = np.asarray(outputs)

        # outputs = []
        # for input_val in x:
        #     activity = self.gauss_kernel_func(input_val)
        #     # print('act: ', activity.shape)
        #     # print('wei: ', self.weights.shape)
        #     out = np.sum(self.weights * activity)
        #     # print('out: ', out.shape)
        #     outputs.append(out)
        #     # raise Exception
        # outputs = np.asarray(outputs)
        # # print('activities: ', self.activities)
        # # print('weights: ', self.weights)
        # # print('outputs: ', outputs)

        return outputs

    def forward(self, x):
        print('====FORWARD=====')
        self.input = x

        self.activities = []
        for input_val in self.input:
            # print('------')
            # print('input: ', input_val)
            # print('centers: ', self.centers[:10])
            activity = self.gauss_kernel_func(input_val)
            # print('act: ', activity[:10])
            self.activities.append(activity)
            # raise Exception
        self.activities = np.asarray(self.activities)

        self.output = np.sum(self.weights * self.activities, axis=1)
        # print('activities: ', self.activities)
        # print('weights: ', self.weights)
        # print('outputs: ', self.output)

        print('weights| (n_hiddenxn_output): ', self.weights.shape)
        print('input| (n_samplesxn_input): ', self.input.shape)
        print('activities| (n_samplesxn_hidden): ', self.activities.shape)
        print('output: | (n_hiddenx1)', self.output.shape)
        return self.output

    def backward(self, target_output):
        if target_output.ndim == 1:
            target_output = np.expand_dims(target_output, 1)
        print('====BACKWARD===')
        print('full test set activities: ', self.activities.shape)
        print('full test set target outputs: ', target_output.shape)
        print('weights: ', self.weights.shape)
        # normal inverse
        if self.n_hidden >= len(self.input):
            print('Inverse')
            inv = np.linalg.inv(self.activities)
            print('inv: ', inv.shape)
            print(inv)
            print('output: ', target_output.shape)
            print(target_output)
            # print('inv*out: ', (inv*target_output).shape)
            # self.weights = np.matmul(np.linalg.inv(self.activities), target_output)
            self.weights = np.linalg.inv(self.activities) @ target_output
        # pseudo inverse
        else:
            print('Pseudo-inverse')
            self.weights = np.linalg.pinv(self.activities) * target_output
        print('FINAL WEIGHTS: ', self.weights.shape)
        print(self.weights)


def generate_inputs():
    #NOTE chane 21 and 20 here to var if want differtent shapes
    x = np.ones((21, 21, 2))
    for ii in range(0, 20+1):
        for jj in range(0, 20+1):
            xi = -2 + 0.2*ii
            xj = -2 + 0.2*jj
            x[ii, jj] = [xi, xj]
    return x

def get_kernel_centers(name=None, n=None):
    x = generate_inputs()
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

def get_ground_truth(x):
    #NOTE chane 21 and 20 here to var if want differtent shapes
    y = np.zeros((21, 21))
    for ii in range(0, 20+1):
        for jj in range(0, 20+1):
            y[ii, jj] = ground_truth_func(x[ii, jj])

    return y



#NOTE that the training data is also the rbf centers
# generates the centers of our RBFs
x = get_kernel_centers()
print('kernel centers original shape: ', np.asarray(x).shape)
# calculate the ground truth using the known function we're modelling
y = get_ground_truth(x)
print('target output original shape: ', np.asarray(y).shape)
x = x.reshape(np.prod(x.shape[:2]), x.shape[-1])
y = y.reshape(np.prod(y.shape))
print('kernel centers flattened shape: ', np.asarray(x).shape)
print('target output flattened shape: ', np.asarray(y).shape)
# flatten our input, otherwise scikit selects by column
# get our train/test split
train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=0)
print('train_data shape: ', train_data.shape)
print('test_data shape: ', test_data.shape)
print('train labels: ', train_labels.shape)
print('test labels: ', test_labels.shape)

# set our layer sizes
n_input = 2
n_hidden = np.prod(train_data.shape[0])
n_output = 1

# returns the widths of our rbfs
width = 1
widths = np.ones(n_hidden) * width
# centers = np.sum(train_data, axis=1)
centers = train_data
print('widths: ', widths.shape)
print('centers: ', centers.shape)
# widths = get_kernel_widths(name='const', n=len(xi), val=width)

# Using training data as rbf centers
print('Creating RBFN with %i input, %i hidden, and %i output' % (n_input, n_hidden, n_output))
rbf = RBF(n_input, n_hidden, n_output, centers=centers, widths=widths)
print('Running forward pass')
# print(train_data)
# print(train_labels)
# raise Exception
output = rbf.forward(train_data)
rbf.backward(train_labels)
print('output shape: ', output.shape)
test_out = rbf.inference(train_data)
# test_out = rbf.inference(train_data)
print('test output shape: ', test_out.shape)

plt.figure()
a1 = plt.subplot(111, projection='3d')
c1 = a1.plot_trisurf(x[:, 0], x[:, 1], y, label='GT')
c1._facecolors2d=c1._facecolors3d
c1._edgecolors2d=c1._edgecolors3d
c2 = a1.plot_trisurf(test_data[:, 0], test_data[:, 1], test_out, label='TEST')
c2._facecolors2d=c2._facecolors3d
c2._edgecolors2d=c2._edgecolors3d

plt.legend()
plt.show()

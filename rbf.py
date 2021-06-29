import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RBF:
    def __init__(self, n_inputs, n_hidden, n_outputs, centers, widths):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.centers = centers
        self.widths = widths
        weight_init_range = 0.5
        self.weights = np.random.uniform(-weight_init_range, weight_init_range, (n_hidden, n_outputs))
        print('WEIGHT MATRIX: ', self.weights.shape)

    def gauss_kernel_func(self, x):
        return np.exp((-1*(abs(x-self.centers))**2)/(2*self.widths**2))

    def forward(self, x):
        a = self.gauss_kernel_func(x)
        self.output = np.sum(self.weights * a, axis=1)
        return self.output

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
# calculate the ground truth using the known function we're modelling
y = get_ground_truth(x)
# get our train/test split
train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=0)
# reshape from 3d to 2d
train_data = train_data.reshape((np.prod(train_data.shape[:2]), train_data.shape[-1]))
test_data = test_data.reshape((np.prod(test_data.shape[:2]), test_data.shape[-1]))
# returns the widths of our rbfs
width = 1
widths = np.ones(train_data.shape) * width
# widths = get_kernel_widths(name='const', n=len(xi), val=width)


# set our layer sizes
n_input = 2
n_hidden = np.prod(train_data.shape[0])
n_output = 1

print('Creating RBFN with %i input, %i hidden, and %i output' % (n_input, n_hidden, n_output))
# Using training data as rbf centers
print('wdith: ', widths.shape)
print('vcen: ', train_data.shape)
rbf = RBF(n_input, n_hidden, n_output, centers=train_data, widths=widths)
print('Running forward pass')
output = rbf.forward(train_data)
print('output shape: ', output.shape)

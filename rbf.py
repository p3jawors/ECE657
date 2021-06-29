import numpy as np
import matplotlib.pyplot as plt

def gauss_kernel_func(x, v, sigma):
    return exp((-1*(abs(x-v))**2)/(2*sigma**2))

def generate_centers():
    x = np.ones((21, 21))
    for ii in range(0, 20+1):
        for jj in range(0, 20+1):
            xi = -2 + 0.2*ii
            xj = -2 + 0.2*jj
            x[ii, jj] = [xi, xj]

    # xi = np.asarray(xi)
    # xj = np.asarray(xj)
    # print('xi: ', xi)
    # print('xj: ', xj)
    # return xi, xj
    return x

def get_kernel_centers(name=None, n=None):
    x = generate_centers()
    if name == "kmeans":
        pass
        # NOTE add https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        # applied to default vals
    if name == "random":
        # NOTE randomly select n from default
        pass

    return x

def get_kernel_widths(name, n=None, val=None):
    if name == 'const':
        widths = np.ones((n, n)) * val
        return widths

def ground_truth_func(x):
    #TODO add function we're trying to model
    pass

#NOTE that the training data is also the rbf centers
# generates the centers of our RBFs
x = get_kernel_centers()

# returns the widths of our rbfs
width = 1
widths = get_kernel_widths(name='const', n=len(xi), val=width)

# set our layer sizes
n_input = len(x[0])
n_hidden = np.prod(x.shape)
n_output = 2

print('Creating RBFN with %i input, %i hidden, and %i output' % (n_input, n_hidden, n_output))

def forward():


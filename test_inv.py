import numpy as np
import matplotlib.pyplot as plt

# plot target vs output
def plot_results(target, output, title):
    plt.figure()
    plt.title(title)
    plt.plot(target, label='target')
    plt.plot(output, linestyle='--', label='output')
    plt.legend()
    plt.show()

# simple example with 2x2 activities and 2x1 target
a = np.array([[-1, 2], [0.4, 2]])
print('a: ', a)
print(a.shape)
target = np.array([-0.6, 0.4])
print('target: ', target)
print(target.shape)

# get our weights through matrix inv
w = np.linalg.inv(a) @ target
print('w = np.linalg.inv(a) @ target: ', w)
print(w.shape)
plot_results(target, a@w, 'simple')

# now load the data of activites from RBFN
data = np.load('test_data.npz')
activities = data['activities']
target_out = data['targets']
print('activities: ', activities.shape)
print('act rank : ', np.linalg.matrix_rank(activities))
print('target_out: ', target_out.shape)

w2 = np.linalg.inv(activities) @ target_out
print('w2: ', w2.shape)
plot_results(target_out, activities@w2, 'rbf_data')

rand_act = np.random.uniform(0, 1, activities.shape)
print('rand activities: ', rand_act.shape)
w3 = np.linalg.inv(rand_act) @ target_out
print('w3: ', w3.shape)
plot_results(target_out, rand_act@w3, 'rand_activities')

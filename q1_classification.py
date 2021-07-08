import numpy as np
import matplotlib.pyplot as plt
import timeit
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam

# load dataset with keras helper function
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# one-hot encode our targets
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# convert our rgb channels to be normalized
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0


n_epochs = 5
batch_size = 32
learning_rate = 0.001
verbose = 1

def sequential_model(learning_rate, n_layers, n_neurons):
    # define model A
    model = Sequential()
    for nn in range(n_layers):
        model.add(Dense(n_neurons, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def conv_modelA(learning_rate):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dense(512, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dense(512, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def conv_modelB(learning_rate):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dense(512, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_training_results(histories, cols, labels, title):
    plt.figure()
    plt.title(title)
    print('len histories: ', len(histories))
    print(histories[0])
    print(histories[0].history['loss'])
    for ii, hist in enumerate(histories):
        # plot loss
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(hist.history['loss'], color=cols[ii], label='train %s' % labels[ii])
        plt.plot(hist.history['val_loss'], color=cols[ii], label='test %s' % labels[ii], linestyle='--')
        plt.legend()
        # plot accuracy
        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(hist.history['accuracy'], color=cols[ii], label='train %s' % labels[ii])
        plt.plot(hist.history['val_accuracy'], color=cols[ii], label='test %s' % labels[ii], linestyle='--')
        plt.legend()
    plt.savefig('Q1_%s.png' % title)
    plt.show()

#======================= QUESTIONS =========================
# TODO's
# List preprocessing steps
# explain why using softmax outputlayer

# MLP changing n_layers
print('Running MLP varying n_layers')
n_layers = [1, 2, 3, 4, 5]
cols = ['r', 'b', 'g', 'y', 'm']
labels = []
histories = []
for ii in n_layers:
    labels.append('%i Layers' % ii)
    model = sequential_model(learning_rate, ii, 512)
    histories.append(model.fit(
            train_x,
            train_y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(test_x, test_y),
            verbose=verbose))

plot_training_results(histories, cols, labels, title='Varying n_layers')

print('Running MLP varying n_neurons')
# MLP changing neurons
n_neurons = [64, 128, 256, 512, 1024]
cols = ['r', 'b', 'g', 'y', 'm']
labels = []
histories = []
for ii in n_neurons:
    labels.append('%i Neurons' % ii)
    model = sequential_model(learning_rate, 2, ii)
    histories.append(model.fit(
            train_x,
            train_y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(test_x, test_y),
            verbose=verbose))

plot_training_results(histories, cols, labels, title='Varying n_neurons')

print('Running Conv comparing networks')
# Run the two conv models
convA = conv_modelA(learning_rate)
convB = conv_modelB(learning_rate)
cols = ['b', 'g']
labels = ['ConvA', 'ConvB']
conv_histories = []
times = []

start = timeit.default_timer()
conv_histories.append(convA.fit(
        train_x,
        train_y,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_x, test_y),
        verbose=verbose))
times.append(start-timeit.default_timer())

start = timeit.default_timer()
conv_histories.append(convB.fit(
        train_x,
        train_y,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_x, test_y),
        verbose=verbose))
times.append(start-timeit.default_timer())
print('CovA runtime: %.2f', times[0])
print('CovB runtime: %.2f', times[1])

plot_training_results(conv_histories, cols, labels, title='Comparing Conv Models')

# If we want to run inference...
# _, acc = model.evaluate(test_x, test_y, verbose=verbose)

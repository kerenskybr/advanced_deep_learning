from keras.models import Sequential
import keras
# print(dir(keras.layers ))
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Where to grab the data https://www.kaggle.com/zalando-research/fashionmnist/kernels
TRAIN_DATA = '/home/roger/Documents/machine_learning_examples/large_files/fashionmnist/fashion-mnist_train.csv'

# using pandas to load data from csv
train_file = pd.read_csv(TRAIN_DATA)
#test_file = pd.read_csv(TEST_DATA)
train_file = train_file.to_numpy()
np.random.shuffle(train_file)

print(train_file.shape)

# Reshape the data (60000) into the format 28x28x1
# Cause the image ix 28x28 1 channel. / 225.0 is to normalize the pixels
X = train_file[:,1:].reshape(-1,28,28,1) / 255.0
Y = train_file[:,0].astype(np.int32)

# Number of classes
K = len(set(Y))

# One hot encode the labels
Y = to_categorical(Y)

model = Sequential()

input_shape = (28, 28, 1)

# Architecture CNN CONV > POOL > CONV > POOL > FLATTEN > DENSE > DENSE
# kernel_size = the convolve kernel to the image
# Strides = the movement in pixels of the kernel
# Padding = the 'border' set to the image while convolving. Same is no padding. 0 is for a border of 0 and so on
# Batch normalization = Stabilizes the learning process and reduces the training epochs needed
# Activation = Decides if the neuron is fired or not
# Max Pooling = Reduces the spatial size of the representation. Grabs the bigger values of the matrix
# Dropout = remove some neurons reducing the complexity. Used to prevento overfitting
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.summary()

'''
# Functional syntax
i = Input(shape=(28,28,1))
x = Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=input_shape)(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=100)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = (Activation('softmax'))(x)

model = Model(inputs=i, outputs=x)
'''

#opt = keras.optimizers.Adam(learning_rate=0.01)
# Categorical, cause it's a classification problem
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)

print(r.history.keys())

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

# The size of image (resized)
IMAGE_SIZE = [100,100]

# path of the training and test data
train_path = 'fruits-360-small/Training'
valid_path = 'fruits-360-small/Validation'

# To grab all the files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# Folders (classes)
folders = glob(train_path + '/*')

# Grabbing a random image to display
# plt from matplotlib, load_img from keras and np.random from numpy
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

# Instance the VGG16 architecture, using data 100x100x3, with pre trained weigths
# removing the top, where we'll put our new architecture to use transfer learning
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# Now, it's time to add our layers, on the removed top of the pre-trained architecture
# We could add as much we want
x = Flatten()(vgg.output)
# Softmax -> multiclass classification
prediction = Dense(len(folders), activation='softmax')(x)

# Creating the model
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()
'''
Above that, the VGG architecture
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 36872     
=================================================================
Total params: 14,751,560
Trainable params: 36,872
Non-trainable params: 14,714,688

Here we see the add top and the total parameters. Note that the only new add
layers will be trained (Trainable params)
'''

# Time to train the model (fit, compile)
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'] # Metric to calculate while doing gradient descent
)

# Grabbing the data and doing some transformation
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)


# Testing the image generator
# Get the label mapping for confuzion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
	labels[v] = k

# Should output in different color, due vgg weights being BGR
for x, y in test_gen:
	print("min: ", x[0].min(), "max: ", x[0].max())
	plt.title(labels[np.argmax(y[0])])
	plt.imshow(x[0])
	plt.show()
	break



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

model.summary()
'''
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
'''
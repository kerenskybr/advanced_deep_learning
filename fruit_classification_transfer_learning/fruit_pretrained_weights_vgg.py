from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, Flatten
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import itertools

# The size of image (resized)
IMAGE_SIZE = [32,32]
batch_size = 64
epochs = 2

# path of the training and test data
train_path = 'fruits-360-small/Training'
valid_path = 'fruits-360-small/Validation'

# Full data
# train_path = '/home/roger/Documents/machine_learning_examples/large_files/fruits-360/Training'
# valid_path = '/home/roger/Documents/machine_learning_examples/large_files/fruits-360/Test'

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
    rotation_range=20, # Degree for random rotation
    width_shift_range=0.1, # Shift width for the image
    height_shift_range=0.1, # Shift height for the image
    shear_range=0.1, # Prune
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input #input a n numpy tensor rank 3, output numpy tensor
)

# Testing the image generator
# Get the label mapping for confuzion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)

# Creating an empty array
labels = [None] * len(test_gen.class_indices)

# Creating labels. 0 for Apple, 1 to avocado, etc
for k, v in test_gen.class_indices.items():
    labels[v] = k

# Should output in different color, due vgg weights being BGR
for x, y in test_gen:
	print("min: ", x[0].min(), "max: ", x[0].max())
	plt.title(labels[np.argmax(y[0])])
	plt.imshow(x[0])
	plt.show()
	break

# Creating the generators for train and valid
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

# Fit (train) the model
# Fit generator stand for generators
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size
)

# Ploting loss
print(r.history.keys())

plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train_acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def get_confusion_matrix(data_path, N):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm

cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')
# Tf 1.15.2

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize

import keras.backend as K
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_style", required=True, help="path to style image")
ap.add_argument("-m", "--input_image", required=True, help="path to image to receive the syle")
ap.add_argument("-o", "--output", required=True, help="path to save the image")
ap.add_argument("-l", "--layer", required=True, help="layer of the net", default=13)

args = vars(ap.parse_args())


def VGG16_AvgPool(shape):
    # Getting rid of maxpooling who throws away information
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # Replace it with average pooling
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)
    return new_model


def VGG16_AvgPool_CutOff(shape, num_convs):
    # There area 13 convolutions in total, we can pick
    # Any of them as output

    if num_convs < 1 or num_convs > 13:
        print('[info] Num of convs must be in the range [1,13]')
        return None

    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    n = 0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        new_model.add(layer)
        if n >= num_convs:
            break

    return new_model


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return  x


def gram_matrix(img):
    # Input is (H, W, C)C = feature maps
    # We first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img, (2,0,1)))

    # Now calculate the gram matrix
    # Gram = XX^T /N
    # The constant is not important since we'll weighting these
    G = K.dot(X, K.transpose(X) / img.get_shape().num_elements())
    return G


def style_loss(y, t):
    # Calculate the grid matrix from both images and calculate the mean
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


# Let's generalize this and put it into a function
def minimize(fn, epochs, batch_shape):
	t0 = datetime.now()
	losses = []
	x = np.random.randn(np.prod(batch_shape))
	for i in range(epochs):
		x, l, _ = fmin_l_bfgs_b(
			func=fn,
			x0=x,
			maxfun=20
		)
	x = np.clip(x, -127, 127)
	print("iter=%s, loss=%s" % (i, l))
	losses.append(l)

	print("duration:", datetime.now() - t0)
	plt.plot(losses)
	plt.savefig('loss.png')
	#plt.show()
	
	newimg = x.reshape(*batch_shape)
	final_img = unpreprocess(newimg)
	return final_img[0]


# load the content image
def load_img_and_preprocess(path, shape=None):
	# Grabbing the source image and converting to array
	img = image.load_img(path, target_size=shape)

	# convert image to array and preprocess for vgg
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	return x


content_img = load_img_and_preprocess(args["input_image"])

# resize the style image
# since we don't care too much about warping it
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess(
  # 'styles/starrynight.jpg',
  # 'styles/flowercarrier.jpg',
  # 'styles/monalisa.jpg',
  args["input_style"],
  (h, w)
)


# we'll use this throughout the rest of the script
batch_shape = content_img.shape
shape = content_img.shape[1:]


# we want to make only 1 VGG here
# as you'll see later, the final model needs
# to have a common input
vgg = VGG16_AvgPool(shape)


# create the content model
# we only want 1 output
# remember you can call vgg.summary() to see a list of layers
# 1,2,4,5,7-9,11-13,15-17
# The smaller the layer, the less features it learns
content_model = Model(vgg.input, vgg.layers[args['layer']].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))


# create the style model
# we want multiple outputs
# we will take the same approach as in style_transfer2.py
symbolic_conv_outputs = [
  layer.get_output_at(1) for layer in vgg.layers \
  if layer.name.endswith('conv1')
]

# make a big model that outputs multiple layers' outputs
style_model = Model(vgg.input, symbolic_conv_outputs)

# calculate the targets that are output at each layer
style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

# we will assume the weight of the content loss is 1
# and only weight the style losses
style_weights = [0.0002,0.0004,0.0003,0.0005,0.0002]
#style_weights = [1,2,3,4,5]


# create the total loss which is the sum of content + style loss
loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
	# gram_matrix() expects a (H, W, C) as input
	loss += w * style_loss(symbolic[0], actual[0])


# once again, create the gradients and loss + grads function
# note: it doesn't matter which model's input you use
# they are both pointing to the same keras Input layer in memory
grads = K.gradients(loss, vgg.input)

# Creating the model
get_loss_and_grads = K.function(
  inputs=[vgg.input],
  outputs=[loss] + grads
)


def get_loss_and_grads_wrapper(x_vec):
	l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
	return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
#plt.show()
plt.savefig(args["output"])
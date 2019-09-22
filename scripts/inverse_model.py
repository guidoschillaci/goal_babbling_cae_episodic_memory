# author Guido Schillaci, Dr.rer.nat. - Scuola Superiore Sant'Anna
# Guido Schillaci <guido.schillaci@santannapisa.it>

from tensorflow.python.keras.layers import Input, Dense, Reshape,  Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import datetime
from random import sample


epochs =2
image_size = 128

# readapted from https://blog.keras.io/building-autoencoders-in-keras.html
def build_autoencoder():
	autoencoder = None
	# this is our input placeholder
	input_img = Input(shape=(image_size, image_size, 1))

	print 'input shape ', input_img.shape

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x) # encoded shape should by (2, 2, 8), so 4D

	#print 'encoded shape ', encoded.shape

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
	
	#print 'decoded shape ', decoded.shape

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	return autoencoder
	

def train_autoencoder(autoencoder, train_data, test_data):
	tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

	autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=32, shuffle=True, validation_data=(test_data, test_data), callbacks=[tensorboard_callback], verbose=1)
	return autoencoder

def build_inverse_model(inverse_model):
    inverse_model = None


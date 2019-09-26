# author Guido Schillaci, Dr.rer.nat. - Scuola Superiore Sant'Anna
# Guido Schillaci <guido.schillaci@santannapisa.it>


from tensorflow.python.keras.layers import Input, Dense, Reshape,  Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
import gzip
import matplotlib.pyplot as plt
import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
import datetime
import  random
from util import clamp_x, clamp_y, x_lims, y_lims, z_lims, speed_lim
from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans


from plots import plots_cae, plots_cae_decoded, plot_som, plots_fwd_model, plots_full_model,plot_som_scatter, imscatter
from models import Models
# from dcgan import DCGAN
from interest_model import InterestModel
from minisom import MiniSom

import cv2

# Limit GPU memory usage
use_gpu_fraction = 0.40


models = Models()

class OfflineTraining():

	def __init__(self):
		## parameters
		self.cae_epochs =20 # training epochs for convolutional autoencoder
		self.fwd_epochs =10 # training epochs for forward model
		self.inv_epochs =2 # training epochs for inverse model
		self.batch_size = 16
		self.image_size=64 # use power of two
		self.num_test = 500 # how many test samples from the dataset
		self.goal_som_size = 3
		self.code_size = 32 # use power of two

		self.test_img_size = 50

	def configure_keras(self, gpu_fraction=0.10): # percentage of gpu resources to use
		if tf.__version__ < "1.8.0":
			config = tf.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
			session = tf.Session(config=config)
		else:
			config = tf.compat.v1.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
			session = tf.compat.v1.Session(config=config)

	def run (self):
		print ('starting time', datetime.datetime.now())
		#configure_keras()

		directory='./pretrained_models/'

		if not os.path.isdir(directory):
			print('directory ', directory, ' does not exist. Creating it...')
			os.mkdir(directory)

		dataset = 'rgb_rectified/compressed_dataset.pkl'

		# Loading data
		step = 100
		train_images, test_images, train_cmds, test_cmds, train_pos, test_pos = models.load_data(dataset, self.image_size, step = step)

		# Load or train models
		autoencoder, encoder, decoder = models.load_autoencoder(directory = directory, train_images = train_images, batch_size=self.batch_size, epochs=self.cae_epochs, code_size = self.code_size, image_size = self.image_size)
		encoded = encoder.predict([test_images[0:self.test_img_size]])
		plots_cae_decoded(decoder, encoded, test_images[0:self.test_img_size],
						  image_size = self.image_size, directory=directory)

		goal_som = models.load_som(directory = directory, encoder=encoder, train_images = train_images, goal_size = self.goal_som_size)
		#plot_som_scatter( encoder, goal_som, train_images)

		k_means = models.load_kmeans(directory = directory, encoder= encoder, train_images = train_images, goal_size = self.goal_som_size)

		interestModel = InterestModel(self.goal_som_size)

		# build forward model
		#forward_model = models.load_forward_model(autoencoder, train_pos = train_pos, test_pos = test_pos, train_images = train_images, test_images = test_images, epochs = self.fwd_epochs, image_size = self.image_size, code_size = self.code_size)
		train_codes = encoder.predict(train_images)
		test_codes = encoder.predict(test_images)
		## uncomment this to train the forward model with the offline data. It is preferralbe to use the goal_babbling with the simulated data
		#forward_code_model = models.load_forward_code_model(train_pos = train_pos, test_pos = test_pos, train_codes = train_codes, test_codes = test_codes, epochs = self.fwd_epochs, image_size = self.image_size, code_size = self.code_size)

		# build DCGAN forward model
	#	dcgan_fwd_model = DCGAN(img_rows=image_size, img_cols=image_size)
	#	dcgan_fwd_model.train(all_positions, all_images, epochs=10000, batch_size=32, save_interval=50)

		# build inverse model
		## uncomment this to train the inverse model with the offline data. It is preferralbe to use the goal_babbling with the simulated data
		#inverse_model = models.load_inverse_model(train_pos = train_pos, test_pos = test_pos, train_images = train_images, test_images = test_images, epochs = self.inv_epochs, image_size = self.image_size)

		#motor_pred = inverse_model.predict([train_images])

		# print all
	#	np.set_printoptions(threshold='nan')
		#print np.hstack(( train_pos, motor_pred))
		#print np.linalg.norm(motor_pred-all_commands, axis=1)
		#print 'total error ', np.linalg.norm(motor_pred-train_pos)
		#print 'average error ', np.mean( np.linalg.norm(motor_pred-train_pos, axis=1))

		
		#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
		#X_tsne = tsne.fit_transform(encoded)

		# Plot images according to t-sne embedding
		#print("Plotting t-SNE visualization of the CAE latent space...")
		#fig, ax = plt.subplots()
		#imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=test_images[0:self.test_img_size], ax=ax, zoom=0.8, imageSize=self.image_size)
		#plt.show()

#		fig, ax = plt.subplots()
#		test_motor_pred = inverse_model.predict([test_images[0:self.test_img_size]])
#		imscatter(test_motor_pred[:, 0], test_motor_pred[:, 1], imageData=test_images[0:self.test_img_size], ax=ax, zoom=0.4, imageSize=self.image_size)
#		plt.show()

		#test_motor_pred = inverse_model.predict([test_images[0:self.test_img_size]])
		#print np.hstack((test_motor_pred,test_pos[0:self.test_img_size]))
		#test_img_pred = forward_model.predict([test_motor_pred])
		#test_img_pred = forward_model.predict([test_pos[0:self.test_img_size]])
		'''
		test_img_pred = decoder.predict(forward_code_model.predict([test_pos[0:self.test_img_size]]))
		fig = plt.figure(figsize=(25, 15))
		size=5
		t = test_images[0:size]
		for i in range(size):
			# display original
			ax1 = plt.subplot(2, size, i+1)
			plt.imshow(t[i].reshape(self.image_size, self.image_size), cmap = 'gray')
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax2 = plt.subplot(2, size, i+size+1)
			plt.imshow(test_img_pred[i].reshape(self.image_size, self.image_size), cmap= 'gray')		
			ax2.get_xaxis().set_visible(False)
			ax2.get_yaxis().set_visible(False)
		plt.show()
		'''

if __name__ == '__main__':
	ot = OfflineTraining()
	ot.configure_keras(use_gpu_fraction)
	ot.run()

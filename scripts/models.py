# author Guido Schillaci, Dr.rer.nat. - Humboldt-Universitaet zu Berlin
# Guido Schillaci <guido.schillaci@informatik.hu-berlin.de>

from tensorflow.python.keras.layers import Input, Dense, Dropout, Reshape, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.models import Model, Sequential, load_model, Input
from tensorflow.python.keras.callbacks import TensorBoard, Callback
from tensorflow.python.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.python.keras import optimizers

import cPickle, h5py, gzip
import cv2
import numpy as np
from util import x_lims, y_lims, z_lims, speed_lim
import sys, getopt, os
from minisom import MiniSom

from sklearn.cluster import KMeans

class Models():

	def __init__(self):
		print 'Models created'

	def getLayerIndexByName(self, model, layername):
		for idx, layer in enumerate(model.layers):
		    if layer.name == layername:
		        return idx

	def parse_data(self, file_name, pixels, reshape, step, channels=1):

		images = []
		positions = []
		commands = []
		test_pos = []
		with gzip.open(file_name, 'rb') as memory_file:
			memories = cPickle.load(memory_file)
			print 'converting data...'
			count = 0
			for memory in memories:
				image = memory['image']
				#image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

				if channels == 1:
					image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#			image = cv2.resize(image, (pixels, pixels))
				image = cv2.resize(image, (pixels, pixels))

				images.append(np.asarray(image))

				cmd = memory['command']
	#			commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1], float(cmd.z) / z_lims[1], float(cmd.speed) / speed_lim])
	#			commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1], float(cmd.z) / z_lims[1]])
				commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1]])
				pos = memory['position']
	#			positions.append([float(pos.x) /x_lims[1], float(pos.y) / y_lims[1], float(pos.z) / z_lims[1]])
				positions.append([float(pos.x) /x_lims[1], float(pos.y) / y_lims[1]])

				#if count%step == 0:
				#	img_name = './sample_images/img_' + str(count) + '.jpg'
				#	cv2.imwrite(img_name, image)
				count+=1

		positions = np.asarray(positions)
		commands = np.asarray(commands)
		images = np.asarray(images)
		if reshape:
			images = images.reshape((len(images), pixels, pixels, channels))

		return images.astype('float32') / 255, commands, positions

	def load_data(self, dataset, image_size, step):

		images, commands, positions = self.parse_data(dataset, step = step, reshape=True, pixels = image_size)
		# split train and test data
		#f = 500  #number of elements for testing
	#		indices = random.sample(range( len(self.all_positions) ),f)
		indices = range(0, len(positions), step)
		# split images
		test_images = images[indices]
		train_images = images
		test_cmds = commands[indices]
		train_cmds = commands
		test_pos = positions[indices]
		train_pos = positions
		print "number of test images: ", len(test_images)

		return train_images, test_images, train_cmds, test_cmds, train_pos, test_pos


	def load_autoencoder(self, directory='./', cae_file = 'autoencoder.h5', e_file = 'encoder.h5', d_file = 'decoder.h5', image_size = 64, channels = 1, code_size = 4, train_images = None, batch_size = 32, epochs = 2, train_offline=True):

		cae_file = directory + cae_file
		e_file = directory + e_file
		d_file = directory + d_file

		autoencoder = []
		encoder=[]
		decoder=[]
		if os.path.isfile(cae_file) and os.path.isfile(e_file) and os.path.isfile(d_file): # trained cae model already exists
			# load convolutional autoencoder 
			print 'Loading existing pre-trained autoencoder: ', cae_file
			K.clear_session()
			autoencoder = load_model(cae_file)
			# Create a separate encoder model
			encoder_inp = Input(shape=(image_size, image_size, channels))
			encoder_layer = autoencoder.layers[1](encoder_inp)
			enc_layer_idx = self.getLayerIndexByName(autoencoder, 'encoded')
			for i in range(2,enc_layer_idx+1):
				encoder_layer = autoencoder.layers[i](encoder_layer)
			encoder = Model(encoder_inp, encoder_layer)
			print encoder.summary()
			# Create a separate decoder model
			decoder_inp = Input(shape=(code_size,))
			decoder_layer = autoencoder.layers[enc_layer_idx+1](decoder_inp)			
			for i in range(enc_layer_idx+2, len(autoencoder.layers)):
				decoder_layer = autoencoder.layers[i](decoder_layer)
#			decoder_layer = autoencoder.layers[-(enc_layer_idx)](decoder_inp)
#			for i in range(-(enc_layer_idx-1),-0):
#				decoder_layer = autoencoder.layers[i](decoder_layer)
			decoder = Model(decoder_inp, decoder_layer)
			print decoder.summary()
			print 'Autoencoder loaded'
		else:
			print 'Could not find autoencoder files. Building and training a new one.'
			autoencoder, encoder, decoder = self.build_autoencoder(code_size = code_size, image_size = image_size)
			if train_offline:
				if train_images is None:
					print 'I need some images to train the autoencoder'
					sys.exit(1)
				self.train_autoencoder(autoencoder, encoder, decoder, train_images, batch_size=batch_size, cae_epochs=epochs)
		return autoencoder, encoder, decoder

	def load_forward_model(self, autoencoder, directory = './', filename = 'forward_model.h5', train = True, train_images = None, test_images = None, train_pos = None, test_pos = None, epochs = 2, batch_size = 32, image_size = 64, code_size = 4):

		filename = directory+ filename

		forward_model = []
		if os.path.isfile(filename):
			print 'Loading existing pre-trained forward model: ', filename
			forward_model = load_model(filename)
			print 'Forward model loaded'
		else:
			print ' image_size load ' , image_size
			forward_model = self.build_forward_model(image_size, autoencoder, code_size = code_size)
			print 'Forward model does not exist, yet. Built and compiled a new one'
			if train:
				if train_images is None or test_images is None or train_pos is None or test_pos is None:
					'I need data for training the forward model'
					sys.exit(1)
				self.train_forward_model(forward_model, train_pos, train_images, test_pos, test_images, batch_size=batch_size, epochs=epochs)
		return forward_model

	def load_forward_code_model(self, directory = './', filename = 'forward_code_model.h5', train = True, train_codes = None, test_codes = None, train_pos = None, test_pos = None, epochs = 2, batch_size = 32, image_size = 64, code_size = 4):
		filename = directory+filename

		forward_model = []
		if os.path.isfile(filename):
			print 'Loading existing pre-trained forward code model: ', filename
			forward_model = load_model(filename)
			print 'Forward code model loaded'
		else:
			print ' image_size load ' , image_size
			forward_model = self.build_forward_code_model(code_size = code_size)
			print 'Forward model does not exist, yet. Built and compiled a new one'
			if train:
				if train_codes is None or test_codes is None or train_pos is None or test_pos is None:
					'I need data for training the forward model'
					sys.exit(1)
				self.train_forward_code_model(forward_model, train_pos, train_codes, test_pos, test_codes, batch_size=batch_size, epochs = epochs)
		return forward_model

	def load_inverse_model(self, directory = './', filename = 'inverse_model.h5', train = True, train_images = None, test_images = None, train_pos = None, test_pos = None, epochs = 2, batch_size = 32, image_size=64 ):
		filename = directory + filename
		# build inverse model
		if os.path.isfile(filename):
			print 'Loading existing pre-trained inverse model: ', filename
			inverse_model = load_model(filename)
			print 'Inverse model loaded'
		else:
			inverse_model = self.build_inverse_model(image_size)			
			print 'Inverse model does not exist, yet. Built and compiled a new one'
			if train:
				if train_images is None or test_images is None or train_pos is None or test_pos is None:
					'I need data for training the inverse model'
					sys.exit(1)
				self.train_inverse_model(inverse_model, train_images, train_pos, test_images,  test_pos, batch_size=batch_size, epochs=epochs)
		return inverse_model


	def load_inverse_code_model(self, directory = './', filename = 'inverse_code_model.h5', train = True, train_codes = None, test_codes = None, train_pos = None, test_pos = None, epochs = 2, batch_size = 32, image_size=64, code_size=4 ):
		filename = directory + filename
		# build inverse model
		if os.path.isfile(filename):
			print 'Loading existing pre-trained inverse code model: ', filename
			inverse_model = load_model(filename)
			print 'Inverse model loaded'
		else:
			inverse_model = self.build_inverse_code_model(code_size = code_size)			
			print 'Inverse model does not exist, yet. Built and compiled a new one'
			if train:
				if train_codes is None or test_codes is None or train_pos is None or test_pos is None:
					'I need data for training the inverse code model'
					sys.exit(1)
				self.train_inverse_model(inverse_model, train_codes, train_pos, test_codes,  test_pos, batch_size=batch_size, epochs=epochs)
		return inverse_model

	def load_kmeans(self, directory = './', filename = 'kmeans.sav', encoder=  None, train_images=None, goal_size = 3, batch_size = 32):
		filename = directory + filename

		kmeans = None
		if os.path.isfile(filename):
			print 'Loading existing KMeans...'
			kmeans = cPickle.load(open(filename, 'rb'))

		else: 
			print 'Could not find KMeans file.'

			if train_images is None:
				print 'I need an encoder and  some images to train the kMeans'
				sys.exit(1)

			# encoding test images
			print 'Encoding train images...'
			train_images_codes = encoder.predict(train_images)
			code_size = len(train_images_codes[0])

			print 'training k-means'
			#kmeans = MiniBatchKMeans(n_clusters=goal_som_size*goal_som_size,random_state=0,batch_size=batch_size)
			kmeans = KMeans(n_clusters=goal_size*goal_size,random_state=0)
	#		for i in xrange(0, len(train_images_codes), batch_size):
	#			kmeans = kmeans.partial_fit(train_images_codes[i:i+batch_size])
			kmeans = kmeans.fit(train_images_codes)
			print 'saving k-means'
			cPickle.dump(kmeans, open(filename, 'wb'))

		return kmeans

	def load_som(self, directory = './', filename = 'goal_som.h5', encoder = None, train_images = None,  goal_size = 3):
		filename = directory + filename

		goal_som = None
		if os.path.isfile(filename):
			print 'Loading existing trained SOM...'
			h5f = h5py.File(filename,'r')
			weights= h5f['goal_som'][:]
			code_size = len(weights[0][0])
			h5f.close()
			print 'code_size read ' , code_size
			goal_som = MiniSom(goal_size, goal_size, code_size) 
			goal_som._weights = weights
			print len(weights)
			print 'Goal SOM loaded! Number of goals: ', str(goal_size*goal_size)
		else:
			print 'Could not find Goal SOM files.'
			if encoder is None or train_images is None:
				print 'I need an encoder and some sample images to train a new SOM!'
				sys.exit(1)
			print 'Creating a new one'
			# creating self-organising maps for clustering the image codes <> the image goals

			# encoding test images
			print 'Encoding train images...'
			train_images_codes = encoder.predict(train_images)
			code_size = len(train_images_codes[0])

			goal_som = MiniSom(goal_size, goal_size, code_size, sigma = 0.5, learning_rate = 0.5) 
			print 'Initialising goal SOM...'
			goal_som.random_weights_init(train_images_codes)

			#plot_som_scatter( encoder, goal_som, train_images)

			print 'som quantization error: ', goal_som.quantization_error(train_images_codes)
			print("Training goal SOM...")
			goal_som.train_random(train_images_codes, 100)  # random training


			trained_som_weights = goal_som.get_weights().copy()
			som_file = h5py.File(filename, 'w')
			som_file.create_dataset('goal_som', data=trained_som_weights)
			som_file.close()
			print("SOM trained and saved!")
		return goal_som

	# build and compile the convolutional autoencoder
	def build_autoencoder(self, image_size = 64, max_pool_size = 2, conv_size = 3, channels = 1, code_size = 4):
		autoencoder = None
		# this is our input placeholder
		input_img = Input(shape=(image_size, image_size, channels), name = 'input')
		x = Conv2D(256, (conv_size, conv_size), activation='relu', padding='same')(input_img) # tanh?
		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
		x = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same')(x)
		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
		x = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same')(x)
		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
		#encoded = MaxPooling2D((max_pool_size, max_pool_size), padding='same', name = 'encoded')(x)
	

#		x = Conv2D(16, (conv_size, conv_size), activation='relu', padding='same')(x)
#		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
#		x = Conv2D(8, (conv_size, conv_size), activation='relu', padding='same')(x)
#		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
#		x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
#		x = MaxPooling2D((2, 2), padding='same')(x)
#		print x.shape
		x= Flatten()(x)
#		x= Dense(code_size)(x)
#		encoded =  Reshape(target_shape=(4,), name='encoded')(x) # encoded shape should by 2*2*2
		encoded = Dense(code_size, name='encoded')(x) 	


		print 'encoded shape ', encoded.shape
	#	x = Reshape(target_shape=(8,8,1))(encoded) #-12
#		x = Dense(code_size, activation='relu')(encoded)
#		x = Reshape(target_shape=(4, 4, 1))(x) #-12
#		x = Conv2D(2, (conv_size, conv_size), activation='relu', padding='same')(x) #-13
#		x = UpSampling2D((max_pool_size, max_pool_size))(x)
#		x = Conv2D(8, (conv_size, conv_size), activation='relu', padding='same')(x)
#		x = UpSampling2D((max_pool_size, max_pool_size))(x)
#		x = Conv2D(8, (conv_size, conv_size), activation='relu', padding='same')(x)
#		x = UpSampling2D((max_pool_size, max_pool_size))(x)
	#	x = Conv2D(8, (conv_size, conv_size), activation='relu', padding='same')(x)
	#	x = UpSampling2D((max_pool_size, max_pool_size))(x)
		ims = 8
		first = True
		x = Dense(ims*ims, activation='relu')(encoded)
		x = Reshape(target_shape=(ims, ims, 1))(x) #-12
		while ims!=image_size:
			x = Conv2D(ims*ims/2, (conv_size, conv_size), activation='relu', padding='same')(x)
			x = UpSampling2D((max_pool_size, max_pool_size))(x)
			ims = ims*max_pool_size
		decoded = Conv2D(channels, (conv_size, conv_size), activation = 'sigmoid', padding='same', name= 'decoded')(x)
		
		print 'decoded shape ', decoded.shape

		autoencoder = Model(input_img, decoded)
		autoencoder.compile(optimizer='adam', loss='mean_squared_error')
	#	autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		#autoencoder.summary()

		# Create a separate encoder model
		encoder = Model(input_img, encoded)
		encoder.compile(optimizer='adam', loss='mean_squared_error')
		encoder.summary()

		# Create a separate decoder model
		decoder_inp = Input(shape=(code_size,))
#		decoder_inp = Input(shape=encoded.output_shape)
		enc_layer_idx = self.getLayerIndexByName(autoencoder, 'encoded')
		print 'encoder layer idx ', enc_layer_idx
		decoder_layer = autoencoder.layers[enc_layer_idx+1](decoder_inp)			
		for i in range(enc_layer_idx+2, len(autoencoder.layers)):
			decoder_layer = autoencoder.layers[i](decoder_layer)
		decoder = Model(decoder_inp, decoder_layer)
		decoder.compile(optimizer='adam', loss='mean_squared_error')
		decoder.summary()

		return autoencoder, encoder, decoder

	def train_autoencoder(self, autoencoder, encoder, decoder, train_data, batch_size = 32, cae_epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/cae', histogram_freq=0, write_graph=True, write_images=True)

		autoencoder.fit(train_data, train_data, epochs=cae_epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard_callback], verbose=1)
		
		autoencoder.save( './models/autoencoder.h5')
		encoder.save('./models/encoder.h5')
		decoder.save('./models/decoder.h5')
		print 'autoencoder trained and saved '


	def train_autoencoder_on_batch(self, autoencoder, encoder, decoder, train_data, batch_size = 1, cae_epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/cae', histogram_freq=0, write_graph=True, write_images=True)

		autoencoder.fit(train_data, train_data, epochs=cae_epochs, batch_size=batch_size, shuffle=True,
						callbacks=[tensorboard_callback], verbose=1)
		#inverse_model.fit(images, motor_cmd, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)#, callbacks=[showLR()])
		#inverse_model.train_on_batch(images, motor_cmd)
		print 'autoencoder updated'

	def build_forward_code_model(self, input_dim=2, code_size = 4):
		print 'building forward code model...'
		#forward_model = Sequential()

		# create fwd model layers, partly from autoencoder (decoder part
		cmd_fwd_inp = Input(shape=(input_dim,), name = 'fwd_input')
		x = Dense(code_size, activation='tanh')(cmd_fwd_inp)
		#x = Dropout(0.3)(x)
		#x = Dense(code_size*30, activation='relu')(x)
		#x = Dropout(0.3)(x)
		x = Dense(code_size*10, activation='tanh')(x)
		x = Dense(code_size*10, activation='tanh')(x)
		#x = Dropout(0.3)(x)
		code = Dense(code_size, name = 'output')(x)
		fwd_model = Model(cmd_fwd_inp, code)
		sgd = optimizers.SGD(lr=0.0014, decay=0.0, momentum=0.8, nesterov=True)
		#fwd_model.compile(optimizer='adadelta', loss='mean_squared_error')
		fwd_model.compile(optimizer=sgd, loss='mean_squared_error')
		print 'forward model'
		fwd_model.summary()
		return fwd_model

	def build_forward_model(self, image_size, autoencoder, input_dim=2, output_channels=1, max_pool_size = 2, conv_size = 3, channels =1, code_size = 4):
		print 'building forward model...'
		#forward_model = Sequential()
		enc_layer_idx = self.getLayerIndexByName(autoencoder, 'encoded')

		# create fwd model layers, partly from autoencoder (decoder part
		cmd_fwd_inp = Input(shape=(input_dim,), name = 'fwd_input')
		x = Dense(code_size, name = 'fwd_dense_1', activation='relu')(cmd_fwd_inp)


		ims = 8
		first = True
		x = Dense(ims*ims, activation='relu')(x)
		x = Reshape(target_shape=(ims, ims, 1))(x) #-12
		while ims!=image_size:
			x = Conv2D(ims*ims*max_pool_size, (conv_size, conv_size), activation='relu', padding='same')(x)
			x = UpSampling2D((max_pool_size, max_pool_size))(x)
			ims = ims*max_pool_size
		decoded = Conv2D(channels, (conv_size, conv_size), padding='same', name= 'decoded')(x)
		fwd_model = Model(cmd_fwd_inp, decoded)
		fwd_model.compile(optimizer='adam', loss='mse')

		print 'forward model'
		fwd_model.summary()
		return fwd_model


	def train_forward_model(self, forward_model, positions, images, test_pos, test_img, batch_size=128, epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/fwd', histogram_freq=0, write_graph=True, write_images=True)
		forward_model.fit(positions, images, validation_data=[test_pos, test_img], epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[tensorboard_callback])

		forward_model.save('./models/forward_model.h5')
		print 'Forward model trained and saved'

	def train_forward_code_model(self, forward_model, positions, codes, test_pos, test_codes, batch_size=128, epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/fwd_code', histogram_freq=0, write_graph=True, write_images=True)

		forward_model.fit(positions, codes, validation_data=[test_pos, test_codes], epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[tensorboard_callback])

		forward_model.save('./models/forward_code_model.h5')
		print 'Forward code model trained and saved'

	def train_forward_model_on_batch(self, forward_model, positions, images, batch_size=32, epochs = 1):
		forward_model.fit(positions, images, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)#, callbacks=[showLR()])
		#forward_model.train_on_batch(positions, images)
		print 'Forward model updated'

	def train_forward_code_model_on_batch(self, forward_model, positions, codes, batch_size=32, epochs = 1):
		#tensorboard_callback = TensorBoard(log_dir='./logs/fwd_code', histogram_freq=0, write_graph=True, write_images=True)
		forward_model.fit(positions, codes, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)#, callbacks=[tensorboard_callback])
		print 'Forward code model updated'


	def build_inverse_model(self,  image_size, input_dim=2, max_pool_size = 2, conv_size = 3, channels=1):
		print 'building inverse model...'

		input_img = Input(shape=(image_size, image_size, channels))
		x = Conv2D(4, (conv_size, conv_size), activation='relu', padding='same')(input_img) # tanh?
	#	x = Dropout(0.3)(x)
#		x = Dropout(0.3)(x)
		x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
	#	x = Conv2D(8, (conv_size, conv_size), activation='relu', padding='same')(x)
	#	x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
	#	x = Conv2D(64, (conv_size, conv_size), activation='relu', padding='same')(x)
	#	x = Dropout(0.3)(x)
	#	x = MaxPooling2D((max_pool_size, max_pool_size), padding='same')(x)
	#	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	#	x = Dropout(0.3)(x)
	#	x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	#	x = Dropout(0.3)(x)
#		x = Dropout(0.3)(x)
		x = MaxPooling2D((2, 2), padding='same')(x)
		#x = Reshape(target_shape=(2*2*4,))(x) 
		x= Flatten()(x)
		x= Dense(image_size, activation='relu')(x)
#		x = Dropout(0.3)(x)
		#command = Dense(input_dim, activation='relu', name='command')(x) 	
		command = Dense(input_dim, name='command')(x) 	

		inv_model = Model(input_img, command)
		#inv_model.compile(optimizer='adadelta', loss='mean_squared_error')	
		#inv_model.compile(optimizer='adam', loss='mse')	

		#adam= optimizers.Adam(lr=0.001)
		#adadelta= optimizers.Adadelta(lr=0.001)
		#sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.8, nesterov=True)


		#sgd = optimizers.SGD(lr=0.000001, decay=0.0, momentum=0.8, nesterov=True)
		#inv_model.compile(optimizer='adadelta', loss='mean_squared_error')
		inv_model.compile(optimizer='adam', loss='mean_squared_error')
		#inv_model.compile(optimizer=sgd, loss='mean_squared_error')		
		print 'inverse model'
		inv_model.summary()
		return inv_model


	def build_inverse_code_model(self,  code_size, input_dim=2, max_pool_size = 2, conv_size = 3, channels=1):
		print 'building inverse code model...'

		input_code = Input(shape=(code_size,), name = 'inv_input')
		x = Dense(code_size, activation='tanh')(input_code)
		x = Dense(code_size*10, activation='tanh')(x)
		x = Dense(code_size*10, activation='tanh')(x)
		command = Dense(input_dim, name = 'command')(x)

		inv_model = Model(input_code, command)
		sgd = optimizers.SGD(lr=0.0014, decay=0.0, momentum=0.8, nesterov=True)
		inv_model.compile(optimizer=sgd, loss='mean_squared_error')
		print 'inverse code model'
		inv_model.summary()
		return inv_model

	def train_inverse_model(self, inverse_model, images, motor_cmd, test_img, test_pos,  batch_size=32, epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/inv', histogram_freq=0, write_graph=True, write_images=True)
		print 'Training inverse model'
		inverse_model.fit(images, motor_cmd, validation_data=[test_img, test_pos], epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[tensorboard_callback])
		inverse_model.save('./models/inverse_model.h5')
		print 'Inverse model trained and saved'

	def train_inverse_code_model(self, inverse_model, codes, motor_cmd, test_codes, test_pos,  batch_size=32, epochs = 1):
		tensorboard_callback = TensorBoard(log_dir='./logs/inv_code', histogram_freq=0, write_graph=True, write_images=True)
		print 'Training inverse model'
		inverse_model.fit(codes, motor_cmd, validation_data=[test_codes, test_pos], epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[tensorboard_callback])
		inverse_model.save('./models/inverse_code_model.h5')
		print 'Inverse code model trained and saved'

	def train_inverse_model_on_batch(self, inverse_model, images, motor_cmd, batch_size=1, epochs = 1):
		inverse_model.fit(images, motor_cmd, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)#, callbacks=[showLR()])
		#inverse_model.train_on_batch(images, motor_cmd)
		print 'Inverse model trained on batch'

	def train_inverse_code_model_on_batch(self, inverse_model, codes, motor_cmd, batch_size=1, epochs = 1):
		#tensorboard_callback = TensorBoard(log_dir='./logs/inv_code', histogram_freq=0, write_graph=True, write_images=True)
		inverse_model.fit(codes, motor_cmd, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)#, callbacks=[tensorboard_callback])#, callbacks=[showLR()])
		print 'Inverse code model trained on batch'

#class showLR(self, Callback ) :
#	def on_epoch_begin(self, epoch, logs=None):
#	    lr = float(K.get_value(self.model.optimizer.lr))
#	    print " epoch={:02d}, lr={:.5f}".format( epoch, lr )


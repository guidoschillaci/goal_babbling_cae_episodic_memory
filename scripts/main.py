#!/usr/bin/env python
from __future__ import print_function  

from minisom import MiniSom

from copy import deepcopy
import h5py
import cv2
from models import Models
from interest_model import InterestModel

from plots import plot_learning_progress, plot_log_goal_inv, plot_log_goal_fwd, plot_simple, plot_cvh, plot_learning_comparisons
from scipy.spatial import ConvexHull
from sklearn.decomposition import  IncrementalPCA
from sklearn.cluster import KMeans

# from cv_bridge import CvBridge, CvBridgeError
import random
import os
import shutil
import pickle
import gzip
import datetime
import numpy as np
import signal
import sys, getopt
from util import clamp_x, clamp_y, x_lims, y_lims, z_lims, speed_lim, Position
import threading
import random
from cam_sim import Cam_sim


import tensorflow as tf

if tf.__version__ < "1.8.0":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.30
    session = tf.Session(config=config)
else:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.30
    session = tf.compat.v1.Session(config=config)


models = Models()

class GoalBabbling():

	def __init__(self):

		# this simulates cameras and positions
		self.cam_sim = Cam_sim("../../rgb_rectified")

		self.lock = threading.Lock()
		signal.signal(signal.SIGINT, self.Exit_call)

	def initialise(self, goal_size = 3, image_size = 64, channels =1, batch_size = 16, goal_selection_mode = 'db', exp_iteration = 0, hist_size = 35, prob_update = 0.1):

		self.exp_iteration = exp_iteration

		self.goal_size= goal_size
		self.image_size = image_size
		self.channels = channels
		self.samples = []

		self.code_size=32

		self.iteration = 0
		self.test_data_step = 500
		self.test_size = 50
		self.max_iterations = 5000

		self.history_size = hist_size # how many batches (*batch_size) are kept?
		self.history_buffer_update_prob = prob_update # what is the probabilty that a data element in the history buffer is substituted with a new one

		self.pos = []
		self.cmd = []
		self.img = []
		self.history_pos = [] # kept to prevent catastrophic forgetting
		self.history_img = [] # kept to prevent catastrophic forgetting
		
		self.goal_code = []
		self.mse_inv = []
		self.mse_fwd = []
		self.mse_inv_goal = [] # on goals only
		self.mse_fwd_goal = [] # on goals only

		self.samples_pos=[]
		self.samples_img=[]
		self.samples_codes=[]
		self.test_positions=[]

		self.goal_selection_mode = goal_selection_mode # 'som' or 'kmeans'

		self.move = False

		self.autoencoder, self.encoder, self.decoder = None, None, None
		#self.forward_model = None
		self.forward_code_model = None
		#self.inverse_model = None
		self.inverse_code_model = None
		self.goal_som = None
		self.interest_model = InterestModel(self.goal_size)
		self.current_goal_x = -1
		self.current_goal_y = -1
		self.current_goal_idx = -1
		self.cae_epochs = 1
		self.inv_epochs = 2
		self.fwd_epochs = 2
		self.random_cmd_flag = False
		self.random_cmd_rate = 0.30
#		self.goal_db = ['./sample_images/img_0.jpg','./sample_images/img_200.jpg','./sample_images/img_400.jpg','./sample_images/img_600.jpg','./sample_images/img_800.jpg','./sample_images/img_1000.jpg','./sample_images/img_1200.jpg','./sample_images/img_1400.jpg','./sample_images/img_1600.jpg' ]
		self.goal_image = np.zeros((1, self.image_size, self.image_size, channels), np.float32)	
		self.batch_size=batch_size

		self.count = 1

		self.log_lp= [] # learning progress for each goal
		#self.log_lr_inv = [] # learning rate inverse model
		self.log_goal_pos = [] # ccurrent goal position (x,y xcarve position)
		for i in range(self.goal_size*self.goal_size):
			self.log_goal_pos.append([])
		self.log_goal_pred = [] # ccurrent xcarve position (x,y xcarve position)
		for i in range(self.goal_size*self.goal_size):
			self.log_goal_pred.append([])

		self.log_goal_img = [] # ccurrent goal position (x,y xcarve position)
		for i in range(self.goal_size*self.goal_size):
			self.log_goal_img.append([])

		self.log_curr_img = [] # ccurrent xcarve position (x,y xcarve position)
		for i in range(self.goal_size*self.goal_size):
			self.log_curr_img.append([])

		self.log_goal_id = []
		self.log_cvh_inv = []

		dataset = '../../rgb_rectified/compressed_dataset.pkl'
		print ('Loading test dataset ', dataset)
		self.train_images, self.test_images, self.train_cmds, self.test_cmds, self.train_pos, self.test_pos = models.load_data(dataset, self.image_size, step = self.test_data_step)
		# load models
		self.autoencoder, self.encoder, self.decoder = models.load_autoencoder(directory='./models/',code_size = self.code_size, image_size = self.image_size, batch_size=self.batch_size, epochs=self.cae_epochs)
#		self.forward_model = models.load_forward_model(train=False)
		self.forward_code_model = models.load_forward_code_model( code_size = self.code_size, train=False)
		#self.inverse_model = models.load_inverse_model(train=False)
		self.inverse_code_model = models.load_inverse_code_model(code_size = self.code_size, train=False)
		if self.goal_selection_mode =='kmeans':
			self.kmeans = models.load_kmeans()
		if self.goal_selection_mode =='som':
			self.goal_som = models.load_som(encoder = self.encoder, goal_size = self.goal_size)

		# initialise convex hulls (debug stuff)
		np.random.seed(10) # generate always the same random input to the convex hull
		pt_inv = []
		for i in range(3):
			a = np.random.rand(2)*0.01
			pt_inv.append(a)

		np.random.seed() # change the seed
		
		self.convex_hull_inv =ConvexHull(np.asarray(pt_inv), incremental=True)

		p = Position()
		p.x = int(0)
		p.y = int(0)
		p.z = int(-90)
		p.speed = int(1800)

		self.prev_pos=p
		#self.run_babbling()


	def get_current_inv_mse(self):
		#motor_pred = self.inverse_model.predict(self.test_images[0:self.test_size])# (self.goal_size*self.goal_size)])
		img_codes = self.encoder.predict(self.test_images[0:self.test_size])
		motor_pred = self.inverse_code_model.predict(img_codes)# (self.goal_size*self.goal_size)])
		#motor_pred = self.inverse_model.predict(self.test_images[0:self.test_size])# (self.goal_size*self.goal_size)])
#		mse = (np.linalg.norm(motor_pred-self.test_pos[0:(self.goal_size*self.goal_size)]) ** 2) / (self.goal_size*self.goal_size)
		mse = (np.linalg.norm(motor_pred-self.test_pos[0:self.test_size]) ** 2) / self.test_size
		print ('Current mse inverse code model: ', mse)
		self.mse_inv.append(mse)

#	def get_current_inv_mse_on_goals(self):
#		mse = 0
#		if self.goal_selection_mode =='db' or self.goal_selection_mode =='random':
#			img_codes = self.encoder.predict(self.test_images[0:(self.goal_size*self.goal_size)])
#			motor_pred = self.inverse_code_model.predict(img_codes)
#			mse = (np.linalg.norm(motor_pred-self.test_pos[0:(self.goal_size*self.goal_size)]) ** 2) / (self.goal_size*self.goal_size)
#		print 'Current mse inverse code model on goals: ', mse
#		self.mse_inv_goal.append(mse)


	def get_current_fwd_mse(self):
		#img_pred = self.forward_model.predict(self.test_pos)
		#img_pred = self.decoder.predict(self.forward_code_model.predict(self.test_pos))
		#img_obs_code = self.encoder.predict(self.test_images[0:(self.goal_size*self.goal_size)])
		img_obs_code = self.encoder.predict(self.test_images[0:self.test_size])
		#img_pred_code = self.encoder.predict(img_pred)
#		img_pred_code = self.forward_code_model.predict(self.test_pos[0:(self.goal_size*self.goal_size)])
		img_pred_code = self.forward_code_model.predict(self.test_pos[0:self.test_size])
		#mse = (np.linalg.norm(img_pred_code-img_obs_code) ** 2) /  (self.goal_size*self.goal_size)
		mse = (np.linalg.norm(img_pred_code-img_obs_code) ** 2) /  self.test_size
		print ('Current mse fwd model: ', mse)
		self.mse_fwd.append(mse)

	def run_babbling(self):
		p = Position()
			
		for _ in range(self.max_iterations):
			#test_motor_pred = self.inverse_model.predict([self.test_images[0:self.goal_size*self.goal_size]])
			#print np.hstack((test_motor_pred,self.test_pos[0:self.goal_size*self.goal_size]))

			# record logs and data
			self.get_current_inv_mse()
			self.get_current_fwd_mse()
			#self.log_lr_inv.append(K.eval(self.inverse_model.optimizer.lr))
			#print 'iteration opt inv', K.eval(self.inverse_model.optimizer.iteration)
			#print 'current lr: ', self.log_lr_inv[-1]
		
			print ('Iteration : ', self.iteration, ' goal_mode ', self.goal_selection_mode)
			self.iteration = self.iteration+1
#			if self.iteration > self.max_iterations:
#				self.save_models()
#				return

			# select a goal
			self.current_goal_idx, self.current_goal_x, self.current_goal_y = self.interest_model.select_goal()
			if self.goal_selection_mode =='kmeans':
				self.goal_code  = self.kmeans.cluster_centers_[self.current_goal_idx].reshape(1, self.code_size)
				print (' code ', self.goal_code)
			elif self.goal_selection_mode =='db' or self.goal_selection_mode =='random' :
				#self.goal_image=np.asarray(cv2.imread(self.goal_db[self.current_goal_idx], 0)).reshape(1, self.image_size, self.image_size, self.channels).astype('float32') / 255
				self.goal_image = self.test_images[self.current_goal_idx].reshape(1, self.image_size, self.image_size, self.channels)
				self.goal_code  = self.encoder.predict(self.goal_image)
			
			elif self.goal_selection_mode =='som':
				self.goal_code  = self.goal_som._weights[self.current_goal_x, self.current_goal_y].reshape(1, self.code_size)
			
			else:
				print ('wrong goal selection mode, exit!')
				sys.exit(1)

			motor_pred = []
			if self.goal_selection_mode == 'db':
				#cv2.imshow("Goal", cv2.imread(self.goal_db[self.current_goal_idx], 0))
				####cv2.imshow("Goal", self.test_images[self.current_goal_idx])
				####cv2.waitKey(1)
				#motor_pred = self.inverse_model.predict(self.goal_image)
				motor_pred = self.inverse_code_model.predict(self.goal_code)
				print ('pred ', motor_pred, ' real ', self.test_pos[self.current_goal_idx])
			else:
				goal_decoded = self.decoder.predict(self.goal_code)
				####cv2.imshow("CAE Decoded Goal", goal_decoded[0])
				####cv2.waitKey(1)
	#				motor_pred = self.inverse_model.predict(goal_decoded)
				motor_pred = self.inverse_code_model.predict(self.goal_code)
			#image_pred = self.forward_model.predict(np.asarray(motor_pred))
			image_pred = self.decoder.predict(self.forward_code_model.predict(np.asarray(motor_pred)))
			####cv2.imshow("FWD Model Predicted Image", image_pred[0])
			####cv2.waitKey(1)

			#image_pred_curr = forward_model.predict(np.asarray(prev_cmd))
			#cv2.imshow("FWD Model Predicted Image curr ", image_pred_curr[0])
			#cv2.waitKey(1) 
	#			motor_cmd=motor_pred[0]

	#			p.x = clamp_x(motor_pred[0][0]*x_lims[1])
	#			p.y = clamp_x(motor_pred[0][1]*y_lims[1])
			noise_x = np.random.normal(0,0.02)
			noise_y = np.random.normal(0,0.02)
			p.x = clamp_x((motor_pred[0][0]+noise_x)*x_lims[1])
			p.y = clamp_y((motor_pred[0][1]+noise_y)*y_lims[1])

			ran = random.random()
			if ran < self.random_cmd_rate or (len(self.history_pos) / self.batch_size) != self.history_size or self.goal_selection_mode =='random': # choose random motor commands from time to time
				print ('generating random motor command')
				p.x = random.uniform(x_lims[0], x_lims[1])
				p.y = random.uniform(y_lims[0], y_lims[1])
				self.random_cmd_flag = True
			else:
				self.random_cmd_flag=False

			print ('predicted position ', motor_pred[0], 'p+noise ', motor_pred[0][0]+noise_x, ' ' , motor_pred[0][1]+noise_y, ' clamped ', p.x, ' ' , p.y, ' noise.x ', noise_x, ' n.y ', noise_y)

			p.z = int(-90)
			p.speed = int(1400)
		
			self.create_simulated_data(p, self.prev_pos)
			self.prev_pos=p
		
			if self.iteration % 50 == 0:
				#test_codes= self.encoder.predict(self.test_images[0:self.goal_size*self.goal_size].reshape(self.goal_size*self.goal_size, self.image_size, self.image_size, self.channels))
				if self.goal_selection_mode == 'db' or self.goal_selection_mode == 'random':
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=self.test_pos[0:self.goal_size*self.goal_size], num_goals=self.goal_size*self.goal_size)
				elif self.goal_selection_mode == 'kmeans':
					goals_pos = self.inverse_code_model.predict(self.kmeans.cluster_centers_[0:self.goal_size*self.goal_size].reshape(self.goal_size*self.goal_size, self.code_size))
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=goals_pos, num_goals=self.goal_size*self.goal_size)
				elif self.goal_selection_mode == 'som':
					goals_pos = self.inverse_code_model.predict(self.goal_som._weights.reshape(len(self.goal_som._weights)*len(self.goal_som._weights[0]), len(self.goal_som._weights[0][0]) ))
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=goals_pos, num_goals=self.goal_size*self.goal_size)
				#test_codes_ipca = self.fwd_ipca.fit_transform(test_codes)
				#plot_cvh(self.convex_hull_fwd, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_fwd', iteration = self.iteration, dimensions=2, log_goal=test_codes_ipca, num_goals=self.goal_size*self.goal_size)

			observation = self.img[-1]
			####cv2.imshow("Current observation", observation)
			# update competence of the current goal (it is supposed that at this moment the action is finished
			if len(self.img)>0 and (not self.random_cmd_flag or self.goal_selection_mode == 'random'):

				#observation_code = self.encoder.predict(observation.reshape(1, self.image_size, self.image_size, self.channels))
				#prediction_error = np.linalg.norm(np.asarray(self.goal_code[:])-np.asarray(observation_code[:]))
				cmd = [p.x/float(x_lims[1]), p.y/float(y_lims[1])]
				prediction_code = self.forward_code_model.predict(np.asarray(cmd).reshape((1,2)))			

				prediction_error = np.linalg.norm(np.asarray(self.goal_code[:])-np.asarray(prediction_code[:]))
				self.interest_model.update_competences(self.current_goal_x, self.current_goal_y, prediction_error)
				#print 'Prediction error: ', prediction_error, ' learning progress: ', self.interest_model.get_learning_progress(self.current_goal_x, self.current_goal_y)
		
				self.log_lp.append(np.asarray(deepcopy(self.interest_model.learning_progress)))
				self.log_goal_id.append(self.current_goal_idx)

			
		
			#print self.log_lp
			# fit models	
			if len(self.img) > self.batch_size:
				if len(self.img) == len(self.pos):	
					# first fill the history buffer, then update the models
					if (len(self.history_pos) / self.batch_size) != self.history_size:
						if len(self.history_pos) ==0:
							self.history_pos = deepcopy(self.pos[-(self.batch_size):])
							self.history_img = deepcopy(self.img[-(self.batch_size):])
						self.history_pos= np.vstack((self.history_pos, self.pos[-(self.batch_size):] ))
						self.history_img= np.vstack((self.history_img, self.img[-(self.batch_size):] ))
					
					else:
						img_io = []
						pos_io = []
						if (self.history_size != 0):
							for i in range(-(self.batch_size), 0):
								#print i
								r = random.random()
								if r < self.history_buffer_update_prob:
									r_i = random.randint(0, self.history_size * self.batch_size - 1)
									#print 'r_i ', r_i, ' i ', i
									self.history_pos[r_i] = deepcopy(self.pos[i])
									self.history_img[r_i] = deepcopy(self.img[i])

							# update models
							img_io =  np.vstack(( np.asarray(self.history_img[:]).reshape( len(self.history_img), self.image_size, self.image_size, self.channels), np.asarray(self.img[-(self.batch_size):]).reshape(self.batch_size, self.image_size, self.image_size, self.channels) ))
							pos_io = np.vstack((np.asarray(self.history_pos[:]), np.asarray(self.pos[-(self.batch_size):]) ))
						else:
							img_io =  np.asarray(self.img[-(self.batch_size):]).reshape(self.batch_size, self.image_size, self.image_size, self.channels)
							pos_io =  np.asarray(self.pos[-(self.batch_size):])
						#models.train_forward_model_on_batch(self.forward_model, pos_io, img_io, batch_size=self.batch_size, epochs = self.epochs)
						codes_io = self.encoder.predict(img_io)
						models.train_forward_code_model_on_batch(self.forward_code_model, pos_io, codes_io, batch_size=self.batch_size, epochs = self.fwd_epochs)
						#models.train_inverse_model_on_batch(self.inverse_model, img_io, pos_io, batch_size=self.batch_size, epochs = self.inv_epochs)
						models.train_inverse_code_model_on_batch(self.inverse_code_model, codes_io, pos_io, batch_size=self.batch_size, epochs = self.inv_epochs)

						#train_autoencoder_on_batch(self.autoencoder, self.encoder, self.decoder, np.asarray(self.img[-32:]).reshape(32, self.image_size, self.image_size, self.channels), batch_size=self.batch_size, cae_epochs=5)

					# update convex hulls
					obs_codes= self.encoder.predict(np.asarray(self.img[-(self.batch_size):]).reshape(self.batch_size, self.image_size, self.image_size, self.channels))
			
					#print self.pos[-32:]
					#print np.asarray(self.pos[-32:]).reshape((32,2))
					self.convex_hull_inv.add_points(np.asarray(self.pos[-(self.batch_size):]).reshape(((self.batch_size),2)), restart=True)
					self.log_cvh_inv.append(self.convex_hull_inv.volume)
					#print 'conv hull inv. volume: ', self.convex_hull_inv.volume
					#print 'a', obs_codes[:]
					#self.fwd_ipca.partial_fit(obs_codes[:])
					#ipca_codes = self.fwd_ipca.transform(obs_codes[:])
					#print 'p', ipca_codes
					#print 'p', ipca_codes[:]
					#self.convex_hull_fwd.add_points(np.asarray(obs_codes[:]), restart=False)
					# this may be inconsistent. IPCA could change over time, so previous points projected for the CVH calculation could change. Check this
					#self.convex_hull_fwd.add_points(np.asarray(ipca_codes[:]), restart=True) 
					#print 'conv hull fwd. volume: ', self.convex_hull_fwd.volume
					#self.log_cvh_fwd.append(self.convex_hull_fwd.volume)

				else:
					print ('lenghts not equal')

			
			if not self.random_cmd_flag and len(self.cmd)>0:
				#print 'test pos', self.test_positions[0:10]
				test_p = self.test_pos[self.current_goal_idx]
				#curr_cmd = self.cmd[-1]
				#pred_p = self.inverse_model.predict(self.goal_image)
				pred_p = self.inverse_code_model.predict(self.goal_code)
				self.log_goal_pos[self.current_goal_idx].append([test_p[0],test_p[1] ])
				#self.log_curr_pos[self.current_goal_idx].append([curr_cmd[0], curr_cmd[1] ])
				self.log_goal_pred[self.current_goal_idx].append([pred_p[0][0], pred_p[0][1] ])
				#print 'hg ', self.log_goal_pos
				#g_code = self.forward_model.predict(np.asarray(test_p).reshape((1,2)))			
				#g_code = self.decoder.predict(self.forward_code_model.predict(np.asarray(test_p).reshape((1,2))))
				#g_code = self.fwd_ipca.transform(self.forward_code_model.predict(np.asarray(test_p).reshape((1,2))) )
				#c_code = self.forward_model.predict(np.asarray(curr_cmd).reshape((1,2)))
				#c_code = self.decoder.predict(self.forward_code_model.predict(np.asarray(curr_cmd).reshape((1,2))) ) 
				#c_code = self.fwd_ipca.transform(self.forward_code_model.predict(np.asarray(curr_cmd).reshape((1,2))) ) 
				#print 'g ipca ', g_code
				#print 'c ipca ', c_code
				#self.log_goal_img[self.current_goal_idx].append([g_code[0][0],g_code[0][1], g_code[0][2],g_code[0][3] ])
				#self.log_curr_img[self.current_goal_idx].append([c_code[0][0],c_code[0][1], c_code[0][2],c_code[0][3] ])
				#self.log_goal_img[self.current_goal_idx].append([g_code[0][0],g_code[0][1] ])
				#self.log_curr_img[self.current_goal_idx].append([c_code[0][0],c_code[0][1] ])

		print ('Saving models')
		self.save_models()

	def create_simulated_data(self, cmd, pos):
		self.lock.acquire()
		a = [int(pos.x), int(pos.y)]
		b = [int(cmd.x),int(cmd.y)]

#		b[0] = int(cmd.x)
#       b[1] = int(cmd.y)
		tr = self.cam_sim.get_trajectory(a,b)
		trn = self.cam_sim.get_trajectory_names(a,b)
		#a[0] = b[0]
		#a[1] = b[1]

        
		rounded  = self.cam_sim.round2mul(tr,5) # only images every 5mm
#		print 'trajectory ' 
		for i in range(len(tr)): 
#			print 'curr_dir ' , os.getcwd()
#			print(tr[i],rounded[i],trn[i])
			self.pos.append([float(rounded[i][0]) / x_lims[1], float(rounded[i][1]) / y_lims[1]] )
			self.cmd.append([float(int(cmd.x)) / x_lims[1], float(int(cmd.y)) / y_lims[1]] )
			cv2_img = cv2.imread(trn[i],1 )
			if self.channels ==1:
				cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
			cv2_img = cv2.resize(cv2_img,(self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR)
			cv2_img = cv2_img.astype('float32') / 255
			cv2_img.reshape(1, self.image_size, self.image_size, self.channels)		
			self.img.append(cv2_img)

		self.lock.release()

	def ats_callback(self, image, cmd, pos):
		self.lock.acquire()
		self.msg_cmd = cmd
		self.msg_pos = pos
		self.msg_image = image

		self.pos.append([float(self.msg_pos.x) / x_lims[1], float(self.msg_pos.y) / y_lims[1]] )

		self.cmd.append([float(self.msg_cmd.x) / x_lims[1], float(self.msg_cmd.y) / y_lims[1]] )
		self.samples.append({'image': self.msg_image, 'position':self.msg_pos, 'command':self.msg_cmd})

		cv2_img = bridge.imgmsg_to_cv2(self.msg_image, "bgr8")
		if self.channels ==1:
			cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
		cv2_img = cv2_img.astype('float32') / 255
		cv2_img.reshape(1, self.image_size, self.image_size, self.channels)		
		self.img.append(cv2_img)

		#img_list = []
		#for i in range(self.batch_size):
		#	img_list.append(self.img[-1].reshape(self.image_size, self.image_size, self.channels))
		#print np.asarray(self.img[-1]).reshape(1, self.image_size, self.image_size, self.channels).shape
		#img_code = self.encoder.predict(np.asarray(self.img[-1]).reshape(1, self.image_size, self.image_size, self.channels)	)
		#print np.asarray(img_list).shape
		#img_code = self.encoder.predict(np.asarray(img_list))
		#error = np.fabs(np.linalg.norm(np.asarray(self.goal_code)-np.asarray(img_code[0])))
		#self.log_distance_to_goal.append(error)
		#error=0
		#print 'error ', error
		self.lock.release()


	#		if len(images) ==batches+1:
	#			print 'Updating fwd model'
				#print np.asarray(prev_poss).shape, " ", np.asarray(prev_cmds).shape

	#			train_forward_model_on_batch(forward_model, np.hstack((np.asarray(prev_pos), np.asarray(prev_cmd))) , np.asarray(images), batchttps://stackoverflow.com/questions/961632/converting-integer-to-string-in-pythonh_size=10)

	#			print 'Updating inv model'
	#			train_inverse_model_on_batch(inverse_model, encoder, np.asarray(prev_pos), np.asarray(images), np.asarray(prev_cmd), batch_size=10)
	#			prev_pos =[]#
	#			prev_cmd =[]
	#			images = []
				#current_pos = []


	def save_models(self):



		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%X')
		name = 'goal_babbling'
		filename = './data/' + name + '_' + timestamp + '.pkl'
		#with gzip.open(filename, 'wb') as file:
		#	cPickle.dump(self.samples, file, protocol=2)
		####cv2.destroyAllWindows()
		#print 'Dataset saved to ', filename

		self.autoencoder.save('./models/autoencoder.h5', overwrite=True)
		self.encoder.save('./models/encoder.h5', overwrite=True)
		self.decoder.save('./models/decoder.h5', overwrite=True)
#		self.inverse_model.save('./models/inverse_model.h5')
		self.inverse_code_model.save('./models/inverse_code_model.h5', overwrite=True)
		#self.forward_model.save('./models/gb/forward_model.h5')
		self.forward_code_model.save('./models/forward_code_model.h5', overwrite=True)

		#trained_som_weights = self.goal_som.get_weights().copy()
		#som_file = h5py.File('./models/goal_som.h5', 'w')
		#som_file.create_dataset('goal_som', data=trained_som_weights)
		#som_file.close()

		#np.savetxt('./models/log_goal_pos.txt', self.log_goal_pos)
		pickle.dump(self.log_goal_pos, open('./models/log_goal_pos.txt', 'wb'))
		#go_file = open('./models/log_goal_pos.txt','w')
		#go_file.write(str(self.log_goal_pos))
		#go_file.close()

		#np.savetxt('./models/log_curr_pos.txt', self.log_curr_pos)
#		cPickle.dump(self.log_curr_pos, open('./models/log_curr_pos.txt', 'wb'))
		pickle.dump(self.log_goal_pred, open('./models/log_goal_pred.txt', 'wb'))
		#cu_file = open('./models/log_curr_pos.txt','w')
		#cu_file.write(str(self.log_curr_pos))
		#cu_file.close()

		np.savetxt('./models/log_learning_progress.txt', self.log_lp)
		#lp_file = open('./models/log_learning_progress.txt','w')
		#lp_file.write(str(self.log_lp))
		#lp_file.close()

		np.savetxt('./models/log_goal_id.txt', self.log_goal_id)
		#gi_file = open('./models/log_goal_id.txt','w')
		#gi_file.write(str(self.log_goal_id))
		#gi_file.close()

		#np.savetxt('./models/log_distance_to_goal.txt', self.log_distance_to_goal)
		#dg_file = open('./models/log_distance_to_goal.txt','w')
		#dg_file.write(str(self.log_distance_to_goal))
		#dg_file.close()


		np.savetxt('./models/log_mse_inv.txt', self.mse_inv)
		plot_simple(self.mse_inv, 'MSE INV', save = True, show = False)

		np.savetxt('./models/log_mse_fwd.txt', self.mse_fwd)
		plot_simple(self.mse_fwd, 'MSE FWD', save = True, show = False)

		np.savetxt('./models/log_cvh_inv.txt', self.log_cvh_inv)
		plot_simple(self.log_cvh_inv, 'CVH Inv Volume', save = True, show = False)

		#np.savetxt('./models/log_cvh_fwd.txt', self.log_cvh_fwd)
		#plot_simple(self.log_cvh_fwd, 'CVH Fwd Volume', save = True, show = False)

		#np.savetxt('./models/log_lr_inv.txt', self.log_lr_inv)
		#plot_simple(self.log_lr_inv, 'LR_inv')

		plot_learning_progress(self.log_lp, self.log_goal_id, num_goals = self.goal_size*self.goal_size, save = True, show = False)
#		plot_log_goal_inv(self.log_goal_pos, self.log_curr_pos, num_goals = self.goal_size*self.goal_size, save = True, show = False)
		plot_log_goal_inv(self.log_goal_pos, self.log_goal_pred, num_goals = self.goal_size*self.goal_size, save = True, show = False)
		#plot_log_goal_fwd(self.log_goal_img, self.log_curr_img, num_goals = self.goal_size*self.goal_size, save = True, show = False)

		self.clear_session()

		print ('Models saved')
		self.goto_starting_pos()

	def clear_session(self):
		print ('Clearning variables')
		#del self.log_goal_pred
		#del self.log_curr_img
		#del self.log_cvh_inv
		#del self.log_goal_id
		#del self.log_goal_img
		#del self.log_goal_pos
		#del self.log_lp
		#del self.samples
		#del self.samples_img
		#del self.samples_codes
		#del self.samples_pos
		#del self.autoencoder
		#del self.encoder
		#del self.decoder
		#del self.inverse_code_model
		#del self.forward_code_model
		#del self.mse_inv
		#del self.mse_fwd
		#del self.history_img
		#del self.history_pos

		# reset
		print('Clearing TF session')
		if tf.__version__ < "1.8.0":
			tf.reset_default_graph()
		else:
			tf.compat.v1.reset_default_graph()

	def Exit_call(self, signal, frame):
		self.goto_starting_pos()
		self.save_models()

	def goto_starting_pos(self):
		p = Position()
		p.x = int(0)
		p.y = int(0)
		p.z = int(-50)
		p.speed = int(1400)
		
		self.create_simulated_data(p, self.prev_pos)
		self.prev_pos=p

if __name__ == '__main__':

	# reset
	print('Clearing TF session')
	if tf.__version__ < "1.8.0":
		tf.reset_default_graph()
	else:
		tf.compat.v1.reset_default_graph()

	goal_babbling = GoalBabbling()
	os.chdir('experiments')
	exp_iteration_size = 5
	exp_type = ['db', 'random', 'som']#, 'kmeans']
	history_size = [0, 10, 20]
	prob = [0.1, 0.01]

	for e in range(len(exp_type)):
		print ('exp ', exp_type[e])

		for h in range( len (history_size)):
			print('history size ', history_size[h])

			for p in range(len(prob)):
				print('prob update ', prob[p])

				for i in range(exp_iteration_size):
					print( 'exp ', exp_type[e], ' history size ', str(history_size[h]), ' prob ', str(prob[p]), ' iteration ', str(i) )
					directory = './'+exp_type[e]+'_'+str(history_size[h])+'_'+str(prob[p])+'_'+str(i)+'/'
					if not os.path.exists(directory):
						os.makedirs(directory)

						if not os.path.exists(directory+'models'):
							os.makedirs(directory+'models')

						shutil.copy('../pretrained_models/autoencoder.h5', directory+'models/autoencoder.h5')
						shutil.copy('../pretrained_models/encoder.h5', directory+'models/encoder.h5')
						shutil.copy('../pretrained_models/decoder.h5', directory+'models/decoder.h5')
						shutil.copy('../pretrained_models/goal_som.h5', directory+'models/goal_som.h5')
						shutil.copy('../pretrained_models/kmeans.sav', directory+'models/kmeans.sav')

						os.chdir(directory)
						if not os.path.exists('./models/plots'):
							os.makedirs('./models/plots')
						if not os.path.exists('./data'):
							os.makedirs('./data')
						print ('current directory: ', os.getcwd())

						goal_babbling.initialise( goal_selection_mode= exp_type[e], exp_iteration = i, hist_size= history_size[h], prob_update=prob[p])
						goal_babbling.run_babbling()
						os.chdir('../')
						#GoalBabbling().
						print ('finished experiment ', exp_type[e], ' history size ', str(history_size[h]),' prob ', str(prob[p]), ' iter ', str(i))

						goal_babbling.clear_session()
					print ('experiment ', directory, ' already carried out')
	os.chdir('../')
	plot_learning_comparisons(model_type = 'fwd', exp_size = exp_iteration_size, save = True, show = True)
	plot_learning_comparisons(model_type = 'inv', exp_size = exp_iteration_size, save = True, show = True)


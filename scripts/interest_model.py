# author Guido Schillaci, Dr.rer.nat. - Scuola Superiore Sant'Anna
# Guido Schillaci <guido.schillaci@santannapisa.it>
import numpy as np
import random

class InterestModel():

	def __init__(self, goals_size, competence_measure = 'euclidean', decay_factor = 0.9):
		self.learning_progress = 0.0 # interest factor for each goal
		self.pe_history = [] # history of prediction errors for each goal
		# self.pe_derivatives = [] # derivative of the prediction errors for each goal
		self.competence_measure = competence_measure
		self.goals_size = goals_size
		self.initialise_im()
		self.decay_factor = decay_factor

	def initialise_im(self):
		self.learning_progress= np.resize(self.learning_progress, (self.goals_size*self.goals_size ,))
		for i in range (0, self.goals_size*self.goals_size):
			self.pe_history.append([0.0])
			#self.pe_derivatives.append([0.0])

	def select_goal(self):
		ran = random.random()
		goal_idx = 0
		if ran < 0.85 and np.sum(self.learning_progress)>0: # 70% of the times
			# select goal with highest interest factor			
			goal_idx = self.get_most_interesting_goal_index()
			print 'Selected most interesting goal.'
		else:
			# select random goal
			goal_idx = random.randint(0, self.goals_size*self.goals_size-1)
			print 'Selected random goal.'

		goal_x = goal_idx / self.goals_size  
		goal_y = goal_idx % self.goals_size
		print 'Goal idx: ', goal_idx, ' x: ', goal_x, ' y: ', goal_y
		return goal_idx, goal_x, goal_y

	def update_competences(self, goal_id_x, goal_id_y, prediction_error):

		# decay all
		self.learning_progress = self.decay_factor * self.learning_progress

		goal_id = goal_id_x * self.goals_size + goal_id_y
		# self.interest_factors -= 0.05 # decrease all
		#low_vals = self.learning_progress < 0
		#self.learning_progress[low_vals] = 0
#		self.pe_history[goal_id].append(np.power(prediction_error,2))
		self.pe_history[goal_id].append(prediction_error)
		ped = 0
		if len(self.pe_history[goal_id]) >= 2:

			if self.competence_measure == "euclidean":
				ped = prediction_error-self.pe_history[goal_id][-2] 
				#self.pe_derivatives[goal_id].append(ped)
				self.learning_progress[goal_id] = np.tanh( np.fabs(ped)*2 )

			elif self.competence_measure == "exponential":
				# Measure from: C. Moulin-Frier, S. M. Nguyen, and P.-Y. Oudeyer. Self-organization of early vocal development in infants and machines: the role of intrinsic motivation. Frontiers in psychology, 4,2013.
				ped = prediction_error-self.pe_history[goal_id][-2]
				self.learning_progress[goal_id] = np.exp( - np.fabs (ped) )

			else:
				print 'wrong competence measure! Specify either euclidean or exponential'

			print 'Learning progress of goal [0-1] ', goal_id, ', idx: ', goal_id_x, ' idy:', goal_id_y, ' is: ', self.learning_progress[goal_id], ' ped ', ped, ' PE: ', prediction_error

			# interest_factors = interest_factors / np.sum(interest_factors)
			# threshold interest factor
			if self.learning_progress[goal_id] < 0.01:
				self.learning_progress[goal_id] = 0.0
				print 'Learning process on goal', goal_id, ' set to 0'
#		sum_lp = np.sum(self.learning_progress)
#		if sum_lp !=0:
#			self.learning_progress = self.learning_progress / sum_lp

	def get_learning_progress(self, x, y):
		return self.learning_progress[x * self.goals_size + y]

	def get_all_progresses(self):
		return self.learning_progress

	# get most interesting goal index
	def get_most_interesting_goal_index(self):
		return np.argmax(self.learning_progress)


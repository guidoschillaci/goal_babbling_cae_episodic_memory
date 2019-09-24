# Authors:
# Antonio Pico Villalpando, Humboldt-Universitaet zu Berlin

import sys
import os
import cv2
import numpy as np
import re
#from goal_babbling.msg import Position
import gzip 
import pickle
from util import Position
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError

class Cam_sim():
	def __init__(self,imagesPath):
		self.imagesPath = imagesPath
		if self.imagesPath[-1] != '/':
			self.imagesPath += '/'

	def round2mul(self,number, multiple):
		half_mult = multiple/2.
		result = np.floor( (number + half_mult) / multiple  ) * multiple
		return result.astype(np.int)
	
	def get_trajectory(self,start,end):
		trajectory =  np.array(self.get_line(start, end))
		return trajectory

	def get_trajectory_names(self,start,end):
		trajectory = self.get_trajectory(start,end)
		t_rounded = self.round2mul(trajectory,5) #there is only images every 5 mm, use closer image to real coordinate
		t_images = []
		for i in t_rounded:
			img_name = self.imagesPath + "x{:03d}_y{:03d}.jpeg".format(i[0],i[1])
			t_images.append(img_name)
		return t_images

	def get_line(self,start, end):
		"""Bresenham's Line Algorithm
		Produces a list of tuples from start and end

		>>> points1 = get_line((0, 0), (3, 4))
		>>> points2 = get_line((3, 4), (0, 0))
		>>> assert(set(points1) == set(points2))
		>>> print points1
		[(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
		>>> print points2
		[(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
		"""
		# Setup initial conditions
		x1, y1 = start
		x2, y2 = end
		dx = x2 - x1
		dy = y2 - y1

		# Determine how steep the line is
		is_steep = abs(dy) > abs(dx)

		# Rotate line
		if is_steep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2

		# Swap start and end points if necessary and store swap state
		swapped = False
		if x1 > x2:
			x1, x2 = x2, x1
			y1, y2 = y2, y1
			swapped = True

		# Recalculate differentials
		dx = x2 - x1
		dy = y2 - y1

		# Calculate error
		error = int(dx / 2.0)
		ystep = 1 if y1 < y2 else -1

		# Iterate over bounding box generating points between start and end
		y = y1
		points = []
		for x in range(x1, x2 + 1):
			coord = (y, x) if is_steep else (x, y)
			points.append(coord)
			error -= abs(dy)
			if error < 0:
				y += ystep
				error += dx

		# Reverse the list if the coordinates were swapped
		if swapped:
			points.reverse()
		return points




def extract_images(file_name):
	with gzip.open(file_name, 'rb') as memory_file:
		memories = pickle.load(memory_file)
		print ('extracting images...')
		count = 0
		for memory in memories:
			image = memory['image']
			#image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

			cmd = memory['position']
			title = './rgb_rectified/x'+str(cmd.x)+'_y'+str(cmd.y)+'.jpeg'
			cv2.imwrite(title,image)

if __name__ == '__main__':

	path="./rgb_rectified"

	compressed_dataset = path+'/compressed_dataset.pkl'
	if os.path.isfile(compressed_dataset):
		print ('compressed dataset already exists')
		extract_images(compressed_dataset)

	else:
		print ('creating compressed dataset')

		samples = []
		counter=0
		for file in os.listdir(path):
			filename_full = os.path.basename(file)
			filename = os.path.splitext(filename_full)[0]
			splitted = re.split('x|_|y', filename)
			p = Position()
			p.x=splitted[1]
			p.y=splitted[3]
			p.z=-90
			p.speed=1400
			#print path+'/'+os.path.relpath(file)
			cv_img = cv2.imread(path+'/'+os.path.relpath(file))
			cv_img = cv2.resize(cv_img, (64, 64))
			#image_msg= bridge.cv2_to_imgmsg(cv_img, "bgr8")
			#samples.append({'image': image_msg, 'position':p, 'command':p})
			samples.append({'image': cv_img, 'position': p, 'command': p})
			#print int(p.x), ' ', int(p.y)
			counter=counter+1
			print (counter)
		with gzip.open(compressed_dataset, 'wb') as file:
			pickle.dump(samples, file, protocol=2)
		print ('saved')
		sys.exit(1)


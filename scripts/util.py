
import gzip
import pickle
import cv2
import re
import numpy as np


x_lims=[0.0,750.0]
y_lims=[0.0,750.0]
z_lims=[-10.0,-90.0]
speed_lim = 3400.0

def clamp_x(x):
	if x <= x_lims[0]:
		return x_lims[0]
	if x > x_lims[1]:
		return x_lims[1]
	return x

def clamp_y(y):
	if y <= y_lims[0]:
		return y_lims[0]
	if y > y_lims[1]:
		return y_lims[1]
	return y

def clamp_z(z):
	if z <= z_lims[0]:
		return z_lims[0]
	if z > z_lims[1]:
		return z_lims[1]
	return z


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.speed = 0
import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('cleandata_students.mat')
mat_contents1 = sio.loadmat('noisydata_students.mat')

noisy_data = mat_contents['x']
noisy_labels = mat_contents['y']

training_data = mat_contents['x']
labels = mat_contents['y']

def convert_to_binary(labels, target_number):

	""" Takes as input the labels vector (Nx1) and converts this vector
		to a binary one having 1 in the target_number(emotion) and 0's
		everywhere else"""

	target_emotion = target_number
	N = len(labels)
	binary_vec = np.zeros(N)

	for i in range(N):
		if labels[i] == target_emotion:
			binary_vec[i] = 1

	return binary_vec

#convert_to_binary(labels,4)

def get_training_data():
	return 	training_data

def get_labels():
	return 	labels

def get_noisy_data():
	return noisy_data

def get_noisy_labels():
	return noisy_labels

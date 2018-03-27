import pickle
import main
import data_import
import sys
import scipy.io as sio
import numpy as np 

if len(sys.argv) != 2 :
	print("Wrong number of arguments! Please run: \n python testTrees.py <name_of_input>.mat")
else:	
	data = sys.argv[1]

	mat_contents = sio.loadmat(data)

	testing_data = mat_contents['x']
	output = mat_contents['y']
	#reduced_test_data = testing_data[:10]

	T = pickle.load(open("trained_trees.p", "r"))

	pred, depth_counter = main.testTrees(T,testing_data)
	predicted_emotions = main.binary_to_emotions(pred, depth_counter)

	output = output.tolist()
	flattened_output = []
	for x in output:
	    for y in x:
	        flattened_output.append(y)

	correct_predictions = 0
	for i in range(len(testing_data)):
		if predicted_emotions[i] == flattened_output[i]:
			correct_predictions = correct_predictions + 1

	total_predictions  = len(testing_data)

	classification_rate = float(correct_predictions)/float(total_predictions)
	#print flattened_output
	print predicted_emotions
	print str(classification_rate*100) + '%'


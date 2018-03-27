from __future__ import division
import tree
import evaluation
import data_import
import numpy as np
import pandas as pd
import copy
import pickle
from PIL import Image,ImageDraw
import draw_tree

tree_size = 0

#load data
examples = data_import.get_training_data()
labels = data_import.get_labels()
k_fold = False

noisy_examples = data_import.get_noisy_data()
noisy_labels = data_import.get_noisy_labels()

#Flag to indicate which dataset to be used
use_clean = False

#data after removing attribute with specific class
def get_new_data(examples,binary_targets,attribute,_class):
    examples=np.array(examples)
    if examples != []:
        examples_for_search = examples[:,attribute]
        indices = [i for i,x in enumerate(examples_for_search) if x==_class]
        new_examples = [examples[x,:] for x in indices]
        if new_examples != []:
            new_examples = np.vstack(new_examples)
            new_binary_targets = [binary_targets[x] for x in indices]
        else:
            new_examples = []
            new_binary_targets = []

    else:
        new_examples = []
        new_binary_targets = []

    return new_examples, new_binary_targets

#training algorithm
def decision_tree_learning(examples,attrs,binary_targets):
    new_tree = tree.Tree(examples,attrs,binary_targets)
    if new_tree.compare_targets():
        new_tree.class_  = int(binary_targets[0])
    elif new_tree.attr_is_empty():
        new_tree.class_  = int(new_tree.majority_value())
    else:
        best_attribute, i_gain = new_tree.choose_best_attribute()
        new_tree.op = best_attribute
        new_tree.info_gain = i_gain
        new_attrs = copy.deepcopy(attrs)
        new_attrs[best_attribute] = 0
        for i in range(2):
            new_examples, new_binary_targets = get_new_data(examples, binary_targets, best_attribute, i)

            if len(new_examples) == 0:
                leaf_node = tree.Tree(1,1,1)
                leaf_node.class_  = int(new_tree.majority_value())
                new_tree.kids.append(leaf_node)
            else:
                new_tree.kids.append(decision_tree_learning(new_examples,new_attrs,new_binary_targets))
    return new_tree

#testing algorithm (use for predictions of many trees)
def testTrees(T,x2):
    # T -> the 6 trained trees
    # x2 -> Nx45 test data
    # output -> Nx6 predicted labels
    N = x2.shape[0]
    predictions = np.zeros((N,6))
    gain_counter = np.zeros((N,6))
    for n_tree in range(6):
        for i in range(N):
            g_counter = 0
            testing_tree = T[n_tree]
            test_example = x2[i,:]
            while(testing_tree.op != None):
                g_counter = g_counter + testing_tree.info_gain
                attr_value = test_example[testing_tree.op]
                testing_tree = testing_tree.kids[attr_value]
            predictions[i,n_tree] = testing_tree.class_
            gain_counter[i,n_tree] = g_counter
    return predictions, gain_counter


#convert the binary values to emotions 1-6
def binary_to_emotions(predictions, depth_counter):
    pred_emotions = []
    for i in range(len(predictions)):
        found = False
        #array which stores the emotions (1-6) that were
        #identified for each example. May be only one
        #emotion (most cases) or more, per example.
        emotions_found = []
        for y in range(6):
            if predictions[i,y] == 1:
                emotions_found.append(y + 1)
                #pred_emotions.append(y+1)
                found = True
        if not found:
            #all six values are equal to '0'
            max_index = -1
            max_value = -1
            for y in range(6):
                if depth_counter[i, y] > max_value:
                    max_value = depth_counter[i, y]
                    max_index = y
            pred_emotions.append(max_index + 1)
            #pred_emotions.append(0)
        else:
            #there is exactly one '1' (the ideal situation)
            if len(emotions_found) == 1:
                pred_emotions.append(emotions_found[0])
            else:
                #start with a low value
                max_index = -1
                max_value = -1
                #more than one '1' exist in a row since the algorithm
                #has reached these lines
                for y in range(len(emotions_found)):
                    next_index = emotions_found[y] - 1
                    next_value = depth_counter[i,next_index]
                    if next_value > max_value:
                        max_value = next_value
                        max_index = y
                pred_emotions.append(emotions_found[max_index])

    return pred_emotions

#train and evaluate using corss validation
#split examples and labels in 10 parts, every time use 1 part for test, 9 for train
#repeat the following process 10 times
#for each emotion, train a tree and append to list of trained trees
#when all 6 complete, test the performance of the training with all the evaluation metr.
def train_with_cross_validation(examples,labels):
	k_rates = []
	k_fold_examples = evaluation.split_to_k(examples,10)
	k_fold_targets = evaluation.split_to_k(labels,10)

	ones = np.ones(45)

	list_of_e = [1,2,3,4,5,6]
	sum_of_confusions = pd.DataFrame(0,index = list_of_e, columns = list_of_e)

	recalls = pd.DataFrame(0,index=range(10), columns=range(1,7))
	precisions = pd.DataFrame(0,index=range(10), columns=range(1,7))

	for k in range(10): #Repeat the following 10 times
		T = []
		test_examples = k_fold_examples[k]
		train_examples = []
		for k_1 in range(10): #concat k fold examples and leave out sublist k
				if k_1 != k:
		    			for y in k_fold_examples[k_1]:
       	 					train_examples.append(y)
		for e in range(6): #Do this for 6 trees
			binary_targets = data_import.convert_to_binary(labels,e+1)
			k_fold_labels = evaluation.split_to_k(binary_targets,10)
			test_labels = k_fold_labels[k]
			train_labels = []
			for k_2 in range(10): #concat k fold labels and leave sublist k
				if k_2 != k:
		    			for y in k_fold_labels[k_2]:
       		 				train_labels.append(y)
			ktree = decision_tree_learning(train_examples,ones,train_labels)
			T.append(ktree)

		#When all trees are created obtain the metrics
		pred, depth_counter = testTrees(T,test_examples)
		predicted_emotions = binary_to_emotions(pred, depth_counter)
		actual_emotions = k_fold_targets[k]
		rate = evaluation.classification_rate(actual_emotions,predicted_emotions)
		k_rates.append(rate)

		confusion = evaluation.confusion_matrix(actual_emotions,predicted_emotions)
		sum_of_confusions += confusion

		#Precision and recall for every emotion
		recalls.iloc[k,0] = evaluation.recall(1,confusion)
		recalls.iloc[k,1] = evaluation.recall(2,confusion)
		recalls.iloc[k,2] = evaluation.recall(3,confusion)
		recalls.iloc[k,3] = evaluation.recall(4,confusion)
		recalls.iloc[k,4] = evaluation.recall(5,confusion)
		recalls.iloc[k,5] = evaluation.recall(6,confusion)

		precisions.iloc[k,0] = evaluation.precision(1,confusion)
		precisions.iloc[k,1] = evaluation.precision(2,confusion)
		precisions.iloc[k,2] = evaluation.precision(3,confusion)
		precisions.iloc[k,3] = evaluation.precision(4,confusion)
		precisions.iloc[k,4] = evaluation.precision(5,confusion)
		precisions.iloc[k,5] = evaluation.precision(6,confusion)
	return k_rates, sum_of_confusions, recalls, precisions

def train_on_whole_dataset(examples,labels):
    T = []
    for i in range(6):
        binary_targets = data_import.convert_to_binary(labels,i+1)
        tree = decision_tree_learning(examples,np.ones(45),binary_targets)
        T.append(tree)
    return T

def draw_trees(trees):
	count =1
	for trained_tree in trees:

		w=draw_tree.getwidth(trained_tree)*100
		h=draw_tree.getdepth(trained_tree)*100+120

		img=Image.new('RGB',(w,h),(255,255,255))
		draw_img=ImageDraw.Draw(img)

		draw_tree.drawnode(draw_img,trained_tree,w/2,20)
		img.save('tree'+str(count)+'.jpg','JPEG')
		count+=1

def start_process():
    #Train trees with cross validation
    if (k_fold):
    	#Train and evaluate trees with cross validation, obtain rate, conf matrix, prec and rec
    	CR, Confusion, Precisions, Recalls = train_with_cross_validation(examples,labels)

    	#classification rate average of 10 fols
    	average_classification_rate = np.round(sum(CR)/float(len(CR)),2)
    	print 'Average Classification Rate: ', average_classification_rate
    	print

    	print 'Confusion Matrix'
    	print Confusion
    	print

    	#precision per emotion
    	precision = []
    	for i in range(6):
    		precision.append(np.round(Precisions.iloc[:,i].mean(),2))

    	print 'Precision'
    	print precision
    	print

    	#recall per emotion
    	recall = []
    	for i in range(6):
    		recall.append(np.round(Recalls.iloc[:,i].mean(),2))

    	print 'Recall'
    	print recall
    	print

    	#F measure for every emotion
    	f1_1 = np.round(evaluation.fa(recall[0], precision[0], 1),2)
    	f1_2 = np.round(evaluation.fa(recall[1], precision[1], 1),2)
    	f1_3 = np.round(evaluation.fa(recall[2], precision[2], 1),2)
    	f1_4 = np.round(evaluation.fa(recall[3], precision[3], 1),2)
    	f1_5 = np.round(evaluation.fa(recall[4], precision[4], 1),2)
    	f1_6 = np.round(evaluation.fa(recall[5], precision[5], 1),2)
    	print 'f metric for every emotion:'
    	print f1_1, ' ', f1_2, ' ', f1_3, ' ', f1_4, ' ', f1_5, ' ', f1_6
    	print
    else:
        T = train_on_whole_dataset(examples,labels)
        draw_trees(T)
    	pickle.dump(T, open("trained_trees.p", "w"))

#start_process()

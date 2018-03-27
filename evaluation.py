from __future__ import division
import pandas as pd
import numpy as np

def confusion_matrix(y_a_all, y_p_all):
    emotions = [1,2,3,4,5,6]
    matrix = pd.DataFrame(0,index = emotions, columns = emotions)
    for i in emotions:
        for j in emotions:
            for ya,yb in zip(y_a_all,y_p_all):
                if ya == i and yb == j:
                    matrix.iloc[i-1,j-1] += 1
    return matrix

def recall(emotion,matrix):
    tp = matrix.iloc[emotion-1,emotion-1]
    ap = sum(matrix.iloc[emotion-1,:])
    return tp/ap

def precision(emotion,matrix):
    tp = matrix.iloc[emotion-1,emotion-1]
    pp = sum(matrix.iloc[:,emotion-1])
    return tp/pp

def fa(recall,precision,weight):
    return (1+weight**2)*(precision*recall)/((weight**2*precision)+recall)

def classification_rate(y1,y2):
    count = 0
    for x1,x2 in zip(y1,y2):
        if x1==x2:
            count += 1
    rate = count/len(y1)
    return rate

def split_to_k(X,k):
    size_of_all = len(X)
    size_of_k = int(size_of_all/k)
    remainder = size_of_all%size_of_k
    X_split = []
    for i in range(0,k):
        if i!=k-1:
            X_split.append(X[size_of_k*i:size_of_k*(i+1)])
        else:
            X_split.append(X[size_of_k*i:])
    return X_split


#def fa_metric(y_actual, y_pred, weight):
#	tp = tn = fp = fn = 0
#	for ya,yp in zip(y_actual,y_pred):
#		if(yp==1 and ya==yp):
#			tp += 1
#		elif(yp==0 and ya==yp):
#			tn += 1
#		elif(yp==1 and ya!=yp):
#			fp += 1
#		elif(yp==0 and ya!=yp):
#			fn += 1
#	recall = tp/(tp+fn)
#	precision = tp/(tp+fp)
#	f = (1+weight)*(precision*recall)/((weight*precision)+recall)
#	return f

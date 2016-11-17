# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:01:21 2016

@author: alisazhila
"""

import numpy as np

ETA = 0.01 
EPSILON = 0.01



class Line:
    'Class to represent a line in 2D'
    a = 0
    b = 0

    def __init__(self, a, b):
       self.a = a
       self.b = b


def generate_line():
    #np.random.seed(323)    
    point1 = np.random.uniform(-1, 1, size=2)
    point2 = np.random.uniform(-1, 1, size=2)
    
    a = (point1[1]-point2[1])/(point1[0]-point2[0])
    b = point1[1] - a*point1[0]
    
    line = Line(a, b)
    return line 
    
    
def generate_dataset(N): 
    #np.random.seed(323)
    dataset = []
    x1_set = np.random.uniform(-1, 1, size=N)
    x2_set = np.random.uniform(-1, 1, size=N)
    for i in range(N): 
        dataset.append([x1_set[i], x2_set[i]])
    return dataset

def add_x0_to_dataset(dataset): 
    dataset_w_x0 = []
    for i in range(len(dataset)):
        new_point = [1.0]
        new_point.extend(dataset[i])
        dataset_w_x0.append(new_point)
    return dataset_w_x0 

#modifies the dataset
def add_labels(dataset, line): 
    same_side = False
    ones = 0 
    minus_ones = 0
    for datapoint in dataset: 
        if line.a*datapoint[0]+line.b <= datapoint[1]: 
            datapoint.append(1.0) 
            ones+=1
        else: 
            datapoint.append(-1.0) 
            minus_ones+=1
    if ones == 0 or minus_ones == 0: 
        same_side = True
    return dataset, same_side


#does not modify the dataset
def get_labels(dataset, line):
    same_side = False
    ones = 0
    minus_ones = 0
    labels = []
    for datapoint in dataset:
        if line.a*datapoint[0]+line.b <= datapoint[1]:
            labels.append(1.0)
            ones+=1
        else:
            labels.append(-1.0)
            minus_ones+=1
    if ones == 0 or minus_ones == 0:
        same_side = True
    return labels, same_side



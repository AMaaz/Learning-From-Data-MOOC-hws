# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:00:35 2016

@author: alisazhila
"""



import numpy as np 

LEFT_BOUNDARY = -1
RIGHT_BOUNDARY = 1 

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
    
    point1 = np.random.uniform(-1, 1, size=2)
    point2 = np.random.uniform(-1, 1, size=2)
    
    a = (point1[1]-point2[1])/(point1[0]-point2[0])
    b = point1[1] - a*point1[0]
    
    line = Line(a, b)
    return line 
    

    
def generate_dataset(N): 
    data_set = []
    x1_set = np.random.uniform(-1, 1, size=N)
    x2_set = np.random.uniform(-1, 1, size=N)
    for i in range(N): 
        data_set.append((1, x1_set[i], x2_set[i]))
    return data_set



def calculate_derivative(datapoint, w, line): 
    y = -1.0 
    if datapoint[2] > line.a*datapoint[1]+line.b: 
        y = 1.0 
    d_w0 = -1.0*y*datapoint[0]/(1.0+np.exp(y*(w[0]*datapoint[0]+w[1]*datapoint[1]+w[2]*datapoint[2])))        
    d_w1 = -1.0*y*datapoint[1]/(1.0+np.exp(y*(w[0]*datapoint[0]+w[1]*datapoint[1]+w[2]*datapoint[2])))    
    d_w2 = -1.0*y*datapoint[2]/(1.0+np.exp(y*(w[0]*datapoint[0]+w[1]*datapoint[1]+w[2]*datapoint[2])))
    #print (d_w0, d_w1, d_w2)
    return (d_w0, d_w1, d_w2)


def log_reg_with_sgd(line, data_set, eta, epsilon):        
    w = [0.0, 0.0, 0.0]
    w_t = [0.0, 0.0, 0.0]
    delta = 1 
    epoch = 0
    iterations = 0 
    while delta >= epsilon and epoch < 2000:   
        permutation = np.random.permutation(len(data_set))
        epoch +=1
        #print "epoch # ", epoch         
        w_t[0] = w[0]
        w_t[1] = w[1]
        w_t[2] = w[2]
        for i in permutation: 
            iterations += 1
            derivative = calculate_derivative(data_set[i], w, line)
            w[0] = w[0]-eta*derivative[0]
            w[1] = w[1]-eta*derivative[1]
            w[2] = w[2]-eta*derivative[2]
            #print "w_t+1=", w
            #print "w_t=", w_t
            #print (w_t[0]- w[0]), (w_t[1]- w[1])
            #delta = np.sqrt((w_t[0]- w[0])*(w_t[0]- w[0]) + (w_t[1]- w[1])*(w_t[1]- w[1]))
            #print "delta = ", delta            
        #print "w_t+1=", w
        #print "w_t=", w_t    
        delta = np.sqrt((w_t[0]- w[0])*(w_t[0]- w[0]) + (w_t[1]- w[1])*(w_t[1]- w[1])+(w_t[2]- w[2])*(w_t[2]- w[2]))
        #print "delta_epoch = ", delta
    return w, epoch                  


def calculate_err_out(test_set_size, w, line): 
    test_set = generate_dataset(test_set_size)
    #print test_set[:3]
    
    err = 0.0 
    for datapoint in test_set:  
        y = -1.0 
        if datapoint[2] > line.a*datapoint[1]+line.b: 
            y = 1.0 
        #print w, datapoint, y     
        err_i = np.log(1.0+np.exp(-1.0*y*np.dot(w, datapoint)))  
        #print "err_i=", err_i
        err += err_i
    return err/float(test_set_size) 
    
    
def experiment(num_exp_runs, tr_set_size, test_set_size): 
    err_avg = 0 
    epoch_avg = 0
    for j in range(num_exp_runs):
        #print "==========================="
        #print "Running experiment #", j+1
        line = generate_line()
        #print line.a, line.b 
        training_set = generate_dataset(tr_set_size)
        #print training_set[0:3]
        w = []
        w, epoch = log_reg_with_sgd(line, training_set, ETA, EPSILON) 
        print w, epoch  
        err_out = calculate_err_out(test_set_size, w, line)
        print err_out  
        err_avg += err_out  
        epoch_avg += epoch
    return err_avg/float(num_exp_runs), epoch_avg/float(num_exp_runs)    


print "err_out_avg, epoch_avg = ", experiment(100, 100, 200)    

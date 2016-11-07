# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 20:26:47 2016

@author: alisazhila
"""
 
import random
import numpy as np 


random.seed(8080)

def read_dta(fname):
   data = [] 
   for s in open(fname): 
       values = s.strip().split()
       values = np.array(map(float, values))
       for v in values: 
           v = float(v)
       data.append(values)
   return data     
    


def nonlinear_transformation(data):
    transformed_data = []
    labels = []
    for datapoint in data: 
        transformed_datapoint = []
        transformed_datapoint.append(1)             #1
        transformed_datapoint.append(datapoint[0])  #x1
        transformed_datapoint.append(datapoint[1])  #x2
        transformed_datapoint.append(datapoint[0]*datapoint[0])  #x1^2        
        transformed_datapoint.append(datapoint[1]*datapoint[1])  #x2^2
        transformed_datapoint.append(datapoint[0]*datapoint[1])  #x1*x2
        transformed_datapoint.append(abs(datapoint[0]-datapoint[1])) #|x1-x2|
        transformed_datapoint.append(abs(datapoint[0]+datapoint[1])) #|x1+x2|
        transformed_data.append(transformed_datapoint)
        labels.append(datapoint[2])
    return transformed_data, labels 
    
    
def scalar_product(w, x): 
    product = 0 
    for i in range(len(w)): 
        product += w[i]*x[i]
    return product     
    
    
def update_weights(w, datapoint): 
    w_new = []
    y = datapoint[-1]
    for i in range(len(w)): 
       w_new.append(w[i]+y*datapoint[i])
    return w_new   
         
         
    
def perceptron(training_data): 
    w = [0]*(len(training_data[0])-1)
    print training_data[0][-1]
    print w
    is_misclassified = True
    iteration = 0 
    while is_misclassified and iteration < 10000:     
        misclassifieds = []
        iteration +=1
        is_misclassified = False 
        for datapoint in training_data:
            #print scalar_product(w, datapoint[:-1])
            if scalar_product(w, datapoint[:-1])*datapoint[-1] <= 0:
            #if scalar_product(w, datapoint[:-1])!= datapoint[-1]:    
                is_misclassified = True 
                misclassifieds.append(datapoint)
        if is_misclassified:         
            rand_datapoint_index = random.randint(0, len(misclassifieds)-1)
            print rand_datapoint_index
            w = update_weights(w, misclassifieds[rand_datapoint_index])
            print w 
    print iteration    
    return w    
  
   
#(X.T*X).I*X.T*Y
def linear_reg(transf_data, labels): 
    X = np.matrix(transf_data)
    Y = np.matrix(labels)
    Xsq = X.T*X
    XsqI = Xsq.I
    Xdg = XsqI*X.T
    return Xdg*Y.T
  
#(X.T*X+lambda*E).I*X.T*Y
def linear_reg_w_weight_decay(transf_data, labels, lambda_coef): 
    print lambda_coef
    X = np.matrix(transf_data)
    Y = np.matrix(labels)
    Xsq = X.T*X
    I = np.identity(Xsq.shape[0])
    #E = np.ones_like(Xsq)        
    #print lambda_coef*E
    #print type(lambda_coef*E)
    #print "==========================================="
    XsqReg = (Xsq+lambda_coef*I)     
    #print XsqReg
    #print "==========================================="
    XsqI = XsqReg.I
    #print XsqI
    #print "============================================ Xdg"
    
    Xdg = XsqI*X.T
    #print Xdg
    #print "============================================ Xdg*Y"
    #print Xdg*Y.T
    return Xdg*Y.T


def estimate_err(w, dataset, labels): 
    err = 0 
    N = len(dataset)
    for i in range(len(dataset)):
        datapoint = dataset[i]
        y = labels[i]
        if y*scalar_product(w, datapoint) <=0: 
            err+=1
    return err/float(N)        
            

def ghost_experiment(): 

    #tr_data = read_dta('/Users/alisazhila/Dropbox/Study/Online_courses/ML@CalTechEdX/hws/data/in.dta')
    #test_data = read_dta('/Users/alisazhila/Dropbox/Study/Online_courses/ML@CalTechEdX/hws/data/out.dta')
    
    tr_data = read_dta('./data/in_ghost.dta')
    transformed_tr_data, labels = nonlinear_transformation(tr_data)    
    #w = linear_reg(transformed_tr_data, labels)
    w = linear_reg_w_weight_decay(transformed_tr_data, labels, lambda_coef)
    err_in  = estimate_err(w, transformed_tr_data, labels)

    test_data = read_dta('./data/out_ghost.dta')    
    transformed_test_data, labels = nonlinear_transformation(test_data)
    err_out = estimate_err(w, transformed_test_data, labels)
    return w, err_in, err_out 


#print experiment()

#tr_data = read_dta('/Users/alisazhila/Dropbox/Study/Online_courses/ML@CalTechEdX/hws/data/in.dta')
#print tr_data[0][2]+tr_data[1][2]
 
def experiment(): 
    tr_data = read_dta('./data/in.dta')
    transformed_tr_data, labels = nonlinear_transformation(tr_data)
    #w = linear_reg(transformed_tr_data, labels)
    w = linear_reg_w_weight_decay(transformed_tr_data, labels, lambda_coef)
    #print w
    err_in  = estimate_err(w, transformed_tr_data, labels)
    print "err_in=", err_in
    test_data = read_dta('./data/out.dta')
    transformed_data, labels = nonlinear_transformation(test_data)
    err_out = estimate_err(w, transformed_data, labels)
    print "err_out=", err_out 
    return w, err_in, err_out 
    
#lambda_coef = np.power(10.0, 3)  
#print "lambda = ", lambda_coef  

#print ghost_experiment()
errs = []
for k in [2, 1, 0, -1, -2]: 
   print "k=", k 
   lambda_coef = np.power(10.0, k)  
   print "lambda = ", lambda_coef 

   w, err_in, err_out =  experiment()
   errs.append(err_out)
print errs   
print min(errs)
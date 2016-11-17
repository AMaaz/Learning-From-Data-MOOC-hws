# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:04:17 2016

@author: alisazhila
"""

import numpy as np 
import hw6_2LinRegRegularized

def read_dta_25_10(fname):
   tr_data = [] 
   val_data =[]
   data = []
   for s in open(fname): 
       values = s.strip().split()
       values = np.array(map(float, values))
       for v in values: 
           v = float(v)
       data.append(values)
   tr_data = data[:25]
   val_data = data[-10:]    
       
   return tr_data, val_data  
   
   
def nonlinear_transformation_to_k(data, k):
    transformed_data = []
    labels = []
    if k > 0: 
      for datapoint in data: 
        transformed_datapoint = []                
        transformed_datapoint.append(1)             #1
        transformed_datapoint.append(datapoint[0])  #x1
        transformed_datapoint.append(datapoint[1])  #x2
        if k >=3: 
            transformed_datapoint.append(datapoint[0]*datapoint[0])  #x1^2        
        if k >=4:     
            transformed_datapoint.append(datapoint[1]*datapoint[1])  #x2^2
        if k >=5:     
            transformed_datapoint.append(datapoint[0]*datapoint[1])  #x1*x2
        if k >=6:     
            transformed_datapoint.append(abs(datapoint[0]-datapoint[1])) #|x1-x2|
        if k >=7:     
            transformed_datapoint.append(abs(datapoint[0]+datapoint[1])) #|x1+x2|
        transformed_data.append(transformed_datapoint)
        labels.append(datapoint[2])
    return transformed_data, labels    
   


def experiment_1(k): 
    tr_data, val_data = read_dta_25_10('./data/in.dta')
    transformed_tr_data, tr_labels = nonlinear_transformation_to_k(tr_data, k)
    #model training
    w = hw6_2LinRegRegularized.linear_reg(transformed_tr_data, tr_labels)
    #print w
    err_in  = hw6_2LinRegRegularized.estimate_err(w, transformed_tr_data, tr_labels)
    print "err_in=", err_in

    transformed_val_data, val_labels = nonlinear_transformation_to_k(val_data, k)
    err_out = hw6_2LinRegRegularized.estimate_err(w, transformed_val_data, val_labels)
    print "err_out=", err_out 
    return w, err_in, err_out 


def experiment_2(k): 
    tr_data, val_data = read_dta_25_10('./data/in.dta')
    transformed_tr_data, tr_labels = nonlinear_transformation_to_k(tr_data, k)
    #model training
    w = hw6_2LinRegRegularized.linear_reg(transformed_tr_data, tr_labels)
    #print w
    err_in  = hw6_2LinRegRegularized.estimate_err(w, transformed_tr_data, tr_labels)
    print "err_in=", err_in
     
    test_data = hw6_2LinRegRegularized.read_dta('./data/out.dta') 
    transformed_test_data, test_labels = nonlinear_transformation_to_k(test_data, k)
    err_out = hw6_2LinRegRegularized.estimate_err(w, transformed_test_data, test_labels)
    print "err_out=", err_out 
    return w, err_in, err_out 


def experiment_3(k): 
    val_data, tr_data = read_dta_25_10('./data/in.dta')
    transformed_tr_data, tr_labels = nonlinear_transformation_to_k(tr_data, k)
    #model training
    w = hw6_2LinRegRegularized.linear_reg(transformed_tr_data, tr_labels)
    #print w
    err_in  = hw6_2LinRegRegularized.estimate_err(w, transformed_tr_data, tr_labels)
    print "err_in=", err_in

    transformed_val_data, val_labels = nonlinear_transformation_to_k(val_data, k)
    err_out = hw6_2LinRegRegularized.estimate_err(w, transformed_val_data, val_labels)
    print "err_out=", err_out 
    return w, err_in, err_out 


def experiment_4(k): 
    val_data, tr_data = read_dta_25_10('./data/in.dta')
    transformed_tr_data, tr_labels = nonlinear_transformation_to_k(tr_data, k)
    #model training
    w = hw6_2LinRegRegularized.linear_reg(transformed_tr_data, tr_labels)
    #print w
    err_in  = hw6_2LinRegRegularized.estimate_err(w, transformed_tr_data, tr_labels)
    print "err_in=", err_in
     
    test_data = hw6_2LinRegRegularized.read_dta('./data/out.dta') 
    transformed_test_data, test_labels = nonlinear_transformation_to_k(test_data, k)
    err_out = hw6_2LinRegRegularized.estimate_err(w, transformed_test_data, test_labels)
    print "err_out=", err_out 
    return w, err_in, err_out 


errs = []
for k in [3,4,5,6,7]: 
   print "k=", k 

   w, err_in, err_out =  experiment_2(k)
   errs.append(err_out)
print errs   
print min(errs)
   
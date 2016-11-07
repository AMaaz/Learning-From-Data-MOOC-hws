# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:35:54 2016

@author: alisazhila
"""
import random, math


def generate_datset(N):
    X = []
    for i in range(N): 
      X.append(random.uniform(-1,1))
    return X

def expr_derivative_tanh(N):
    a = 0 
    for i in range(N): 
        x1 =  random.uniform(-1,1) 
        #print x1
        x2 =  random.uniform(-1,1)
        #print x2
        ai = (x1*math.tanh(math.pi*x1)+x2*math.tanh(math.pi*x2))/(x1*x1+x2*x2)
        a +=ai
        
    #print a/N
    return a/N, x1, x2
   
def perceptron(X): 
    a = 0
    is_missclassified = True 
    iteration_max = 1000
    iteration  = 0 
    while is_missclassified and iteration < iteration_max: 
        iteration+=1
        is_missclassified = False 
        H = []
        for x in X:  
           h = a*x
           H.append(h)
        missclassifieds = []   
        for i in range(len(H)): 
            if H[i] != math.tanh(math.pi*X[i]):
                is_missclassified = True 
                missclassifieds.append(X[i])
        if len(missclassifieds) > 0: 
            j = random.randint(0, len(missclassifieds)-1)
            #w1 = w0 + a( d - y )x 
            a = a + (math.tanh(math.pi*X[j]) - a*X[j])*X[j]            
    #print iteration
    return a    
    
def bias_tanh(N): 
   E = 0 
   for i in range(1, N+1): 
       x =  random.uniform(-1,1) 
       e = (1.65*x - math.tanh(math.pi*x))*(1.65*x - math.tanh(math.pi*x))
       E+=e
   return E/float(N)     
    

def variance_tanh(N): 
   E = 0 
   for i in range(1, N+1): 
       #ai, x1, x2 = expr_derivative_tanh(1000)
       ai = perceptron(generate_datset(2))
       x =  random.uniform(-1,1) 
       e = math.pow((1.65*x - ai*x),2)
       E+=e
   return E/float(N)     
   



#print expr_derivative_tanh(10000)    
#print bias_tanh(100000)    
print variance_tanh(1000)    
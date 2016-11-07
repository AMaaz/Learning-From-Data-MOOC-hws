# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 01:02:55 2016

@author: alisazhila
"""

import random, math 

#random.seed(100)


def expr_derivative(N):
    a = 0 
    for i in range(N): 
        x1 =  random.uniform(-1,1) 
        #print x1
        x2 =  random.uniform(-1,1)
        #print x2
        ai = (x1*math.sin(math.pi*x1)+x2*math.sin(math.pi*x2))/(x1*x1+x2*x2)
        a +=ai
        
    print a/N   


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
            if H[i] != math.sin(math.pi*X[i]):
                is_missclassified = True 
                missclassifieds.append(X[i])
        if len(missclassifieds) > 0: 
            j = random.randint(0, len(missclassifieds)-1)
            #w1 = w0 + a( d - y )x 
            a = a + (math.sin(math.pi*X[j]) - a*X[j])*X[j]            
    #print iteration
    return a         



def expr_derivative_tanh(N):
    a = 0 
    for i in range(N): 
        x1 =  random.uniform(-1,1) 
        #print x1
        x2 =  random.uniform(-1,1)
        #print x2
        ai = (x1*math.tanh(math.pi*x1)+x2*math.tanh(math.pi*x2))/(x1*x1+x2*x2)
        a +=ai
          
    return a/N
    
def bias(N): 
   E = 0 
   for i in range(1, N+1): 
       x =  random.uniform(-1,1) 
       e = (1.43*x - math.sin(math.pi*x))*(1.43*x - math.sin(math.pi*x))
       E+=e
   return E/float(N)     
    

def variance(N): 
   E = 0 
   for i in range(1, N+1): 
       ai = perceptron(generate_datset(2))
       x =  random.uniform(-1,1) 
       e = math.pow((1.43*x - ai*x),2)
       E+=e
   return E/float(N)    
       
   
   
   
def generate_datset(N):
    X = []
    for i in range(N): 
      X.append(random.uniform(-1,1))
    return X
    
    
def experiment(K):   
    A = 0 
    for i in range(1,K+1):  
        X = generate_datset(2)
        if K < 10: 
            print X 
        a = perceptron(X)        
        A+=a
        #print a, A/float(i) 
        
    return A/float(K)     
       
       
#print expr_derivative(100000000)
#print experiment(1000000)        
#print bias(100000)
#print variance(10000)       
       
#g = b 
def out_of_sample_a(N): 
   E = 0 
   for i in range(N): 
       X = generate_datset(2)
       bi = (math.sin(math.pi*X[0])+ math.sin(math.pi*X[1]))/2
       x =  random.uniform(-1,1) 
       e = math.pow((math.sin(math.pi*x)-bi),2)
       E+=e
   return E/float(N)    
       
#g = ax           
def out_of_sample_b(N): 
   E = 0 
   for i in range(N): 
       X = generate_datset(2)
       ai = (X[0]*math.sin(math.pi*X[0])+X[1]*math.sin(math.pi*X[1]))/(X[0]*X[0]+X[1]*X[1])
       x =  random.uniform(-1,1) 
       e = math.pow((math.sin(math.pi*x)-ai*x),2)
       E+=e
   return E/float(N)           
          
#g = ax+b           
def out_of_sample_c(N): 
   E = 0 
   for i in range(N): 
       X = generate_datset(2)
       ai = (math.sin(math.pi*X[0])- math.sin(math.pi*X[1]))/(X[0]-X[1])
       bi = math.sin(math.pi*X[0]) - ai*X[0]       
       x =  random.uniform(-1,1) 
       e = math.pow((math.sin(math.pi*x)-(ai*x+bi)),2)
       E+=e
   return E/float(N)           
    
#g = ax^2           
def out_of_sample_d(N): 
   E = 0 
   for i in range(N): 
       X = generate_datset(2)
       ai = (X[0]*X[0]*math.sin(math.pi*X[0])+X[1]*X[1]*math.sin(math.pi*X[1]))/(math.pow(X[0],4)+math.pow(X[1],4))
       x =  random.uniform(-1,1) 
       e = math.pow((math.sin(math.pi*x)-ai*x*x),2)
       E+=e
   return E/float(N)
 
#g = ax^2+b           
def out_of_sample_e(N): 
   E = 0 
   for i in range(N): 
       X = generate_datset(2)
       ai = (math.sin(math.pi*X[0])- math.sin(math.pi*X[1]))/(X[0]*X[0]-X[1]*X[1])
       bi = math.sin(math.pi*X[0]) - ai*X[0]*X[0]       
       #print ai, "*x^2+", bi
       x =  random.uniform(-1,1) 
       e = math.pow((math.sin(math.pi*x)-(ai*x*x+bi)),2)
       #print e 
       E+=e
   return E/float(N) 
   

print out_of_sample_a(10000)
print out_of_sample_b(10000)   
print out_of_sample_c(10000)
print out_of_sample_d(10000)
print out_of_sample_e(10000)
#print out_of_sample_e(10)
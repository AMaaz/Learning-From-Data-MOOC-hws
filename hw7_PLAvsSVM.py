# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:09:02 2016

@author: alisazhila
"""

import dataset_generator as gd
import random 
import quadprog as qp
import numpy as np
import cvxopt
from sklearn.svm import SVC
import scipy.optimize
#import sys
#sys.exit()

TINY_ADD = np.power(10.0, -13)

#np.testing.assert_array_almost_equal(result.x, xf)
def solve_qp_scipy(G, a, C, b, meq=0):
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    if C is not None and b is not None:
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    else:
        constraints = []

    result = scipy.optimize.minimize(f, x0=np.zeros(len(G)), method='COBYLA',
        constraints=constraints, tol=1e-10)
    return result




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
    
    #random.seed(100)
    #start PLA with all-zero vector
    w = [0]*(len(training_data[0])-1)  #w/o label point 
    #print w
    is_misclassified = True
    iteration = 0 
    while is_misclassified and iteration < 10000:     
        misclassifieds = []
        iteration +=1
        is_misclassified = False 
        #datapoint : (x0, x1, x2)
        i = 0 
        for datapoint in training_data:
            #print scalar_product(w, datapoint[:-1])
            #print "datapoint: ", i, " (h, y): ", np.sign(scalar_product(w, datapoint[:-1])), datapoint[-1]
            i+=1
            if scalar_product(w, datapoint[:-1])*datapoint[-1] <= 0:
            #if scalar_product(w, datapoint[:-1])!= datapoint[-1]:    
                is_misclassified = True 
                misclassifieds.append(datapoint)
        if is_misclassified:         
            rand_datapoint_index = random.randint(0, len(misclassifieds)-1)
            #print rand_datapoint_index
            w = update_weights(w, misclassifieds[rand_datapoint_index])
            #print w 
    #print iteration    
    return w, iteration
    

#test_set: (x0, x1, x2, label)    
def calculate_err_out_pla(test_set_w_labels, w): 
    #print test_set[:3]    
    err = 0.0 
    for datapoint in test_set_w_labels:  
        if scalar_product(w, datapoint[:-1])*datapoint[-1] <= 0:
            err+=1
    return err/float(len(test_set_w_labels)) 
    

#Minimize     1/2 x^T G x - a^T x
#Subject to   C.T x >= b
'''
solve_qp(...)
    Solve a strictly convex quadratic program

    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b
    
Parameters
    ----------
    G : array, shape=(n, n)
        matrix appearing in the quadratic function to be minimized
    a : array, shape=(n,)
        vector appearing in the quadratic function to be minimized
    C : array, shape=(n, m)
        matrix defining the constraints under which we want to minimize the
        quadratic function
    b : array, shape=(m), default=None
        vector defining the constraints
    meq : int, default=0
        the first meq constraints are treated as equality constraints,
        all further as inequality constraints (defaults to 0).
    factorized : bool, default=False
        If True, then we are passing :math:`R^{−1}` (where :math:`G = R^T R`)
        instead of the matrix G in the argument G.

    Returns
    -------
    x : array, shape=(n,)
        vector containing the solution of the quadratic programming problem.
'''

def svm(dataset_w_labels): 
   #random.seed(100)

   N = len(dataset_w_labels)
   G = []
   a = []
   C_T = []  # N x m=N+1 (Y_T*alpha = 0 constraint + alpha_i>=0 for i[1..N])
   C_T.append([]) # stub for Y^T ; dim N, 1
   C2_T = []
   C2_T.append([])
   C2_T.append([]) # stub for minus_Y^T ; dim N, 1
   Y_T = []
   minus_Y_T = []
   b = [] #dim m=N+1
   for i in range(N):
       row = []
       a.append(1.0)
       datapoint_i = dataset_w_labels[i]
       Y_T.append(datapoint_i[-1])
       minus_Y_T.append(-1*datapoint_i[-1])
       b.append(0.0)
       I = []
       for j in range(N): 
           datapoint_j=dataset_w_labels[j]
           #print datapoint_i[-1], datapoint_j[-1]
           g_ij = datapoint_i[-1]*datapoint_j[-1]*scalar_product(datapoint_i[:-1], datapoint_j[:-1])
           # sly modification to make teh G mtx positive definite: 
           if i == j: 
               g_ij += TINY_ADD
           row.append(g_ij)
           I.append(0)
       I[i] = 1    
       G.append(row)
       C_T.append(I)
       C2_T.append(I)
   C_T[0] = Y_T
   C2_T[0] = Y_T
   C2_T[1] = minus_Y_T
   b.append(0.0)
   #for the "equality through 2 inequalities" case 
   b2 = []
   b2.extend(b)
   b2.append(0.0)
   

   a = np.asarray(a, dtype=np.double)
   #print "a=1vec:", a

   G = np.array([np.array(gi, dtype=np.double) for gi in G])
   #print G

   #b = np.transpose(np.asarray(b, dtype=np.double))
   b = np.asarray(b, dtype=np.double)
   #print "b=0vec:", b
   C_T =   np.array([np.array(ci, dtype=np.double) for ci in C_T]) 
   #print "C_before_T: \n", C_T
   C = np.transpose(C_T)
   #print C
   
   alpha, f, xu, iters, lagr, iact  = qp.solve_qp(G, a, C, b, meq = 1)

   '''
   print 'equality via 2 inequalities:' 
   b2 = np.asarray(b2, dtype=np.double)   
   #print b2   
   C2 = np.transpose(np.array([np.array(ci, dtype=np.double) for ci in C2_T]))
   #print C2   
   alpha, f, xu, iters, lagr, iact  = qp.solve_qp(G, a, C2, b2)
   '''
   
   #print alpha, f, xu, iters, lagr, iact
   #print "alpha = ", alpha
   #print "lagr =", lagr

   #result = solve_qp_scipy(G, a, C2, b2, meq=0)
   #alpha2 = result.x
   #print "alpha2:", result.x
   
   w = [0, 0]
   sv_num = 0
   non_zero_ids = [] 
   #print "threshold = ", np.power(10.0, -2)
   print "threshold = ", 1/float(N*N)
   for i in range(N):
       datapoint_i = dataset_w_labels[i]
       "the sly modification, pt 2: because we get more non-zero alphas," 
       "we're discarding small values"
       #print "alpha, epsilon: ", alpha[i], 1/float(N*N)
       if alpha[i] > 1/float(N*N): 
       #if alpha[i] > 1/float(N):     
       #if alpha[i] > np.power(10.0, -2):     
           #print i, alpha[i]
           non_zero_ids.append(i)
           sv_num+=1
           w[0] += alpha[i]*datapoint_i[-1]*datapoint_i[0]
           w[1] += alpha[i]*datapoint_i[-1]*datapoint_i[1]
                   
   #print "Non zero SVs:", non_zero_ids

   #line W*X+c
   #taking first non-zero vec
   rand_idx = random.randint(0, len(non_zero_ids)-1)
   sv = dataset_w_labels[non_zero_ids[rand_idx]]
   #print sv
   #print w
   #print scalar_product(w, sv[:-1])   
   #print 1/float(sv[-1])   
   c =  1/float(sv[-1])-scalar_product(w, sv[:-1])
   #print c
 
   return  w, c, alpha, sv_num 



'''
Minimize     1/2 x^T P x + q^T x
subject to   Gx <= h
             Ax = b
             
from cvxopt import solvers
sol = solvers.qp(P,q,G,h)
That’s it! If you had A, b as well, you would call:
sol = solvers.qp(P,q,G,h,A,b)
             
             
'''
def svm_cvxopt(dataset_w_labels):   
   #random.seed(100)

   N = len(dataset_w_labels)
   P = []
   q = []
   G = []  # N x N (alpha_i>=0 for i[1..N])
   Y = []
   h = [] #dim N
   b = [0.0] #Ax = b => Y_T*x = 0
   for i in range(N):
       row = []
       q.append(-1.0)
       datapoint_i = dataset_w_labels[i]
       Y.append([datapoint_i[-1]])
       h.append(0.0)
       I = []
       for j in range(N): 
           datapoint_j=dataset_w_labels[j]
           #print datapoint_i[-1], datapoint_j[-1]
           p_ij = datapoint_i[-1]*datapoint_j[-1]*scalar_product(datapoint_i[:-1], datapoint_j[:-1])
           ' sly modification to make the P mtx positive definite:' 
           if i == j: 
               p_ij += TINY_ADD
           row.append(p_ij)
           I.append(0.0)
       I[i] = -1.0    
       P.append(row)
       #inequality condition matrix
       G.append(I)

   q=cvxopt.matrix(q, tc='d') 
   #print "q=-1vec:", q
   P = cvxopt.matrix(P, tc='d')#np.array([np.array(p_i, dtype=np.double) for p_i in P])
   #print P
   h = cvxopt.matrix(h, tc='d')
   #print "h=0vec:", h
   G = cvxopt.matrix(G, tc='d') 
   #print "Ineq constr mtx: \n", G   
   A = cvxopt.matrix(Y, tc='d') 
   #print "A vec: ", A   
   b = cvxopt.matrix(b, tc='d')
   #print "b =", b
      
   solution = cvxopt.solvers.qp(P,q,G,h,A,b)
   alpha = solution['x']
   #print alpha
   
   w = [0, 0]
   sv_num = 0
   non_zero_ids = [] 
   for i in range(N):
       datapoint_i = dataset_w_labels[i]
       "the sly modification, pt 2: because we get more non-zero alphas," 
       "we're discarding small values"
       #if alpha[i] > 1/float(N*N): 
       if alpha[i] > np.power(10.0, -2):     
       #if alpha[i] > 0: 
           #print i, alpha[i]
           non_zero_ids.append(i)
           sv_num+=1
           w[0] += alpha[i]*datapoint_i[-1]*datapoint_i[0]
           w[1] += alpha[i]*datapoint_i[-1]*datapoint_i[1]
                   
   print "Non zero SVs:", non_zero_ids

   'line W*X+c'
   'taking random non-zero SV'
   rand_idx = random.randint(0, len(non_zero_ids)-1)
   sv = dataset_w_labels[non_zero_ids[rand_idx]]
   #print sv
   #print w
   #print scalar_product(w, sv[:-1])   
   #print 1/float(sv[-1])   
   c =  1/float(sv[-1])-scalar_product(w, sv[:-1])
   #print c
 
   return  w, c, alpha, sv_num 

    
    
#test_set: (x1, x2, label)    
def calculate_err_out_svm(test_set_w_labels, w, c): 
    #print test_set[:3]    
    err = 0.0 
    for datapoint in test_set_w_labels:  
        if (scalar_product(w, datapoint[:-1])+c)*datapoint[-1] <= 0:
            err+=1
    return err/float(len(test_set_w_labels)) 
   


'''
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.svm import SVC
>>> clf = SVC()
>>> clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> print(clf.predict([[-0.8, -1]]))
[1]
'''
def svm_skt(dataset_w_labels):

    'modifying the dataset as needed by SVC'
    datapoints = []
    labels = []
    for datapoint in dataset_w_labels:
        datapoints.append(datapoint[:-1])
        labels.append(datapoint[-1])

    clf = SVC(C=float("inf"), kernel='linear')
    clf.fit(datapoints, labels)
    return clf


def calculate_error_out_svm_skt(test_set_w_labels, model):
    err = 0.0
    for datapoint in test_set_w_labels:
        if model.predict([datapoint[:-1]])*datapoint[-1] <= 0:
            err+=1
    return err/float(len(test_set_w_labels))


def experiment(num_exp_runs, tr_set_size, test_set_size): 
    err_avg_pla = 0 
    err_avg_svm = 0
    svm_better_pla_num = 0 
    sv_num_avg = 0
    
    for j in range(num_exp_runs):
        #print "==========================="
        #print "Running experiment #", j+1
        print j+1
        line = gd.generate_line()
        #print line.a, line.b 
        same_side = True 
        
        # discarding cases where all points happen to be on one side
        while(same_side): 
            tr_set = gd.generate_dataset(tr_set_size)
            tr_set_w_labels, same_side = gd.add_labels(tr_set, line)
        #print tr_set[:2]    
        #print tr_set_w_labels[:2]
        tr_set_w_x0 = gd.add_x0_to_dataset(tr_set_w_labels)
        #print tr_set_w_x0[0:2]

        test_set = gd.generate_dataset(test_set_size)
        #print test_set[:2]
        test_set_w_labels, same_side = gd.add_labels(test_set, line)
        test_set_w_x0 = gd.add_x0_to_dataset(test_set_w_labels)



        w_pla, iterations = perceptron(tr_set_w_x0)
        #print "w_pla", w_pla
        #print w_pla, iterations  
                
        err_out_pla = calculate_err_out_pla(test_set_w_x0, w_pla)
        #print "Err_PLA: ", err_out_pla  
        err_avg_pla += err_out_pla  

        '''
        #w_svm, c, alpha, sv_num = svm(tr_set_w_labels)
        w_svm, c, alpha, sv_num = svm_cvxopt(tr_set_w_labels) 
        #print "w_svm: ", w_svm, c         
        #print w_svm, c, alpha 
        sv_num_avg += sv_num
        '''

        model = svm_skt(tr_set_w_labels)
        #print model.support_
        #print model.support_vectors_
        sv_num = len(model.support_)
        sv_num_avg += sv_num

        
        #print test_set_w_labels[:2]
        #err_out_svm = calculate_err_out_svm(test_set_w_labels, w_svm, c)
        err_out_svm = calculate_error_out_svm_skt(test_set_w_labels, model)
        err_avg_svm += err_out_svm        
        
        #print "(Err_PLA, Err_SVM): ", err_out_pla,  err_out_svm
        if err_out_svm <= err_out_pla:
            svm_better_pla_num+=1

    #return err_avg_pla/float(num_exp_runs), err_avg_svm/float(num_exp_runs)
    return svm_better_pla_num/float(num_exp_runs), sv_num_avg/float(num_exp_runs)

'''
alpha_expected = [0.5, 0.5, 1, 0]
'''
def ghost_experiment():
    tr_set_w_labels = [[0.0, 0.0, -1.0], 
                  [2.0, 2.0, -1.0], 
                  [2.0, 0.0, 1.0], 
                  [3.0, 0.0, 1.0]]
                  
    svm_cvxopt(tr_set_w_labels)



num1, num2 = experiment(1000, 100, 200)
print num1, num2
#print "err_avg_pla, err_avg_svm = ", err_pla, err_svm

#ghost_experiment()
    
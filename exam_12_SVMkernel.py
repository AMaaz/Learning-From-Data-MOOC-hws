'''
project_name: exam_12_SVMkernel
@author: alisazhila
date: 11/27/16
'''

import cvxopt
import numpy as np
import random
from sklearn.svm import SVC

TINY_ADD = np.power(10.0, -13)



def scalar_product(w, x):
    product = 0.0
    print w
    print x
    for i in range(len(w)):
        product += w[i]*x[i]
    return product

#K(x1, x2) = (1+x1*x2)^2
def kernel(datapoint1, datapoint2):
    p1 = (1.0 + scalar_product(datapoint1, datapoint2))
    return p1*p1


def my_kernel(X, Y):
    mtx = []
    for i in range(len(X)):
        mtx.append([])
        for j in range(len(X)):
            print Y[i]*Y[j]*kernel(X[i], X[j])
            mtx[i].append(Y[i]*Y[j]*kernel(X[i], X[j]))
    print mtx
    M = np.array(mtx)
    print M
    print M.shape
    return M





'''
Minimize     1/2 x^T P x + q^T x
subject to   Gx <= h
             Ax = b

from cvxopt import solvers
sol = solvers.qp(P,q,G,h)
That's it! If you had A, b as well, you would call:
sol = solvers.qp(P,q,G,h,A,b)
'''
def svm_kernel_cvxopt(datapoints, labels, kernel):
    # random.seed(100)

    N = len(datapoints)
    P = []
    q = []
    G = []  # N x N (alpha_i>=0 for i[1..N])
    Y = []
    h = []  # dim N
    b = [0.0]  # Ax = b => Y_T*x = 0
    for i in range(N):
        row = []
        q.append(-1.0)
        Y.append([labels[i]])
        h.append(0.0)
        I = []
        for j in range(N):
            # print datapoint_i[-1], datapoint_j[-1]
            p_ij = labels[i] * labels[j] * kernel(datapoints[i], datapoints[j])
            ' sly modification to make the P mtx positive definite:'
            if i == j:
                p_ij += TINY_ADD
            row.append(p_ij)
            I.append(0.0)
        I[i] = -1.0
        P.append(row)
        # inequality condition matrix
        G.append(I)

    q = cvxopt.matrix(q, tc='d')
    # print "q=-1vec:", q
    P = cvxopt.matrix(P, tc='d')  # np.array([np.array(p_i, dtype=np.double) for p_i in P])
    # print P
    h = cvxopt.matrix(h, tc='d')
    # print "h=0vec:", h
    G = cvxopt.matrix(G, tc='d')
    # print "Ineq constr mtx: \n", G
    A = cvxopt.matrix(Y, tc='d')
    # print "A vec: ", A
    b = cvxopt.matrix(b, tc='d')
    # print "b =", b

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = solution['x']
    # print alpha

    w = [0, 0]
    sv_num = 0
    non_zero_ids = []
    for i in range(N):
        "the sly modification, pt 2: because we get more non-zero alphas,"
        "we're discarding small values"
        # if alpha[i] > 1/float(N*N):
        if alpha[i] > np.power(10.0, -2):
            # if alpha[i] > 0:
            # print i, alpha[i]
            non_zero_ids.append(i)
            sv_num += 1
            w[0] += alpha[i] * labels[i] * datapoints[i][0]
            w[1] += alpha[i] * labels[i] * datapoints[i][1]

    print "Non zero SVs:", non_zero_ids

    'line W*X+c'
    'taking random non-zero SV'
    rand_idx = random.randint(0, len(non_zero_ids) - 1)
    sv = datapoints[non_zero_ids[rand_idx]]
    # print sv
    # print w
    # print scalar_product(w, sv[:-1])
    # print 1/float(sv[-1])
    c = 1 / float(labels[non_zero_ids[rand_idx]]) - scalar_product(w, sv)
    # print c

    return w, c, alpha, sv_num


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
def svm_skt_kernel(datapoints, labels):

    clf = SVC(C=float("inf"), kernel=my_kernel)

    clf.fit(datapoints, labels)
    return clf



def problem12(X, Y):
    w, c, alpha, sv_num = svm_kernel_cvxopt(X, Y)
    print alpha, sv_num


    print "=================================="
    print "Trying scikit package SVM w/ custom kernel... "
    model = svm_skt_kernel(X, Y)
    print len(model.support_), model.support_vectors_

    #print "=================================="
    #print "Trying scikit package SVM... "
    #clf = SVC(C=float("inf"), kernel="poly", degree=2)
    #model = clf.fit(X, Y)
    #print len(model.support_), model.support_vectors_




x1 = [1.0, 0.0]
x2 = [0.0, 1.0]
x3 = [0.0, -1.0]
x4 = [-1.0, 0.0]
x5 = [0.0, 2.0]
x6 = [0.0, -2.0]
x7 = [-2.0, 0.0]

Y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]

X = [x1, x2, x3, x4, x5, x6, x7]


#problem12(X, Y)

my_kernel(X, Y)

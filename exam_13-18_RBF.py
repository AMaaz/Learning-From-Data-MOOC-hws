'''
project_name: exam_13-18_RBF
@author: alisazhila
date: 11/27/16
'''

import site
site.addsitedir("/usr/local/lib/python2.7/site-packages")


import dataset_generator as gen
import numpy as np
import cvxopt
import random
from sklearn.svm import SVC
import quadprog as qp
#import matplotlib.pyplot as plt
import time


TINY_ADD = np.power(10.0, -13)

#sign(x2-x1+0.25*sin(pi*x1))
def func(datapoint):
    f = datapoint[1] - datapoint[0] + 0.25*np.sin(np.pi*datapoint[0])
    #f = datapoint[1] - datapoint[0] + 0.25 * np.cos(np.pi * datapoint[0])
    #print datapoint, f
    return np.sign(f)


def get_labels(datapoints):
    labels = []
    for datapoint in datapoints:
        labels.append(func(datapoint))
    return labels


def scalar_product(w, x):
    product = 0.0
    for i in range(len(w)):
        product += w[i]*x[i]
    return product


def squared_euclidian_distance(datapoint1, datapoint2):
    return np.power((datapoint1[0] - datapoint2[0]), 2.0) + np.power((datapoint1[1] - datapoint2[1]), 2.0)


'''
def plot_lloyds(datapoints, centroids):

    for datapoint in datapoints:
        plt.plot(datapoint[0], datapoint[1], 'ro')
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], 'bs')
    plt.axis([-1, 1, -1, 1])
    plt.show()
    time.sleep(1)
'''

def find_centroid(cluster):
    if len(cluster) == 0:
        return None
    s = len(cluster)
    dimensions = len(cluster[0])
    centroid = [0.0]*dimensions
    for datapoint in cluster:
        for i in range(dimensions):
            centroid[i] += datapoint[i]
    centroid = np.array(centroid)
    return centroid/float(s)


def check_if_points_are_same(datapoint1, datapoint2):
    is_close = np.isclose(datapoint1, datapoint2)
    for val in is_close:
        if not val:
            return False
    return True


def lloyds_clustering(datapoints, mus):

    centroids_num = len(mus)
    centroids = np.array(mus)

    centroids_changed = True
    iteration_limit = 1000
    iteration = 0
    while centroids_changed or iteration >= iteration_limit:
        #clusters = [[]] * centroids_num
        #plot_lloyds(datapoints, centroids)
        #print datapoints
        #print centroids
        clusters = {}
        for datapoint in datapoints:
            distances = []
            for centroid in centroids:
                distances.append(squared_euclidian_distance(datapoint, centroid))
            #print distances
            id = distances.index(min(distances))
            idx = str(id)
            #print idx, type(idx)
            if idx in clusters:
                clusters[idx].append(datapoint)
            else:
                clusters[idx] = [datapoint]
            #print "clusters: \n", clusters
        new_centroids = []
        if len(clusters) < centroids_num:
            print "[WARNING] empty clusters!!!"
            return None
        changed_in_iteration = []
        for i in range(len(clusters)):
            new_centroid = find_centroid(clusters[str(i)])
            # print i, new_centroid
            new_centroids.append(new_centroid)
            #print "Centroids changed in iter: ", changed_in_iteration
            changed_in_iteration.append(not check_if_points_are_same(new_centroid, centroids[i]))
        #print "New_centroids:", new_centroids

        centroids_changed = (np.array(changed_in_iteration).any() == True)
        centroids = new_centroids
        iteration+=1

    return centroids, iteration


'''
   G = np.array([np.array(gi, dtype=np.double) for gi in G])
   #print G

   #b = np.transpose(np.asarray(b, dtype=np.double))
   b = np.asarray(b, dtype=np.double)
   #print "b=0vec:", b
   C_T =   np.array([np.array(ci, dtype=np.double) for ci in C_T])
   #print "C_before_T: \n", C_T
   C = np.transpose(C_T)
'''
def pseudo_inverse(datapoints, labels, centroids, gamma):
    N = len(datapoints)
    K = len(centroids)

    F = []
    for i in range(N):
        # for w0
        row = [1.0]
        for j in range(K):
            #row.append(np.exp(-1.0*gamma*squared_euclidian_distance(datapoints[i], centroids[j])))
            row.append(rbf_kernel(datapoints[i], centroids[j], gamma))
        F.append(row)

    #print F[0]
    F = np.matrix(F)
    #print F[0]
    #print F.shape
    Y = np.matrix(labels)
    #print Y.shape
    Fsq = F.T * F
    FsqI = Fsq.I
    #print np.allclose(np.dot(Fsq, FsqI), np.eye(Fsq.shape[0]))
    Fdg = FsqI * F.T
    w = Fdg * Y.T
    return w, F



def rbf(datapoints, labels, mus, gamma):
    output = lloyds_clustering(datapoints, mus)
    if not output:
        print "[discarding run] empty clusters"
        return None
    centroids, iters = output
    #print iters
    if iters >= 1000:
        print "Iterations more than limit!!!"
        return None

    #datapoints_w_w0 = gen.add_x0_to_dataset(datapoints)
    #centroids_w_w0 = gen.add_x0_to_dataset(centroids)
    #print centroids_w_w0
    #print len(centroids)

    #w = pseudo_inverse(datapoints_w_w0, labels, centroids_w_w0, gamma)
    w, F = pseudo_inverse(datapoints, labels, centroids, gamma)
    return w, centroids, F


def rbf_kernel(datapoint1, datapoint2, gamma):
    return np.exp(-1.0 * gamma * squared_euclidian_distance(datapoint1, datapoint2))

#sum(wk*exp(-gamma*(x_vec-centroid_k)^2))+b
def rbf_function(datapoint, centroids, w, b, gamma):
    sum = 0.0
    for k in range(len(centroids)):
        sum+=w[k]*rbf_kernel(datapoint, centroids[k], gamma)
    sum += b
    return np.sign(sum)

def calculate_err_rbf(datapoints,labels, w, b, gamma, centroids):
    #print test_set[:3]
    err = 0.0
    for i in range(len(datapoints)):
        #print np.sign(scalar_product(w, datapoints[i])+b), labels[i]
        if rbf_function(datapoints[i], centroids, w, b, gamma)*labels[i] <= 0:
            err+=1
    return err/float(len(datapoints))


###
###  DO NOT USE
###
def calculate_err_svm(datapoints, labels, alpha, b, svs):
    #print test_set[:3]
    err = 0.0
    for i in range(len(datapoints)):
        break
        #print np.sign(scalar_product(w, datapoints[i])+b), labels[i]
        #if (scalar_product(w, datapoints[i])+b)*labels[i] <= 0:
        #    err+=1
    return err/float(len(datapoints))









'''
Minimize     1/2 x^T P x + q^T x
subject to   Gx <= h
             Ax = b

from cvxopt import solvers
sol = solvers.qp(P,q,G,h)
That's it! If you had A, b as well, you would call:
sol = solvers.qp(P,q,G,h,A,b)
'''
def svm_kernel_cvxopt(datapoints, labels, kernel, gamma=1.5):
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
            p_ij = labels[i] * labels[j] * kernel(datapoints[i], datapoints[j], gamma)
            ' sly modification to make the P mtx positive definite:'
            #if i == j:
                #p_ij += TINY_ADD
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
    #print alpha

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

    #print "Non zero SVs:", non_zero_ids

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


def svm_skt(datapoints, labels, gamma):
    clf = SVC(C=float("inf"), kernel='rbf', gamma=gamma)
    clf.fit(datapoints, labels)
    return clf


def calculate_error_skt(datapoints, labels, model):
    err = 0.0
    for i in range(len(datapoints)):
        #print np.sign(model.predict([datapoints[i]])), labels[i]
        if model.predict([datapoints[i]])*labels[i] <= 0:
            err+=1.0
    return err/float(len(datapoints))


def svm_qp(datapoints, labels, kernel, gamma):
    # random.seed(100)

    N = len(datapoints)
    G = []
    a = []
    C_T = []  # N x m=N+1 (Y_T*alpha = 0 constraint + alpha_i>=0 for i[1..N])
    C_T.append([])  # stub for Y^T ; dim N, 1
    C2_T = []
    C2_T.append([])
    C2_T.append([])  # stub for minus_Y^T ; dim N, 1
    Y_T = []
    minus_Y_T = []
    b = []  # dim m=N+1
    for i in range(N):
        row = []
        a.append(1.0)
        Y_T.append(labels[i])
        minus_Y_T.append(-1 * labels[i])
        b.append(0.0)
        I = []
        for j in range(N):
            g_ij = labels[i] * labels[j] * kernel(datapoints[i], datapoints[j], gamma)
            # sly modification to make the G mtx positive definite:
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
    # for the "equality through 2 inequalities" case
    b2 = []
    b2.extend(b)
    b2.append(0.0)

    a = np.asarray(a, dtype=np.double)
    # print "a=1vec:", a

    G = np.array([np.array(gi, dtype=np.double) for gi in G])
    # print G

    # b = np.transpose(np.asarray(b, dtype=np.double))
    b = np.asarray(b, dtype=np.double)
    # print "b=0vec:", b
    C_T = np.array([np.array(ci, dtype=np.double) for ci in C_T])
    # print "C_before_T: \n", C_T
    C = np.transpose(C_T)
    # print C

    alpha, f, xu, iters, lagr, iact = qp.solve_qp(G, a, C, b, meq=1)

    '''
    print 'equality via 2 inequalities:'
    b2 = np.asarray(b2, dtype=np.double)
    #print b2
    C2 = np.transpose(np.array([np.array(ci, dtype=np.double) for ci in C2_T]))
    #print C2
    alpha, f, xu, iters, lagr, iact  = qp.solve_qp(G, a, C2, b2)
    '''

    # print alpha, f, xu, iters, lagr, iact
    # print "alpha = ", alpha
    # print "lagr =", lagr

    # result = solve_qp_scipy(G, a, C2, b2, meq=0)
    # alpha2 = result.x
    # print "alpha2:", result.x

    w = [0, 0]
    sv_num = 0
    non_zero_ids = []
    # print "threshold = ", np.power(10.0, -2)
    print "threshold = ", 1 / float(N * N)
    for i in range(N):
        "the sly modification, pt 2: because we get more non-zero alphas,"
        "we're discarding small values"
        # print "alpha, epsilon: ", alpha[i], 1/float(N*N)
        if alpha[i] > 1 / float(N * N):
            # if alpha[i] > 1/float(N):
            # if alpha[i] > np.power(10.0, -2):
            # print i, alpha[i]
            non_zero_ids.append(i)
            sv_num += 1
            w[0] += alpha[i] * labels[i] * datapoints[i][0]
            w[1] += alpha[i] * labels[i] * datapoints[i][1]

    # print "Non zero SVs:", non_zero_ids

    # line W*X+c
    # taking first non-zero vec
    rand_idx = random.randint(0, len(non_zero_ids) - 1)
    sv = datapoints[non_zero_ids[rand_idx]]
    # print sv
    # print w
    # print scalar_product(w, sv[:-1])
    # print 1/float(sv[-1])
    c = 1 / float(labels[non_zero_ids[rand_idx]]) - scalar_product(w, sv)
    # print c

    return w, c, alpha, sv_num



def problem13(gamma, N, M):

    m = 0
    err_in_svm_0_ct = 0
    err_in_skt_0_ct = 0
    while m < M:
        datapoints = gen.generate_dataset(N)
        #print datapoints[:3]
        labels = get_labels(datapoints)
        #w, b, alpha, sv_num = svm_kernel_cvxopt(datapoints, labels, rbf_kernel, gamma)
        w, b, alpha, sv_num = svm_qp(datapoints, labels, rbf_kernel, gamma)
        #print w
        print sv_num
        err_in_svm = calculate_err_svm(datapoints, labels, w, b)
        #print err_in_svm
        if err_in_svm == 0:
            err_in_svm_0_ct +=1



        model = svm_skt(datapoints, labels, gamma)
        err_in_skt = calculate_error_skt(datapoints, labels, model)

        print err_in_svm, err_in_skt, model.score(datapoints, labels)
        if err_in_skt == 0:
            err_in_skt_0_ct +=1

        m+=1

    print  "Err_in_0_% = ", err_in_svm_0_ct*100.0/float(M), "Err_in_skt_0_% = ", err_in_skt_0_ct*100.0/float(M)



def problem14_15(N, M, gamma, K):

    m = 0
    err_out_svm_avg = 0.0
    err_out_rbf_avg = 0.0
    beat_ct = 0
    while m < M:
        datapoints = gen.generate_dataset(N)
        labels = get_labels(datapoints)

        model = svm_skt(datapoints, labels, gamma)
        err_in_skt = calculate_error_skt(datapoints, labels, model)

        if err_in_skt > 0.0:
            print "[discarding run] SVM non-separable"
            continue

        mus = gen.generate_dataset(K)
        output = rbf(datapoints, labels, mus, gamma)
        # if output is None, skip
        if not output:
            continue
        w, centroids, F = output
        b = w[0]
        w = w[1:]
        #print w, b, centroids

        test_datapoints = gen.generate_dataset(200)
        test_labels = get_labels(test_datapoints)

        err_out_svm = 1.0 - model.score(test_datapoints, test_labels)
        err_out_svm_avg += err_out_svm

        err_out_rbf = calculate_err_rbf(test_datapoints, test_labels, w, b, gamma, centroids)
        err_out_rbf_avg += err_out_rbf

        print "Err_svm", err_out_svm, "Err_rbf", err_out_rbf

        if err_out_svm < err_out_rbf:
            beat_ct +=1

        m +=1

    print beat_ct/float(M)


# K2 > K1
def problem16(N, M, gamma, K1, K2):

    a_ct = 0
    b_ct = 0
    c_ct = 0
    d_ct = 0
    e_ct = 0

    print "err_in_k1 err_in_k2 err_out_k1 err_out_k2     A     B     C     D     E"
    m = 0
    while m < M:
        datapoints = gen.generate_dataset(N)
        labels = get_labels(datapoints)

        #for K1
        mus1 = gen.generate_dataset(K1)
        output1 = rbf(datapoints, labels, mus1, gamma)
        # if output is None, skip
        if not output1:
            continue
        w1, centroids1, F1 = output1
        b1 = w1[0]
        w1 = w1[1:]
        # print w, b, centroids

        err_in_k1 = calculate_err_rbf(datapoints, labels, w1, b1, gamma, centroids1)


        #for K2
        mus2 = gen.generate_dataset(K2)
        output2 = rbf(datapoints, labels, mus2, gamma)
        # if output is None, skip
        if not output2:
            continue
        w2, centroids2, F2 = output2
        b2 = w2[0]
        w2 = w2[1:]
        # print w, b, centroids

        err_in_k2 = calculate_err_rbf(datapoints, labels, w2, b2, gamma, centroids2)

        test_datapoints = gen.generate_dataset(200)
        test_labels = get_labels(test_datapoints)

        err_out_k1 = calculate_err_rbf(test_datapoints, test_labels, w1, b1, gamma, centroids1)
        err_out_k2 = calculate_err_rbf(test_datapoints, test_labels, w2, b2, gamma, centroids2)

        A = (err_in_k1 > err_in_k2) and (err_out_k1 < err_out_k2)
        B = (err_in_k1 < err_in_k2) and (err_out_k1 > err_out_k2)
        C = (err_in_k1 < err_in_k2) and (err_out_k1 < err_out_k2)
        D = (err_in_k1 > err_in_k2) and (err_out_k1 > err_out_k2)
        E = (err_in_k1 == err_in_k2) and (err_out_k1 == err_out_k2)

        print "{:9.3f} {:9.3f} {:10.3f} {:10.3f} {} {} {} {} {}".format(err_in_k1, err_in_k2, err_out_k1, err_out_k2, A, B, C, D, E)

        if A:
            a_ct +=1
        elif B:
            b_ct +=1
        elif C:
            c_ct +=1
        elif D:
            d_ct +=1
        elif E:
            e_ct +=1

        m += 1

    print "As: ",  a_ct/float(M), "Bs: ", b_ct/float(M), "Cs: ", c_ct/float(M), "Ds: ", d_ct/float(M), "Es: ", e_ct/float(M)

# gamma2 > gamma1
def problem17(N, M, gamma1, gamma2, K):

    a_ct = 0
    b_ct = 0
    c_ct = 0
    d_ct = 0
    e_ct = 0

    print "err_in_k1 err_in_k2 err_out_k1 err_out_k2     A     B     C     D     E"
    m = 0
    while m < M:
        datapoints = gen.generate_dataset(N)
        labels = get_labels(datapoints)

        #for gamma1
        mus = gen.generate_dataset(K)
        output1 = rbf(datapoints, labels, mus, gamma1)
        # if output is None, skip
        if not output1:
            continue
        w1, centroids1, F1 = output1
        b1 = w1[0]
        w1 = w1[1:]
        # print w, b, centroids

        err_in_1 = calculate_err_rbf(datapoints, labels, w1, b1, gamma, centroids1)


        #for gamma2
        #mus2 = gen.generate_dataset(K)
        output2 = rbf(datapoints, labels, mus, gamma2)
        # if output is None, skip
        if not output2:
            continue
        w2, centroids2, F2 = output2
        b2 = w2[0]
        w2 = w2[1:]
        # print w, b, centroids

        err_in_2 = calculate_err_rbf(datapoints, labels, w2, b2, gamma, centroids2)

        test_datapoints = gen.generate_dataset(200)
        test_labels = get_labels(test_datapoints)

        err_out_1 = calculate_err_rbf(test_datapoints, test_labels, w1, b1, gamma1, centroids1)
        err_out_2 = calculate_err_rbf(test_datapoints, test_labels, w2, b2, gamma2, centroids2)

        A = (err_in_1 > err_in_2) and (err_out_1 < err_out_2)
        B = (err_in_1 < err_in_2) and (err_out_1 > err_out_2)
        C = (err_in_1 < err_in_2) and (err_out_1 < err_out_2)
        D = (err_in_1 > err_in_2) and (err_out_1 > err_out_2)
        E = (err_in_1 == err_in_2) and (err_out_1 == err_out_2)

        print "{:9.3f} {:9.3f} {:10.3f} {:10.3f} {} {} {} {} {}".format(err_in_1, err_in_2, err_out_1, err_out_2, A, B, C, D, E)

        if A:
            a_ct +=1
        elif B:
            b_ct +=1
        elif C:
            c_ct +=1
        elif D:
            d_ct +=1
        elif E:
            e_ct +=1

        m += 1

    print "As: ",  a_ct/float(M), "Bs: ", b_ct/float(M), "Cs: ", c_ct/float(M), "Ds: ", d_ct/float(M), "Es: ", e_ct/float(M)

#RBF, E_in == 0
def problem18(N, M, gamma, K):

    m = 0
    err_in_0_ct = 0.0
    while m < M:
        datapoints = gen.generate_dataset(N)
        labels = get_labels(datapoints)

        mus = gen.generate_dataset(K)
        output = rbf(datapoints, labels, mus, gamma)
        if not output:
            continue
        w, centroids, F = output
        b = w[0]
        w = w[1:]
        #print w, b, centroids

        err_in = calculate_err_rbf(datapoints, labels, w, b, gamma, centroids)

        #print err_in

        if err_in == 0.0:
            err_in_0_ct +=1

        m +=1

    print "Fraction of Err_in == 0:", err_in_0_ct/float(M)

'''

For function: sign(x2-x1+0.25cos(pi*x1))

Checkpoints:

- centroids after Lloyd:
[[ 0.61758517, 0.58378869], [ 0.7368476, -0.13679382], [-0.81081814, -0.4964405 ], [ 0.24210492, -0.12446564], [-0.15233168, -0.4063166 ], [ 0.57386114, -0.79290577], [-0.35568478, 0.77926568], [-0.34401819, -0.86867122], [-0.62234109, 0.16090516]]
[array([ 0.61758517,  0.58378869]), array([ 0.7368476 , -0.13679382]), array([-0.81081814, -0.4964405 ]), array([ 0.24210492, -0.12446564]), array([-0.15233168, -0.4063166 ]), array([ 0.57386114, -0.79290577]), array([-0.35568478,  0.77926568]), array([-0.34401819, -0.86867122]), array([-0.62234109,  0.16090516])]

Resulting weights for regular RBF (includes w0):
w = [ 0.41798458 -0.23746045 -2.20524644 1.20854356 0.93871109 0.40556867 -0.57351241 0.73365732 -2.51586377 -0.05725362]

phi[0] = [1, 0.14481508599990736, 0.24004587995597113, 0.26238895063335149, 0.78187557968772847, 0.8062618698136611, 0.14417495193383784, 0.20060831056436523, 0.25854621285767554, 0.52168402210998177]
       [1.0, 0.14481508548913302, 0.24004588176606337, 0.26238895389801237, 0.78187557938535457, 0.80626187026702045, 0.14417495134281133, 0.20060831194208667, 0.2585462105079851, 0.521684026681279]
Resham's  [1,  0.14481508548913302, 0.24004588176606337, 0.26238895389801237, 0.7818755793853546, 0.8062618702670205, 0.14417495134281133, 0.20060831194208667, 0.2585462105079851, 0.521684026681279]


Ein_svm = 0
Ein_rbf = 0.032

'''
def ghost_problem(gamma):

    i = 0
    max_i = 250
    fin = open('./data/out.dta')
    datapoints = []
    while i < max_i:
        s = fin.readline()
        tokens = s.split()
        datapoints.append([float(tokens[0]), float(tokens[1])])
        i+=1

    labels = get_labels(datapoints)

    model = svm_skt(datapoints, labels, gamma)
    err_in_svm = calculate_error_skt(datapoints, labels, model)

    mus = [[ 0.86966258, 0.40174556],
           [ 0.96628889, 0.33046748],
           [-0.80634808, -0.39875312],
           [ 0.34387146, -0.23656711],
           [-0.16891906, -0.40718149],
           [ 0.49933254, -0.49183645],
           [ 0.27494242, 0.00421283],
           [-0.08493553, -0.5694352 ],
           [-0.20150429, -0.28749611]]

    output = rbf(datapoints, labels, mus, gamma)
    if output is None:
        print "RBF returned None... Exiting..."
        return
    w, centroids, F = output
    print w, '\n Centroids: \n', centroids
    print w.shape, F[0].shape
    print F[0].dot(w)
    b = w[0]
    w = w[1:]
    print w, b
    err_in_rbf = calculate_err_rbf(datapoints, labels, w, b, gamma, centroids)

    print "Err_in_svm", err_in_svm, 'err_in_rbf', err_in_rbf


#training set size
N = 100
gamma = 1.5

#problem13(1.5, N, 3)



K = 9
#problem14(N, 500, gamma, K)
#problem14_15(N, 400, gamma, K=15)

#ghost_problem(gamma=2.0)

#problem16(N, 100, gamma, K1=9, K2=12)

#problem17(N, 100, gamma1=1.5, gamma2=2.0, K=9)

problem18(N, 500, gamma, K)

'''
datapoints = gen.generate_dataset(N)
mus = gen.generate_dataset(K)


datapoints = [[-3.0, 2.0],
[0.0, -1.0],
[0.0, 3.0],
[1.0, 2.0],
[3.0, -3.0],
[3.0, 5.0],
[3.0, 5.0]]

mus = [[1.0, -1.0], [3.0, 3.0], [2.0, -1.0]]

print "started lloyds clustering"
lloyds_clustering(datapoints, mus)
print "ended lloyds clustering"
'''


'''
[ 0.79162576 -2.5188124   1.51715392  2.10402809 -2.36815641  0.27428107 0.82041679 -0.73326411  0.76209378] -0.136526074405
'''





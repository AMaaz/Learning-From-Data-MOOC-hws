'''
project_name: hw8_softSVM
@author: alisazhila
date: 11/18/16
'''


from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

TR_DATA_FN = './data/features.train'
TEST_DATA_FN = './data/features.test'



#np.loadtxt(fname, dtype= < type 'float' >, comments = '#', delimiter = None, converters = None, skiprows = 0, usecols = None, unpack = False, ndmin = 0
def read_data(fname):
    dataset = np.loadtxt(fname)
    #print type(dataset)
    #print dataset[:3]
    #print dataset[:3,1:]
    #print dataset.item((1,))
    #print dataset.item((2,2))
    digits = dataset[:,0]
    datapoints = dataset[:,1:]
    #print len(digits)
    #print len(datapoints)
    return datapoints, digits


#label for 1-vs-all
def label_data_vs_all(digits, digit):
    labels = []
    for i in range(len(digits)):
        if digits[i] == digit:
            labels.append(1.0)
        else:
            labels.append(-1.0)
    return labels


#label for one-vs-another
def label_data_vs_one(digit1, digit2, digits, datapoints):
    labels = []
    new_datapoints = []
    for i in range(len(digits)):
        if digits[i] == digit1:
            labels.append(1.0)
            new_datapoints.append(datapoints[i])
        elif digits[i] == digit2:
            labels.append(-1.0)
            new_datapoints.append(datapoints[i])
    new_datapoints = np.array(new_datapoints)
    return labels, new_datapoints


'''
class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
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
def soft_svm_skt(datapoints, labels, my_C, my_Q):

    clf = SVC(C=my_C, kernel='poly', degree=my_Q)
    clf.fit(datapoints, labels)
    return clf


def calculate_error_svm_skt(datapoints, labels, model):
    err = 0.0
    N = len(datapoints)
    for i in range(N):
        if model.predict([datapoints[i]])*labels[i] <= 0:
            err+=1
    return err/float(N)



def experiment2_3(digit_set, my_C, my_Q):
    datapoints, digits = read_data(TR_DATA_FN)
    #print datapoints[:5], digits[:5]

    for digit in digit_set:
        labels = label_data_vs_all(digits, digit)
        #print labels[:4]
        model = soft_svm_skt(datapoints, labels, my_C, my_Q)
        err_in = calculate_error_svm_skt(datapoints, labels, model)
        print digit, err_in
        print "# svs = ", len(model.support_)


def experiment5(digit1, digit2, my_Cs, my_Q):
    datapoints, digits = read_data(TR_DATA_FN)
    print len(datapoints), "-> ",
    labels, datapoints = label_data_vs_one(digit1, digit2, digits, datapoints)
    print len(datapoints)

    test_datapoints, test_digits = read_data(TEST_DATA_FN)
    print len(test_datapoints)
    test_labels, test_datapoints = label_data_vs_one(digit1, digit2, test_digits, test_datapoints)
    print len(test_datapoints)
    print "C,       err_in,         err_out,          #SVs"
    for my_C in my_Cs:
        model = soft_svm_skt(datapoints, labels, my_C, my_Q)
        err_in = calculate_error_svm_skt(datapoints, labels, model)
        err_out = calculate_error_svm_skt(test_datapoints, test_labels, model)
        print my_C, err_in, err_out, len(model.support_)


def experiment6(digit1, digit2, my_Cs, my_Qs):
    datapoints, digits = read_data(TR_DATA_FN)
    print len(datapoints), "-> ",
    labels, datapoints = label_data_vs_one(digit1, digit2, digits, datapoints)
    print len(datapoints)

    test_datapoints, test_digits = read_data(TEST_DATA_FN)
    print len(test_datapoints)
    test_labels, test_datapoints = label_data_vs_one(digit1, digit2, test_digits, test_datapoints)
    print len(test_datapoints)

    for my_Q in my_Qs:
        print "======================"
        print "Q =", my_Q
        print "C,       err_in,         err_out,          #SVs"
        for my_C in my_Cs:
            model = soft_svm_skt(datapoints, labels, my_C, my_Q)
            err_in = calculate_error_svm_skt(datapoints, labels, model)
            err_out = calculate_error_svm_skt(test_datapoints, test_labels, model)
            print my_C, err_in, err_out, len(model.support_)


def experiment7_8(digit1, digit2, my_Cs, my_Q):
    datapoints, digits = read_data(TR_DATA_FN)
    print len(datapoints), "-> ",
    labels, datapoints = label_data_vs_one(digit1, digit2, digits, datapoints)
    print len(datapoints)

    #X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size = 0.1, random_state=0)

    C_hits = [0]*len(my_Cs)
    for i in range(1000):
        print i
        C_scores = [0]*len(my_Cs)
        for j in range(len(my_Cs)):
            my_C = my_Cs[j]
            clf = SVC(C=my_C, kernel='poly', degree=my_Q)
            #class sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
            cv = ShuffleSplit(n_splits=10, test_size=0.1)
            scores = cross_val_score(clf, datapoints, labels, cv=cv)
            #print scores
            err= 1.0-scores.mean()
            #print my_C, err
            C_scores[j] = err
        C_scores = np.array(C_scores)
        #print C_scores
        #print np.argmin(C_scores)
        #mins_idx = np.where(C_scores == C_scores.min())
        #print mins_idx
        C_hits[np.argmin(C_scores)]+=1
    print C_hits


def experiment9_10(digit1, digit2, my_Cs):
    datapoints, digits = read_data(TR_DATA_FN)
    print len(datapoints), "-> ",
    labels, datapoints = label_data_vs_one(digit1, digit2, digits, datapoints)
    print len(datapoints)

    test_datapoints, test_digits = read_data(TEST_DATA_FN)
    print len(test_datapoints)
    test_labels, test_datapoints = label_data_vs_one(digit1, digit2, test_digits, test_datapoints)
    print len(test_datapoints)
    print "C,       err_in,         err_out,          #SVs"
    for my_C in my_Cs:
        clf = SVC(C=my_C, kernel='rbf')
        model = clf.fit(datapoints, labels)
        err_in = calculate_error_svm_skt(datapoints, labels, model)
        err_in_2 = 1.0 - model.score(datapoints, labels)
        err_out = calculate_error_svm_skt(test_datapoints, test_labels, model)
        err_out_2 = 1.0 - model.score(test_datapoints, test_labels)
        print my_C, err_in, err_in_2, err_out_2, err_out, len(model.support_)




'''
digit_set_test = [6.0]

digit_set_2 = [0.0, 2.0, 4.0, 6.0, 8.0]
digit_set_3 = [1.0, 3.0, 5.0, 7.0, 9.0]
digit_set_4 = [0.0, 1.0]

#experiment2_3(digit_set_2, 0.01, 2)
#experiment2_3(digit_set_3, 0.01, 2)
#experiment2_3(digit_set_4, 0.01, 2)

Cs = [0.001, 0.01, 0.1, 1.0]
#experiment5(1.0, 5.0, Cs, 2)


Cs = [0.0001, 0.001, 0.01, 1.0]
Qs = [2, 5]
#experiment6(1.0, 5.0, Cs, Qs)

Cs = [0.0001, 0.001, 0.01, 0.1, 1.0]
#experiment7_8(1.0, 5.0, Cs, 2)


Cs = [0.01, 1.0, 100.0, np.power(10.0, 4), np.power(10.0, 6) ]
experiment9_10(1.0, 5.0, Cs)

'''
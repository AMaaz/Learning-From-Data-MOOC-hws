'''
project_name: exam_7-10_LinRegRegularized
@author: alisazhila
date: 11/23/16
'''


import hw6_2LinRegRegularized as rlr

#dataread
import hw8_softSVM as dr
import numpy as np

# @input:   type(input) 'numpy.ndarray', [(x1, x2),...]
# @output: type(input) 'numpy.ndarray', [(1, x1, x2),...]
def transform_add_x0(datapoints):
    #print len(datapoints)
    ones = np.ones((len(datapoints), 1))
    #print len(ones)
    new_datapoints = np.append(ones, datapoints, 1)
    #print type(new_datapoints)
    return new_datapoints


#z = (1, x1, x2, x1*x2, x1^2, x2^2)
def second_order_transform(datapoints):
    transformed_datapoints = np.empty((len(datapoints), 6))
    for i in range(len(datapoints)):
        datapoint = datapoints[i]
        transformed_datapoints[i][0] = 1.0           #1
        transformed_datapoints[i][1] = datapoint[0]  #x1
        transformed_datapoints[i][2] = datapoint[1]  #x2
        transformed_datapoints[i][3] = datapoint[0]*datapoint[1]  #x1*x2
        transformed_datapoints[i][4] = datapoint[0]*datapoint[0]  #x1^2
        transformed_datapoints[i][5] = datapoint[1]*datapoint[1]  #x2^2

    return transformed_datapoints


def problem7(digit_set, lambda_coef):
    datapoints, digits = dr.read_data(dr.TR_DATA_FN)
    #print datapoints[1][1]
    print datapoints[:5], digits[:5]
    print datapoints.shape

    #print type(datapoints)
    datapoints = transform_add_x0(datapoints)
    #print datapoints
    print datapoints[:5]
    print datapoints.shape

    for digit in digit_set:
        labels = dr.label_data_vs_all(digits, digit)
        #print labels[:4]

        w = rlr.linear_reg_w_weight_decay(datapoints, labels, lambda_coef)
        err_in = rlr.estimate_err(w, datapoints, labels)
        print digit, err_in


def problem8(digit_set, lambda_coef):
    datapoints, digits = dr.read_data(dr.TR_DATA_FN)
    #print datapoints[1][1]
    print datapoints[:5], digits[:5]
    print datapoints.shape

    datapoints = second_order_transform(datapoints)
    #print type(datapoints)
    #print datapoints[:5]
    #print datapoints.shape

    test_datapoints, test_digits = dr.read_data(dr.TEST_DATA_FN)
    test_datapoints = second_order_transform(test_datapoints)

    for digit in digit_set:
        labels = dr.label_data_vs_all(digits, digit)
        #print labels[:4]

        w = rlr.linear_reg_w_weight_decay(datapoints, labels, lambda_coef)

        test_labels = dr.label_data_vs_all(test_digits, digit)
        err_out = rlr.estimate_err(w, test_datapoints, test_labels)
        print digit, err_out



def problem9(digit_set, lambda_coef):
    datapoints, digits = dr.read_data(dr.TR_DATA_FN)
    datapoints_x0 = transform_add_x0(datapoints)
    datapoints_2nd_order = second_order_transform(datapoints)

    test_datapoints, test_digits = dr.read_data(dr.TEST_DATA_FN)
    test_datapoints_x0 = transform_add_x0(test_datapoints)
    test_datapoints_2nd_order = second_order_transform(test_datapoints)

    print "# | err_in_x0 | err_out_x0 | err_in_2nd_order | err_out_2nd_order | overfitting  | <=0.95 | >= 1.05"

    for digit in digit_set:
        labels = dr.label_data_vs_all(digits, digit)
        #print labels[:4]

        w_x0 = rlr.linear_reg_w_weight_decay(datapoints_x0, labels, lambda_coef)
        err_in_x0 = rlr.estimate_err(w_x0, datapoints_x0, labels)

        w_2nd_order = rlr.linear_reg_w_weight_decay(datapoints_2nd_order, labels, lambda_coef)
        err_in_2nd_order = rlr.estimate_err(w_2nd_order, datapoints_2nd_order, labels)


        test_labels = dr.label_data_vs_all(test_digits, digit)
        err_out_x0 = rlr.estimate_err(w_x0, test_datapoints_x0, test_labels)
        err_out_2nd_order = rlr.estimate_err(w_2nd_order, test_datapoints_2nd_order, test_labels)
        print "| ".join(str(d) for d in (digit, err_in_x0, err_out_x0, err_in_2nd_order, err_out_2nd_order, (err_out_2nd_order - err_in_2nd_order) > (err_out_x0 - err_in_x0), err_out_2nd_order <= 0.95*err_out_x0, err_out_2nd_order >= 1.05*err_out_x0))


def problem10(digit1, digit2, lambdas):
    datapoints, digits = dr.read_data(dr.TR_DATA_FN)
    labels, datapoints = dr.label_data_vs_one(digit1, digit2, digits, datapoints)
    datapoints_2nd_order = second_order_transform(datapoints)

    test_datapoints, test_digits = dr.read_data(dr.TEST_DATA_FN)
    test_labels, test_datapoints = dr.label_data_vs_one(digit1, digit2, test_digits, test_datapoints)
    test_datapoints_2nd_order = second_order_transform(test_datapoints)

    for lambda_coef in lambdas:
        w_2nd_order = rlr.linear_reg_w_weight_decay(datapoints_2nd_order, labels, lambda_coef)

        err_in_2nd_order = rlr.estimate_err(w_2nd_order, datapoints_2nd_order, labels)
        err_out_2nd_order = rlr.estimate_err(w_2nd_order, test_datapoints_2nd_order, test_labels)
        print lambda_coef, err_in_2nd_order, err_out_2nd_order,






digit_set = [5, 6, 7, 8, 9]
lambda_coef = 1
#problem7(digit_set, lambda_coef)

digit_set = [0, 1, 2, 3, 4]
#problem8(digit_set, lambda_coef)

digit_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#problem9(digit_set, lambda_coef)

lambdas = [0.01, 1]
problem10(1.0, 5.0, lambdas)
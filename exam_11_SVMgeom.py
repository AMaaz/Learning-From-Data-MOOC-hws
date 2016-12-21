'''
project_name: exam_11_SVMgeom
@author: alisazhila
date: 11/27/16
'''



x1 = [1.0, 0.0]
x2 = [0.0, 1.0]
x3 = [0.0, -1.0]
x4 = [-1.0, 0.0]
x5 = [0.0, 2.0]
x6 = [0.0, -2.0]
x7 = [-2.0, 0.0]

Y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]

X = [x1, x2, x3, x4, x5, x6, x7]

def transform(X):
    Z = []
    for datapoint in X:
        z1 = datapoint[1]*datapoint[1] - 2*datapoint[0]-1 # x2^2-2*x1-1
        z2 = datapoint[0]*datapoint[0] - 2*datapoint[1]+1 # x1^2-2*x2+1
        Z.append([z1, z2])
    return Z


Z = transform(X)
for z in Z:
    print z
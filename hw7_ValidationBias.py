# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:55:28 2016

@author: alisazhila
"""

# Finding expectation of a minimun of 2 random uniformly distributed variables  

import random 

e1_avg = 0 
e2_avg = 0
e_avg = 0 
N = 10
for i in range(N):
    e1 = random.uniform(0, 1)
    e2 = random.uniform(0, 1)
    e = min(e1, e2)
    #print e1, e2, e
    e1_avg+=e1
    e2_avg+=e2
    e_avg+=e

print e1_avg/float(N), e2_avg/float(N), e_avg/float(N)   



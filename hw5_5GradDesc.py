# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:40:09 2016

@author: alisazhila
"""

import numpy as np

ETA = 0.1
EPSILON = np.power(10.0, -14.0)
print EPSILON

u0 = 1.0
v0 = 1.0

def derivative_u(u, v): 
    return 2*(u*np.exp(v)-2*v*np.exp(-1*u))*(np.exp(v)+2*v*np.exp(-1*u))
    
def derivative_v(u, v): 
    return 2*(u*np.exp(v)-2*v*np.exp(-1*u))*(u*np.exp(v)-2*np.exp(-1*u))    
    
    
def e_in(u,v): 
    return (u*np.exp(v)-2*v*np.exp(-1*u))*(u*np.exp(v)-2*v*np.exp(-1*u))


def gradient_descent(u0, v0):
    u = u0 
    v = v0 
    iterations = 0
    while e_in(u,v) >= EPSILON: 
        u_new = u - ETA*derivative_u(u,v) 
        v_new = v - ETA*derivative_v(u,v)
        u, v = u_new, v_new
        iterations+=1
        #if iterations%1000 == 0:
        #    print e_in(u,v), iterations    
        if iterations > 50 : 
            break 
    return e_in(u,v), iterations, u, v      
            

def two_step_gradient_descent(u0, v0): 
    u = u0 
    v = v0 
    iterations = 0
    while iterations < 15: 
        u = u - ETA*derivative_u(u,v) 
        v = v - ETA*derivative_v(u,v)
        iterations+=1
    return e_in(u,v), iterations, u, v      
    


print "Final: ", two_step_gradient_descent(u0, v0)    
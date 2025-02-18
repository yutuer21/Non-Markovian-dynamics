%matplotlib inline
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import numpy as np
from scipy import linalg,integrate
from scipy.optimize import minimize
import random

import stocproc as sp
from stocproc import StocProc_FFT

def cal_fun(params):
    p1,p2,q1,q2 = params

    wf=np.array([p1,p2])   
    norm_wf = np.sqrt(np.sum(np.abs(wf)**2))
    wf = wf / norm_wf
    wr = wf.reshape(2**1,1)
    rhoa0 = wr @ np.transpose(np.conj(wr))
    rhoa = rhoa0.copy()

    wf=np.array([q1,q2])   
    norm_wf = np.sqrt(np.sum(np.abs(wf)**2))
    wf = wf / norm_wf
    wr = wf.reshape(2**1,1)
    rhob0 = wr @ np.transpose(np.conj(wr))
    rhob = rhob0.copy()


    trace_dists = []
    trace_dists_diff = []
    for i in range(tlen):

        rhoa[0,1]= rhoa0[0,1]*Alist1[i]
        rhoa[1,0]= rhoa0[1,0]*Alist1[i]
        
        rhob[0,1]= rhob0[0,1]*Alist1[i]
        rhob[1,0]= rhob0[1,0]*Alist1[i]

        dist_t = trace_distance(rhoa,rhob)
        trace_dists.append(dist_t)
        if i > 0:
            trace_dists_diff.append((trace_dists[i]-trace_dists[i-1]) / (tlist[i]-tlist[i-1])) 


    N_RH = 0.
    for i in range(len(trace_dists_diff)):
        if trace_dists_diff[i] > 0:
            N_RH += trace_dists_diff[i] * (tlist[i+1]-tlist[i])
            
    return N_RH



initial_params = [0.1]*4
bounds =[(-1.1,1.1)]*4

result = minimize(cal_fun, initial_params, bounds=bounds, method='Nelder-Mead') 

optimized_params = result.x
max_value = -result.fun

print(optimized_params)
print(max_value)


blist= np.arange(0.2,6,0.1)  
clist= np.arange(0.8,8,0.1) 
NRc=[]
for c in clist:
    NR=[]
    for b in blist:
        # c = 5
        a = 2
        # b=0.2

        t_fin=10     
        tlen=801
        tlist=np.linspace(0, t_fin, tlen) 


        def func2(x):
            return 4/np.pi/x**2*(1-np.cos(x*t))*a*b/(b**2+(x-c)**2)

        Alist1=[]
        t=0.
        for t in tlist:
            A,err = integrate.quad(func2,0,np.infty)
            Alist1.append(np.exp(-A))
            
        N_RH = cal_fun([1,1,1,-1])
        NR.append(N_RH)
    NRc.append(NR)










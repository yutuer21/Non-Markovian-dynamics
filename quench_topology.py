%matplotlib inline
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'


import numpy as np
from scipy import linalg
import time
import math
import random

import stocproc as sp
from stocproc import StocProc_FFT


sx=np.array([[0.,1],[1,0]])
sy=np.array([[0.,-1j],[1j,0]])
sz=np.array([[1.,0],[0,-1]])


def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]



def calnoise(tlist,xlist,ylist,strength,b,wc,nmc,ntype="Markovian"):

    _WC_ = wc
    def lsd(w):
        return b/(b**2 + (w - _WC_)**2)

    def lac(t):
        return np.exp(- b*np.abs(t) - 1j*_WC_*t)

    def cal_stp(t_fin, seed=10):
        stp = StocProc_FFT(spectral_density=lsd, t_max=t_fin, alpha=lac, intgr_tol=1e-2, \
                        intpl_tol=1e-2, seed=0, negative_frequencies=True)
        return stp
    
    stp = StocProc_FFT(spectral_density=lsd, t_max=t_fin, alpha=lac, intgr_tol=1e-2, \
                        intpl_tol=1e-2, seed=0, negative_frequencies=True)



    tlen=len(tlist)
    nx,ny = len(xlist),len(ylist)
    obsx,obsy,obsz=np.zeros([nx,ny,tlen]),np.zeros([nx,ny,tlen]),np.zeros([nx,ny,tlen])  
       
    mx=0.
    my=0. 
    mz=1.2
    t0=1
    tsox=0.2
    tsoy=0.2         

  

    for ix in range(len(xlist)):
        for iy in range(len(ylist)):   
            kx = xlist[ix]
            ky = ylist[iy]
            H0=(mz-t0*math.cos(kx)-t0*math.cos(ky))*sz
            Hso=(mx+tsox*math.sin(kx))*sx+(my+tsoy*math.sin(ky))*sy
            Htp=H0+Hso
        
            for imc in range(nmc):
                noise=np.zeros([3,tlen])
                if ntype=="Markovian":
                    gg=[]
                    for i in range(tlen):
                        gg.append(random.gauss(0,np.sqrt(strength)))
                    noise[2,:] = gg
                elif ntype=="NMarkovian":
#                     stp1x=cal_stp(tlist[-1],seed=30)
                    stp.new_process()
                    noise[2,:] = np.real(stp(tlist))*np.sqrt(2*strength)
               
                    
                    
                wf=np.array([1,0])   
                wr = wf.reshape(2**1,1)
                rho = wr @ np.transpose(np.conj(wr))
                obsx[ix,iy,0]=obsx[ix,iy,0]+np.real(np.trace(rho @ sx))/nmc              
                obsy[ix,iy,0]=obsy[ix,iy,0]+np.real(np.trace(rho @ sy))/nmc      
                obsz[ix,iy,0]=obsz[ix,iy,0]+np.real(np.trace(rho @ sz))/nmc  

                for it in np.arange(tlen-1):
                    B=linalg.expm(-1j*tlist[1]*(Htp+noise[0,it]*sx+noise[1,it]*sy+noise[2,it]*sz))
                    wf=np.matmul(B, wf)
                    
                    wr = wf.reshape(2**1,1)
                    rho = wr @ np.transpose(np.conj(wr))
                    obsx[ix,iy,it+1]=obsx[ix,iy,it+1]+np.real(np.trace(rho @ sx))/nmc              
                    obsy[ix,iy,it+1]=obsy[ix,iy,it+1]+np.real(np.trace(rho @ sy))/nmc      
                    obsz[ix,iy,it+1]=obsz[ix,iy,it+1]+np.real(np.trace(rho @ sz))/nmc 
                    

    return obsx,obsy,obsz





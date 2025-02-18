%matplotlib inline
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import numpy as np
import time

import stocproc as sp
from stocproc import StocProc_FFT

import tensorcircuit as tc
from toqito.state_ops import pure_to_mixed

K = tc.set_backend("jax")

def charge_imbalance(n,s):
    c = tc.Circuit(n, inputs=s)
    os = tc.array_to_tensor([c.expectation_ps(z=[i]) for i in range(n)])
    return (K.sum(os[1::2]) - K.sum(os[0::2]))/n


def cutlist(keep):
    allq=[i for i in range(n)]
    for i in keep:
        allq.remove(i)
    return allq
    

def calvon(s,cut):   # 可以是 state或rho
    rho = tc.quantum.reduced_density_matrix(s, cut)   # get the redueced density matrix, where cut list is the index to be traced out 
    ee = tc.quantum.entropy(rho)
    return ee



def calmutual(s,i,j):
    allq=[i for i in range(10)]
    Si = calvon(s,cutlist([i]))
    Sj = calvon(s,cutlist([j]))
    Sij = calvon(s,cutlist([i,j])) 
    I=1/2.*(Si+Sj-Sij)
    return I

def mutual(rho,n):
    r1=calvon(rho,list(range(n))[0:int(n/2)]) 
    r2=calvon(rho,list(range(n))[int(n/2)::]) 
    return (r1+r2-calvon(rho,[]))/2.



def stor(n,s):
    c = tc.Circuit(n, inputs=s)
    state = c.state()
    wr = state.reshape(2**n,1)
    rho = wr @ K.transpose(K.conj(wr))
    return rho



def R3_negtive(rho):
    # Allow the user to input either a pure state vector or a density matrix.
    rho = pure_to_mixed(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))


    dim = np.array([round_dim])
    dim = dim.T


    dim = [int(x) for x in dim]
    from toqito.channels import partial_transpose

    pt= partial_transpose(rho, [1], dim)

    return -np.log2(np.trace(pt @ pt @ pt)/np.trace(rho @ rho @ rho))


def log_negativity(rho):
    rho = pure_to_mixed(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))


    dim = np.array([round_dim])
    dim = dim.T


    dim = [int(x) for x in dim]
    from toqito.channels import partial_transpose

    pt= partial_transpose(rho, [1], dim)


    return np.log2(np.linalg.norm(pt, ord="nuc"))


def main_full(n, tlistadj, v, nMBL=1, nbath=1, w_dephasing=0, b=1,clean=True, nonmar=True, mar=True):
    
    print("strength",w_mbl,w_dephasing,"_b",b)
    
#     tlist=np.linspace(0, tt, tlen) 
    tt=tlistadj[-1]
    tlen = len(tlistadj)
    tlist=[round(i,ndigits=1) for i in np.linspace(0, tt, int(tt*10+1)).tolist()]   # full time list
  
    
    ps = []
    w = []
    for i in range(n - 1):
        ps.append(tc.quantum.xyz2ps({"x": [i, i + 1]}, n=n))
        w.append(1.0)
        ps.append(tc.quantum.xyz2ps({"y": [i, i + 1]}, n=n))
        w.append(1.0)
        ps.append(tc.quantum.xyz2ps({"z": [i, i + 1]}, n=n))
        w.append(v)
    hbase = tc.quantum.PauliStringSum2COO(ps, w)
    hzs = [
        tc.quantum.PauliStringSum2COO([tc.quantum.xyz2ps({"z": [i]}, n=n)])
        for i in range(n)
    ]



    @K.jit
    def evol_baseline(w_mbl_inst, w_dephasing_inst):
        c = tc.Circuit(n)
        c.x(range(0, n, 2))
        s = c.state()
        rlist = [stor(n,s)]
        slist = [s]
        for i in range(tlen-1):
            dt=tlistadj[i+1]-tlistadj[i]
            tsite = tlist.index(round(tlistadj[i],ndigits=1))
            w = w_mbl_inst + w_dephasing_inst[tsite]
            h = hbase
            for j in range(n):
                h += w[j] * hzs[j]
            hd = K.to_dense(h)
            s = K.expm(-1.0j * dt * hd) @ K.reshape(s, [-1, 1])[:, 0]
            rlist.append(stor(n,s))
            slist.append(s)

        return  K.stack(rlist)   # stack 把list的内容粘成tensor
    
    
    def cal_enrtopy(rhotime):
        entlogneg = []
        for rr in rhotime:
#             entlogneg.append(log_negativity(rr))
            entlogneg.append(calvon(rr,list(range(int(n/2)))))
#             entlogneg.append(mutual(rr,n))
#             entlogneg.append(R3_negtive(rr))

        return K.stack(entlogneg)
    
        
    
    


    _WC_ = 5
    def lsd(w):
        return b/(b**2 + (w - _WC_)**2)

    def lac(t):
        return np.exp(- b*np.abs(t) - 1j*_WC_*t)


    def cal_stp(t_fin, seed=10):
        # print(seed)
        stp = StocProc_FFT(spectral_density=lsd, t_max=t_fin, alpha=lac, intgr_tol=1e-2, \
                        intpl_tol=1e-2, seed=seed, negative_frequencies=True)
        return stp

    stp_list=[]
    for gg in range(n):
        stp_list.append(cal_stp(tt,seed=gg))      
    #####


    entlognegsc = []
    entlognegsm = []
    entlognegsnm = []
    for i in range(nMBL):
#         w_mbl_instance = tc.array_to_tensor(np.random.normal(scale=np.sqrt(w_mbl**2/3.), size=[n]))
        w_mbl_instance = WMBL[i]
        


      
        if clean==True:
            w_dephasing_instance = tc.array_to_tensor(np.random.normal(scale=np.sqrt(0.0**2/3.), size=[len(tlist), n]))
            rlist = evol_baseline(w_mbl_instance, w_dephasing_instance) # 设定一个 H(t), 得到对应 rho(t)
            entlognegsc.append( cal_enrtopy(rlist)) 
            
            
        if nonmar==True:
            rhosnm = []
            for j in range(nbath):
                noise_n=[]
                for k in range(n):
                    stp=stp_list[k]
                    stp.new_process()
                    noise_n.append(np.real(stp(np.array(tlist))) *np.sqrt(2*(w_dephasing)**2/3.))
                w_dephasing_instance = K.transpose(K.stack(noise_n))    # [ntlist, nqubit]

                rlist = evol_baseline(w_mbl_instance, w_dephasing_instance)
                rhosnm.append(rlist)
                
            rhotimenm = K.mean(K.stack(rhosnm),axis=(0))   # 混态
            entlognegsnm.append( cal_enrtopy(rhotimenm)) 
            

        if mar==True:
            rhosm = []
            for j in range(nbath):
                w_dephasing_instance = tc.array_to_tensor(np.random.normal(scale=np.sqrt(w_dephasing**2/3.), size=[len(tlist), n]))


                start = time.time()
                rlist = evol_baseline(w_mbl_instance, w_dephasing_instance)
                end = time.time()
                if (i+j) == 1:
                    print("timem:",end-start)

                rhosm.append(rlist)
                
            rhotimem = K.mean(K.stack(rhosm),axis=(0))   # 混态
            entlognegsm.append( cal_enrtopy(rhotimem)) 
           
    if  clean==True:
        return K.mean(K.stack(entlognegsc),axis=(0)) 
    elif mar==True:
        return K.mean(K.stack(entlognegsm),axis=(0))
    else:
        return K.mean(K.stack(entlognegsnm),axis=(0))
        




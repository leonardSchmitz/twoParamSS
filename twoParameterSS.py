# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""

@author: leonardschmitz
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special
import time
import os


def evC(z,limit):
    S,T,d = z.shape
    ret = np.empty(shape=(S+1,T+1,d),dtype=z.dtype)
    ret[:S,:T,:] = z
    for j in range(d):
      ret[S,:,j] = limit[j]
      ret[:,T,j] = limit[j]
    return ret  

def evZ(z):
    S,T,d = z.shape
    return evC(z,np.zeros(d,dtype=z.dtype))

def Zero_a0(k,z):
    # Zero insertion on axis 0
    S,T,d = z.shape
    res = np.empty([S+1,T,d],dtype=z.dtype)
    res[:k,:,:] = z[:k,:,:]
    res[k,:,:] = np.zeros([T,d],dtype=z.dtype)
    res[(k+1):,:,:] = z[k:,:,:]
    return res

def Zero_a1(k,z):
    # Zero insertion on axis 1
    S,T,d = z.shape
    res = np.empty([S,T+1,d],dtype=z.dtype)
    res[:,:k,:] = z[:,:k,:]
    res[:,k,:] = np.zeros([S,d],dtype=z.dtype)
    res[:,(k+1):,:] = z[:,k:,:]
    return res

def warp_a0(k,z):
    # Zero insertion on axis 0
    S,T,d = z.shape
    res = np.empty([S+1,T,d],dtype=z.dtype)
    res[:(k+1),:,:] = z[:(k+1),:,:]
    res[(k+1):,:,:] = z[k:,:,:]
    return res

def warp_a1(k,z):
    # Zero insertion on axis 1
    S,T,d = z.shape
    res = np.empty([S,T+1,d],dtype=z.dtype)
    res[:,:(k+1),:] = z[:,:(k+1),:]
    res[:,(k+1):,:] = z[:,k:,:]
    return res

def delta(z):
    S,T,d = z.shape
    res = np.empty([S,T,d],dtype=z.dtype)
    for s in range(S-1):
      for t in range(T-1):
        res[s,t,:] = z[s+1,t+1,:] - z[s,t+1,:] - z[s+1,t,:] + z[s,t,:]
    #for s in range(S-1): 
    #  res[s,T-1,:] =  z[s+1,T-1,:] - z[s,T-1,:]
    #for t in range(T-1): 
    #  res[S-1,t,:] =  z[S-1,t+1,:] - z[S-1,t,:]
    #res[S-1,T-1] = z[S-1,T-1] - z_lim
    res[S-1,:,:] = 0 
    res[:,T-1,:] = 0 
    return res

def sigma(z):
    return np.flip(np.cumsum(np.cumsum(np.flip(z,axis=[0,1]),axis=0),axis=1),axis=[0,1])  

def NF_Zero(z):
    # assumes z created by evZ, i.e., z[S-1,:,:]=z[:,T-1,:]=0 for (S,T)=size(z) 
    res = z 
    S,T,d = z.shape
    s_rem = 0
    t_rem = 0
    for s in reversed(range(S-1)):
      if np.all(z[s,:,:]==0):
        s_rem += 1
        res[s:S-1,:,:] = res[(s+1):S,:,:]
    for t in reversed(range(T-1)):
      if np.all(z[:,t,:]==0):
        t_rem += 1
        res[:,t:T-1,:] = res[:,(t+1):T,:]
    return res[:(S-s_rem),:(T-t_rem),:]

def NF_warp(z):
    # assumes z created by evC, i.e., z[S-1,:,:]=z[:,T-1,:]=const for (S,T)=size(z) 
    res = z 
    S,T,d = z.shape
    s_rem = 0
    t_rem = 0
    for s in reversed(range(S-1)):
      if np.array_equal(z[s,:,:],z[s+1,:,:]):
        s_rem += 1
        res[s:S-1,:,:] = res[(s+1):S,:,:]
    for t in reversed(range(T-1)):
      if np.array_equal(z[:,t,:],z[:,t+1,:]):
        t_rem += 1
        res[:,t:T-1,:] = res[:,(t+1):T,:]
    return res[:(S-s_rem),:(T-t_rem),:]

def NF_kerDel(z):
    # assumes z created by evC, i.e., z[S-1,:,:]=z[:,T-1,:]=const for (S,T)=size(z) 
    S,T,d = z.shape
    return z - z[S-1,:,:]  # broadcasting (note that T>1)

def conc_diag_evZ(z,z2): 
    # assumes z,z2 created by evZ
    S,T,d = z.shape
    S2,T2,d2 = z2.shape
    ret = np.zeros(shape=(S+S2-1,T+T2-1,d),dtype=z.dtype)
    ret[:S-1,:T-1,:] = z[:S-1,:T-1,:]
    ret[S-1:,T-1:,:] = z2
    return ret

def conc_diag_evC(z,z2): 
    # assumes z,z2 created by evZ
    S,T,d = z.shape
    S2,T2,d2 = z2.shape
    ret = np.zeros(shape=(S+S2-1,T+T2-1,d),dtype=z.dtype)
    # make z larger and continue limit values
    ret[:S-1,:T-1,:] = z[:S-1,:T-1,:] 
    ret[S-1:,T-1:,:] = z[S-1,T-1,:]
    for s in range(S-1):
      for t in range(T-1,T+T2-1):
        ret[s,t,:] = z[s,T-1,:]
    for t in range(T-1):
      for s in range(S-1,S+S2-1):
        ret[s,t,:] = z[S-1,t,:]
    left_ret = NF_kerDel(ret)
    right_ret = z2
    for s in range(S-1):
      right_ret = warp_a0(0,right_ret)
    for t in range(T-1):
      right_ret = warp_a1(0,right_ret)
    return left_ret + right_ret


####### code for the sum signature. (highly inefficient) #############################

def naiv_eval_letters(_w_st,_i_s,_j_t,_z):
    # input: 1<=jk<=T
    # z in R^(SxTxd)
    # w_st is integer array (exponent vector)  
    S,T,d = _z.shape
    res = 1
    for i in range(0,d):
      res = res * (z[_i_s,_j_t,i]**(_w_st[i]))
    return res

def createDeltaSet(l,r,m):
    # input: m, 1<=l,r<=T
    # output: list of [j1,..,jm] such that l<j1<=..<=jm<r
    if m == 1:
        res = []
        for j in range(l+1,r+1):
            res.append([j])
        return res
    if m > 1:
        res = createDeltaSet(l,r,m-1).copy()
        res_new = []
        for w in res:
            for j in range(l+1,r+1):
                if w.count(j) == 0:
                    w_new = w.copy()
                    w_new.append(j)
                    w_new.sort()
                    if res_new.count(w_new) == 0:
                        res_new.append(w_new)
    return res_new.copy()

def SS_2Param_naiv(l_1,l_2,r_1,r_2,z,w):
    # INPUT: 
    # z in R^(SxTxd)
    # -1<=l_1<r_1<S
    # -1<=l_2<r_2<T
    # matrix composition represented by tensor w of shape (m,n,d)
    #     exponent vector w[s,t,:] for all (s,t)<=(m,n)
    # output: discrete iterated sum signature <ISS_lr(z),w>
    m,n,d = w.shape
    res = 0
    i_s = createDeltaSet(l_1,r_1,m)
    j_s = createDeltaSet(l_2,r_2,n)
    for i in i_s:
      for j in j_s: 
        res_one_fak = 1
        for s in range(m):
          for t in range(n):
            for k in range(d):  # improve code, this loop can be ommited  
              res_one_fak = res_one_fak*(z[i[s],j[t],k]**w[s,t,k]) 
        res = res + res_one_fak
    return res

def SS_naiv(z,w):
    S,T,d = z.shape
    return SS_2Param_naiv(-1,-1,S-1,T-1,z,w)

def SS_2Param_naiv_mat(z,w):
    S,T,d = z.shape
    res = np.ones(shape=(S,S,T,T))
    for l in range(S):
      for r in range(S):
        for l2 in range(T):
          for r2 in range(T):
            res[l,r,l2,r2] = SS_2Param_naiv(l-1,l2-1,r,r2,z,w)
    return res

################ efficient versions of ISS and intitalization ####################

def eval_letter(_z,_w):
   # INPUT 
   # _z in R^(SxTxd)
   # _w exponent vector in N_0^d 
   T1,T2,d = _z.shape
   ret = np.ones(shape=(T1,T2))
   for j in range(d):
     ret = ret*(_z[:,:,j]**_w[j])
   return ret

def np_forwardShift(_z,_a):
   res = np.roll(_z,1,axis=_a)
   if _a == 0:
     res[0,:] = 0
   if _a == 1:
     res[:,0] = 0
   return res
 
def np_backwardShift(_z,_a):
   res = np.roll(_z,-1,axis=_a)
   if _a == 0:
     res[len(_z[:,0])-1,:] = 0
   if _a == 1:
     res[:,len(_z[0,:])-1] = 0
   return res

def SS_wDiag(_z,_w):
   # INPUT:
   # z in R^(SxTxd)
   # w in R^(mxd) representing diag([1,...,d]^{w_{1,:}},...,)
   m,d = _w.shape
   #print(_w," has length", m)
   res = np.cumsum(np.cumsum(eval_letter(_z,_w[0,:]),axis=0),axis=1)
   for j in range(m-1):
     #res = np.roll(np.roll(res,1,axis=0),1,axis=1)
     #res[0,:] = 0
     #res[:,0] = 0
     res = np_forwardShift(np_forwardShift(res,0),1)
     res = np.cumsum(np.cumsum(res*eval_letter(_z,_w[j+1,:]),axis=0),axis=1)
   return res

def SS_wChain(_z,w_compr,a_compr):
   # due to Theorem 4.5
   # INPUT:
   # a_compr list of {-1,0,1} where -1 stands for axes 0 AND 1
   # w_compr in R^(len(a_compr)) with nonzero entries
   S,T,d = _z.shape
   l = len(a_compr)+1
   res = np.ones(shape=(S,T),dtype=_z.dtype)
   for t in range(l-1):
     if a_compr[t] == 0:
       res = np_forwardShift(np.cumsum(res*eval_letter(_z,w_compr[t,:]),axis=0),0)
     elif a_compr[t] == 1:
       res = np_forwardShift(np.cumsum(res*eval_letter(_z,w_compr[t,:]),axis=1),1)
     else:
       res = np.cumsum(np.cumsum(res*eval_letter(_z,w_compr[t,:]),axis=0),axis=1)
       res = np_forwardShift(np_forwardShift(res,0),1)
   return np.cumsum(np.cumsum(res*eval_letter(_z,w_compr[l-1,:]),axis=0),axis=1)


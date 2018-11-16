#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zhang
"""
# MAP estimations for Lomax delegate racing survival analysis with competing risks, implemented by pytorch. 
#%%
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from scipy import stats
import pickle
import time
from __future__ import division
from __future__ import print_function
import math
import random
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import scipy.io
#from keras.utils import to_categorical
import pandas as pd
import gc
#%%
dtype=torch.float
#dat=scipy.io.loadmat("/home/zhang/Dropbox/fall_2016/GIG/torch/linear.mat")
dat=scipy.io.loadmat("/home/zhang/Dropbox/fall_2016/GIG/torch/cosh.mat")
X=torch.tensor(dat['X'],dtype=dtype)
y=torch.tensor(dat['y'],dtype=torch.int)
y-=1 # y=98 if censored
n=y.shape[0]
p=X.shape[1]
s_t=torch.tensor(dat['s_t'],dtype=dtype)
eval_time=torch.tensor(dat['eval_time'],dtype=dtype)
Xt=torch.tensor(dat['Xt'],dtype=dtype)
yt=torch.tensor(dat['yt'],dtype=torch.int)
yt-=1 # y=98 if censored
s_tt=torch.tensor(dat['s_tt'],dtype=dtype)

## add censoring for linear data
#y[s_t>3.5]=98
#s_t[s_t>3.5]=3.5
# add censoring for cosh data
y[s_t>6.5]=98
s_t[s_t>6.5]=6.5
#%%
Is=(y!=98)*1. # censor indicator. 1 if uncensored
Is_m=Is.repeat(LL,1)
#Is.unsqueeze_(0);
r_random=True
SGD=False # stochastic gradient descent
batch_size=500 if SGD else y.shape[0]
sb=1
TT=2
mult=torch.zeros((y.shape[0],TT)) 
for i in range(n):
    if y[i]<98:
        mult[i,y[i]]=1.
    else:
        mult[i,:]=1.
mult.unsqueeze_(0); # dim=1*LL*n
mult.unsqueeze_(-1);
#%%
LL=50
K=5
a0=1.
b0=0.01
# set prior for c0_t, Gam(e0_t, 1/f0_t). For now, just set e0_t=0.01; f0_t=0.01 for all t
e0=0.01; f0=0.01
c0=0.01; gam0=K+1.
r_val=torch.tensor([1.]*K,dtype=dtype)  #(torch.tensor(range(K),dtype=dtype) +1.)/K
r=r_val.repeat((2,1))
r=torch.tensor(r, requires_grad=True)
#rr=r_val*torch.ones([TT,K,ns-burnin])
bet=torch.randn(p,TT,K) *2.
bet=torch.tensor(bet,requires_grad=True )
#bet=torch.zeros([p,TT,K],requires_grad=True)

#optimizer = torch.optim.SGD([bet], lr=0.1, weight_decay=0.0001)
Elog=False
optimizer_bet = torch.optim.Adam([bet], lr=0.001, weight_decay=0.0001)#)# weight_decay is to contol the l2 penalty on parameters
optimizer_r = torch.optim.Adam([r], lr=0.001, weight_decay=0.0001)# set weight_decay=0.0001 for numerical stability. You can also uncomment Line 110 to replace Line 109, but this might lead to numerical error.

for s in range(20000):
    ee=torch.distributions.gamma.Gamma(r, 1.)
    eta=ee.sample(sample_shape=torch.Size([LL,n]))# dim=LL*n*TT*K
    logeta=eta.log() # dim=LL*n*TT*K
    Xbet=torch.einsum('ip,pjk->ijk', (X,bet)) # dim=n*TT*K
    loglam=(Xbet+logeta) # dim=LL*n*TT*K
    loglam_yi=(loglam*mult).sum(2)# dim=LL*n*K
    loglam_yi[:,Is==0,:]=np.log(1./K)# dim=LL*n*K
    common=-s_t.unsqueeze(-1)*(loglam.exp().sum(2).sum(-1,keepdim=True))# dim=LL*n*1
    power=loglam_yi+common # dim=LL*n*K
    power_max=(power.max(0)[0]).max(-1,keepdim=True)[0]# dim=n*1
    loglh=torch.mean( power_max.squeeze(-1) + torch.log(torch.sum(torch.exp(power-power_max), dim=-1).mean(0) ) )
    loss_bet=-loglh
    # r
    # logp(eta|r)
    logp_eta=(-torch.lgamma(r)+(r-1.)*logeta-eta).sum(-1).sum(-1) # dim=LL*n
    ptiyi=(power-power_max).exp().sum(-1) #dim=LL*n
    loss_r=-( (ptiyi*logp_eta).mean(0) /(ptiyi.mean(0))  ).mean()
    #loss_r=-( (ptiyi*logp_eta).mean(0) /(ptiyi.mean(0))  ).mean() - ((gam0/K-1.)*r.log()-c0*r ).sum()
    
    optimizer_bet.zero_grad()
    loss_bet.backward(retain_graph=True)
    optimizer_bet.step()
    
    optimizer_r.zero_grad()
    loss_r.backward()
    optimizer_r.step()
    if (s+1) % 100 == 0:
        print ("iter[{}], Loss_bet: {:.4f}, Loss_r: {:.4f}".format(s+1, loss_bet.item(), loss_r.item()))


        
# calculate CIF and Brier score (BS)
nt=Xt.shape[0]
ee=torch.distributions.gamma.Gamma(r, 1.)
eta=ee.sample(sample_shape=torch.Size([100,nt]))
logeta=eta.log()
Xbet=torch.einsum('ip,pjk->ijk', (Xt,bet))
lam=(Xbet+logeta).exp()
lam_sum_k=lam.sum(-1)  
lam_sum_jk=lam_sum_k.sum(-1,keepdim=True)
pj=(lam_sum_k/lam_sum_jk).unsqueeze(-1)
cdf=1-(-eval_time*(lam_sum_jk.unsqueeze(-1))).exp() # dim=LL*n*1*len(eval_time)
CIF=(pj*cdf).mean(0) # dim=n*TT*len(eval_time)
Ity=torch.zeros(nt, TT, eval_time.shape[0])
for i in range(nt):
    for tt in range(eval_time.shape[0]):
        tau=eval_time[tt].item()
        if s_tt[i]<=tau:
            Ity[i,yt[i],tt]= 1 
            
BS=(Ity-CIF).pow(2).mean(0) # Brier Score







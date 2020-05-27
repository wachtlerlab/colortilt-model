# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:33:54 2020

@author: Ibrahim Alperen Tunc
"""

import cmath as c
import numpy as np
import matplotlib.pyplot as plt
import colclass as col
import sys
sys.path.insert(0,col.pathes.runpath)#!Change the directory accordingly
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict

#OLE TRY_1
#See Salinas&Abbot 1994

#Import a random model
model = col.colmod(None,2,None,stdInt=[2,1],bwType="gradient/sum",phase=0,avgSur=180,startAvg=1,endAvg=360,depInt=[0.2,0.6],depmod=True,stdtransform=False)

def OLE(unitactivity,x):#SOMETHING NOT RIGHT, DEBUG SOME TIME LATER
    #L_j=integral(stimulusVector*responseToStimulus)
    
    #unit activity array (n_units x n_stimang)
    unitactivity = np.squeeze(unitactivity) #so that the matrix notation can be used,
    
    #preallocate covariance matrix Q (n_units x n_units)
    Q = np.zeros([len(unitactivity),len(unitactivity)])
    
    #stimulus indexes for 0 degrees and 359.9 degrees
    idxst , idxend = (list(x).index(0),list(x).index(360))
    #print(idxend)
    #convert stimuli to vectors in complex plane (z=r*exp(i*stimang))
    stims = np.e**(1j * np.deg2rad(x[idxst:idxend]))
    #Compute the center of mass vector L
    L = np.sum(stims * unitactivity[:,idxst:idxend],1)
    for i in range(len(unitactivity)):
        for j in range(i,len(unitactivity)):
            #print(i,j)
            Q[i,j] = np.sum(unitactivity[i,idxst:idxend]*unitactivity[j,idxst:idxend])
            Q[j,i] = Q[i,j]
    
    D = np.sum(np.linalg.inv(Q)*L,1)
    
    #preallocate v.est (n_stims x 1)
    v_est = np.zeros(idxend-idxst)
    
    for i in range(idxst,idxend):
        #print(i)
        v_est[i-idxst] = np.rad2deg(c.phase(np.sum(unitactivity[:,i]*D)))    
        if v_est[i-idxst]<-1:
            v_est[i-idxst]+=360
    return v_est

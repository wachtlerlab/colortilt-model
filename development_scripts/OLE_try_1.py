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
from supplementary_functions import std2kappa, kappa2std, depth_modulator, plotter, param_dict

#OLE TRY_1
#See Salinas&Abbot 1994

#Import a random model
model = col.colmod(None,2,None,stdInt=[2,1],bwType="gradient/sum",phase=0,avgSur=180,startAvg=1,endAvg=360,depInt=[0.2,0.6],depmod=True,stdtransform=False)

kapMod=(2-1)/2*np.cos(2*np.deg2rad(np.linspace(1,360,360)-0))+1+(2-1)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py


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

unact = np.squeeze(model.centery) #all unit activities added here


#Preallocate L
L = np.ones(len(model.unitTracker))*1j
Langle = np.zeros(len(model.unitTracker))
#stimulus indexes for 0 degrees and 359.9 degrees
idxst , idxend = (list(model.x).index(0),list(model.x).index(360))
x = model.x[idxst:idxend]

stimuli = np.e**(1j*np.deg2rad(x))
angles=np.zeros(len(stimuli))
for i in range(angles.shape[0]):#works if angle is necessary, vector is now complex
    angles[i] = np.rad2deg(c.polar(stimuli[i])[1])
    if angles[i]<0:
        angles[i]+=360
               
for i in range(L.shape[0]):
    L[i] = np.sum(stimuli*unact[i,idxst:idxend])

for i in range(L.shape[0]):
    Langle[i] = np.rad2deg(c.phase(L[i]))
    if Langle[i]<0:
        Langle[i]+=360
#preallocate covariance matrix Q (n_units x n_units)
Q = np.zeros([len(unact),len(unact)])

for i in range(len(unact)):
    for j in range(len(unact)):
        if i!=j: 
            var = 0
        else: 
            var = np.deg2rad(kappa2std(kapMod[i])**2)
        #print(i,j)
        Q[i,j] = var+np.sum(unact[i,idxst:idxend]*unact[j,idxst:idxend])

#Q=Q/max(Q[0])

D = np.linalg.inv(Q)@L

#ri (popact) is the unit activity for the given stimulus!!!
popact = np.squeeze(col.decoder.nosurround(270,model.x,model.centery).noSur)

vest = np.sum(popact*D) 

vests = np.zeros(360)

for i in range(360):
    popact = np.squeeze(col.decoder.nosurround(i,model.x,model.centery).noSur)
    vest = np.sum(popact*D) 
    if np.rad2deg(c.phase(vest))<0:
        angle = np.rad2deg(c.phase(vest))+360
    else:
        angle = np.rad2deg(c.phase(vest))
    vests[i] = angle
plt.plot(np.arange(360),vests-np.arange(360)) #either i did something wrong or the OLE does not work properly here!

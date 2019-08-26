# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:41:26 2019

@author: Ibrahim Alperen Tunc
"""
"""
Old version of the circular_normal_curves_ML_updated.py. This script can be totally ignored, instead for a detailed explanation please
look at circular_normal_curves_ML_updated.py.
Returns the colortilt curve plot
"""
'''
Now decoder is maximum likelihood
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#!Change directory where colclass.py is.
from colclass import colmod
"""
Create wrapped normal distribution curve for center population
"""
Kcent=5#Kappa bigger means distribution more dense around peak,1/Kappa ananlogous to variance ?HOW TO IMPLEMENT SIMILAR TO SD
startAvg=1
endAvg=360
Ksur=2
avgSur=180
maxInhRate=0.5
colMod=colmod(Kcent,Ksur,maxInhRate,avgSur=avgSur,startAvg=startAvg,endAvg=endAvg)
'''
Make a look-up table for ML-decoding (the population activity for each stimulus angle without surround modulation)!!!! 
'''
def tab_look(stimulusAngle):
    tableML={}
    tableInd=[]
#stimulusAngle=100
    for i in range(1,len(colMod.unitTracker)+1):
        tableInd.append(colMod.centery[np.where(colMod.unitTracker==i)[0][0]][np.where(colMod.x==stimulusAngle)[0][0]])
    tableML.update({stimulusAngle:tableInd})
    return tableML

tabLook={}#population activity before surround modulation
for i in range(0,len(np.linspace(0,359,num=360))):#i here is the stimulusAngle (bin is 1 atm)
    tabLook.update({i:np.divide(tab_look(i)[i],sum(tab_look(i)[i]))})#creates a dictionary where key value is stimulus angle and corresponding value to key is population activity in stimulus angle
    #tabLook is normalized with area=1 to enable comparison with thing after modulation. (normalizing where peak=1 better approach?, or no normalization at all?)
    #dict.update works like list.append, adds dictionary value. Do it witohut normalization as well and compare 

def ML_decoder(stimulusAngle):
#stimulusAngle=0
    popInh=[]
    for i in range(1,len(colMod.unitTracker)+1):
        popInh.append(colMod.resulty[np.where(colMod.unitTracker==i)[0][0]][np.where(colMod.x==stimulusAngle)[0][0]])
    popInh=popInh/sum(popInh)#same normalization as tabLook.
    inhRMS=[]
    for i in range(0,len(tabLook)):
        inhRMS.append(np.mean(np.sqrt((np.subtract(tabLook[i],popInh))**2)))#root mean square to determine most similar curve gives hue shift after normalizing both curves!!!
    decodedAng=np.where(inhRMS==min(inhRMS))[0][0]#look at thecurve of RMS
    return decodedAng-stimulusAngle, stimulusAngle-avgSur, inhRMS #returns hue shift and center-surround difference

"""
Decoded parameters for different center stimuli hues.
"""
hueShift=[]
centSurDif=[]
difRMS=[]
for i in range(0,len(np.linspace(0,359,num=360))):
    hueShift.append(ML_decoder(i)[0])
    centSurDif.append(ML_decoder(i)[1])
    difRMS.append(ML_decoder(i)[2])

"""
Color tilt plot
"""
plt.figure()
plt.plot(centSurDif,hueShift,'--')
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')

'''
*Comments and notes during development, can be totally ignored.

PROBLEM: with table the angle bins reflect themselves as discrete points for decodedAng, interpolation/smoothing necessary. Decreasing bin size also problematic as 
computation takes too long for bin=.1 and it is still not smooth enough. below are some possible codes as a start (not necessary!)

Qualitative decoding profile sufficient in this moment 
-Plot RMS profiles for stimulus angle (done)
-try to reproduce klauke empirical data by playing model variables (Kappa,inh rate usw.) (kellner manuscript)
-look at the effects of parameters in decoders
-Try ML-decoder with and without normalizing the curves (look again at RMS behavior.) (done, the effect is way more subtle in case of normalization)
-Kellner manuscript figures to reproduce! (look closer again in paper)
-what happens when population code is not uniform?, modulation or tuning parameter change?
-global model describing all surround angle effects (function of kappa,phi,inh instead of kappa and inh are constants in a fixed surround angle!)
'''


'''
z = np.polyfit(centSurDif,hueShift, 5)
f = np.poly1d(z)

x_new = np.linspace(centSurDif[0], centSurDif[-1], 50)
y_new = f(x_new)

plt.plot(x_new, y_new)

import scipy.optimize.curve_fit as sp_op#a very nice possibility but i somehow need a function in advance as model!
sp_op.curvefit()
'''


# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:24:20 2019

@author: Ibrahim Alperen Tunc
"""
"""
Maximum likelihood decoder development script. The script is redundant, the decoder is implemented in colclass.py.
The uniform model is used with the surround stimulus hue angle fixed at 180°
Returns the plots of the color tilt curve as well as the minimum root mean square difference for different center stimuli conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#!Change the directory accordingly
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
Make a look-up table for ML-decoding (the population activity for each stimulus angle without surround modulation)
'''
def tab_look(stimulusAngle):
    """Lookup-table function:
        This function is to create the look-up table for the center stimulus. The dictionaries include the population activity for 
        each center stimulus before surround modulation.
        
        Parameters
        ----------
        stimulusAngle: float. The center hue angle, for which the population activity is wished to be sampled.
        
        Returns
        -------
        tableML: dictionary. The lookup table entry for the given center stimulus hue.
    """
    tableML={}
    tableInd=[]
#stimulusAngle=100
    for i in range(1,len(colMod.unitTracker)+1):
        tableInd.append(colMod.centery[np.where(colMod.unitTracker==i)[0][0]][np.where(colMod.x==stimulusAngle)[0][0]])
    tableML.update({stimulusAngle:tableInd})#values here are unit activities in given stimulus for each unit 
    return tableML

tabStep=10#number of hue angle entries to be skipped in colMod.x. colMod.x (the hue angle values from -60 to 420) has a spacing of 0.1°, so the lookup-table entries
          #have the angular space of 1° for each adjacent stimulus. 
tabLook={}#population activity before surround modulation
for i in range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],tabStep):#i here is the stimulusAngle (bin is 1 atm)
    tabLook.update(tab_look(colMod.x[i]))#creates a dictionary where key value is stimulus angle and corresponding value to key is population activity in stimulus angle
def ML_decoder(stimulusAngle,normalizer=False):
    """
    Maximum likelihood decoder:
        This function chooses the most similar lookup table entry for the given population activity after surround modulation. The similarity is quantified 
        by choosing the table entry having the least root mean square difference with the population activity after surround modulation.
        The decoded hue angle is the hue angle of the chosen entry.
        
        Parameters
        ----------
        stimulusAngle: float. The center stimulus hue angle which is desired to be decoded.
        normalizer: boolean, optional. If true, the lookup-table entries are normalized so the sum of each entry is (the population activity without surround
        modulation for a given center stimulus hue) 1.
        
        Returns
        -------
        stimulusAngle-decodedAng: float. The angular difference between the decoded and the real center stimulus hues in degrees.
        stimulusAngle-avgSur: float. The center-surround angle difference in degrees.
        distLUT: list. The root mean square difference between the lookup table entries and the population activity for a given center stimulus angle.
        inhVal: list. The difference between the lookup table entries and the population activity for a given center stimulus angle.
    """
    popInh=[]#population activity for the given center hue after surround modulation
    #stimulusAngle=130
    for i in range(1,len(colMod.unitTracker)+1):
        popInh.append(colMod.resulty[np.where(colMod.unitTracker==i)[0][0]][np.where(colMod.x==stimulusAngle)[0][0]])
    distLUT=[]#Root mean square difference between the population activity and the lookup table entries.
    if normalizer==False:
        for i in range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],tabStep):
            inhVal=np.asarray(tabLook[colMod.x[i]])-np.asarray(popInh)#Difference between the lookup-table entries and the population activity after surround modulation
            distLUT.append(np.sqrt(np.mean(np.asarray(inhVal)**2)))
    else:
        for i in range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],tabStep):
            inhVal=np.asarray(tabLook[colMod.x[i]])/sum(tabLook[colMod.x[i]])-np.asarray(popInh)/sum(popInh)
            distLUT.append(np.sqrt(np.mean(np.asarray(inhVal)**2)))#both areas of activity before modulation and after modulation is normalized to 1 
    #plt.plot(distLUT)
    decodedAng=colMod.x[range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],tabStep)][np.where(np.asarray(distLUT)==min(distLUT))[0]]
    return decodedAng-stimulusAngle, stimulusAngle-avgSur, distLUT, inhVal #returns hue shift and center-surround difference
 
"""
Create the lists of decoder parameter values for a subset of center stimuli hue angles.
"""
hueShift=[]#induced hue shift
centSurDif=[]#center-surround stimuli hue difference in degrees
MLindex=[]#Root mean square difference
for i in range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],tabStep):
    hueShift.append(ML_decoder(colMod.x[i])[0])
    centSurDif.append(ML_decoder(colMod.x[i])[1])
    MLindex.append(ML_decoder(colMod.x[i])[2])

"""
Plot the color tilt curve and the minimum RMS difference for each center stimulus angle.
"""
plt.figure()
plt.plot(centSurDif,hueShift,'--')
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')
plt.figure()
minML=[]
for i in range(0,len(MLindex)):
    minML.append(min(MLindex[i]))
plt.plot(minML)
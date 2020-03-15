# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:33:28 2019

@author: Ibrahim Alperen Tunc
"""
"""
Function version of model_tester.py, can be rather useful.
The function version of the circular_normal_curves.py
colclass is additionally used to create the model (colclass.py)

Used packages: numpy, matplotlib, cmath, sys
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath as c
from colclass import colmod,pathes
import sys
sys.path.insert(0,pathes.runpath)#!CHANGE THE DIRECTORY WHERE colclass.py is residing.

"""
Create wrapped normal distribution curve for center population ?WHAT I DO HERE IS ACTUALLY TAKING THE MEAN OF POPULATION IN THE DOMAIN OF CIRCULAR STATISTICS?
"""
def model_decoder(Kcent,Ksur,maxInhRate,plotStep=10):#Kappa bigger means distribution more dense around peak,1/Kappa ananlogous to variance ?HOW TO IMPLEMENT SIMILAR TO SD
    """Model test function:
        Enables to try different model parameters for the unifor model and surround stimulus angle=180
        
        Parameters
        ----------
        Kcent: float. Kappa value of the center unit tuning curves 
        Ksur: float. Kappa value of the surround suppression curve
        maxInhRate: float. Maximum value of the surround suppression curve. values between 0,1
        plotStep: integer, optional. Number of units to be skipped for the plot of center unit tuning curves
        
        Returns
        -------
        Plots of the center units before and after surround modulation as well as color tilt curve (with population vector decoder)"""
    
    avgS=180#Surround stimulus angle is set to 180 for the test
    colMod=colmod(Kcent,Ksur,maxInhRate,avgSur=avgS)
    '''
    now +surround
    '''
    def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al.).
        surDecoder=[]
        for i in range(0,len(colMod.resulty),unitStep):
            surDecoder.append(colMod.resulty[i][np.where(colMod.x==stimulusAngle)[0][0]])
        popSurVector=[]
        for i in range(0,len(surDecoder)):
            popSurVector.append(surDecoder[i]*np.e**(1j*np.deg2rad(colMod.unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
            #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
            
        np.sum(popSurVector)#vector sum of neural population
        decodedAngleSur=np.rad2deg(c.phase(np.sum(popSurVector)))
        if decodedAngleSur<0:
            decodedAngleSur=360+decodedAngleSur #transform the negative angles into angles>180.
        return decodedAngleSur-stimulusAngle , stimulusAngle-avgS #returns the shift of decoded angle and center-surround difference
    """
    Induced angular shift against the center-surround discrepancy
    """
    centSurDif=[]
    angShift=[]
    for i in range(0,len(colMod.unitTracker)):
       angShift.append(vector_decoder(stimulusAngle=colMod.unitTracker[i])[0])
       centSurDif.append(vector_decoder(stimulusAngle=colMod.unitTracker[i])[1])   
    
    """   
     Plottings all together down here ;)
    """
    def plotter():
        for i in range(0,len(colMod.centery),plotStep):#range of 90-270 also possible
            plt.plot(colMod.x,colMod.centery[i])
        plt.title('Tuning curves of center units before surround effect')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity')
        
        plt.figure()
        plt.plot(colMod.x,colMod.surroundy)
        plt.title('effect of surround angle (180) on center units')
        plt.xlabel('Preferred Unit Angle')
        plt.ylabel('Neural activity inhibition rate')
        
        plt.figure()
        for i in range(0,len(colMod.resulty),plotStep):
            plt.plot(colMod.x,colMod.resulty[colMod.unitTracker[i]-2])#-2 is to plot maximally inhibited curve as well (resulty[179])
        plt.title('Tuning curves of center units after surround effect')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity')
        
        plt.figure()
        plt.plot(centSurDif,angShift)
        plt.plot([-180,180],[0,0])
        plt.title('Angular shift relative to center-surround angle difference')
        plt.ylabel('Difference between perceived and real angles')
        plt.xlabel('Center-surround difference')
        return
    return plotter()
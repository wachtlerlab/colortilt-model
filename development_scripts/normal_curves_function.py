# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:12:54 2019

@author: Ibrahim Alperen Tunc
"""
"""
Function version of the normal_curves.py, can be totally ignored. The decoder used here is rather nonsensical, was thought by myself :)
"""
import numpy as np
import matplotlib.pyplot as plt


def hue_shifter(stCen,stSur,startAvg=90,plotStep=10):
    """
    Tester function of the model with linear normal distribution:
        Makes various plots for the model of the given parameters.
        
        Parameters
        ----------
        stCen: float. Standard deviation of the center unit tuning curves.
        stSur: float. Standard deviation of the surround modulation curve.
        startAvg: integer, optional. Preferred hue angle of the first center unit. smallest value should be 0
        plotstep: integer, optional. Number of units to be skipped for the center unit tuning curve plots.
        
        Returns
        -------
        Plots of the color tilt curve as well as center unit tuning curves before and after surround modulation.
        
        WARNING
        -------
        The model shows edge effects, that the decoded color tilt values are extremely high at the both positive and negative edges.
        
        See also
        --------
        normal_curves_function_03.05.py
    """
    
    """
    Create center units
    """
    x=np.linspace(0,360,361)
    centery=[]
    totalavg=[]
    for i in range(0,181):
        avg=startAvg+i
        totalavg.append(avg)
        y=1/np.sqrt(2*np.pi*stCen**2)*np.e**(-(x-avg)**2/(2*stCen**2))
        centery.append(y)    
    """
    Create the surround effect curve and combine with center population
    """
    avgSur=180#Surround stimulus is fixed at 180° for testing
    surroundy=1/np.sqrt(2*np.pi*stSur**2)*np.e**(-(x-avgSur)**2/(2*stSur**2))*100#!ASK HERE if *100 makes sense, else the effect is very subtle
    resulty=[]
    for i in range(0,len(centery)):
        ymodulated=centery[i]*(1-surroundy)
        resulty.append(ymodulated)
    """
    Take the population activity readout for the given angle
    """
    unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example
    def angular_difference(chosenUnit):
        activityUnit=[]
        popTotal=[]
        for i in range(0,len(centery)):
            activityUnit.append(resulty[i][np.arange(chosenUnit-int(round(stCen,0)),chosenUnit+int(round(stCen,0))+1)])
            unitReadout=sum(activityUnit[i][:])
            popTotal.append(unitReadout)#this code makes not much sense, try to make it based on angles
        
        popTotalNormalized=popTotal/max(popTotal)
        
        popReadout=[]
        for i in range(0,len(popTotal)):
            c=popTotalNormalized[i]*(unitTracker[i]-unitTracker[np.where(unitTracker==chosenUnit)[0][0]])
            popReadout.append(c)
        
        
        arealSum=sum(popReadout[np.where(unitTracker==chosenUnit)[0][0]-int(round(stCen,0)):np.where(unitTracker==chosenUnit)[0][0]+int(round(stCen,0))+1])
        return arealSum , chosenUnit-avgSur#gives you the shift and center-surround difference
    
    """
    Induced angular shift against the center-surround discrepancy
    """
    centSurDif=[]
    angShift=[]
    for i in range(int(round(stCen,0)),len(unitTracker)-int(round(stCen,0))):
        ydif=angular_difference(chosenUnit=unitTracker[i])[0]
        xdif=angular_difference(chosenUnit=unitTracker[i])[1]
        centSurDif.append(xdif)
        angShift.append(ydif)
        
        
    """   
     Plottings all together down here ;)
    """
    def plotter():    
        for i in range(0,len(centery)+1,plotStep):
            plt.plot(x,centery[i])
        plt.title('Tuning curves of center units before surround effect')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity')
        
        plt.figure()
        plt.plot(x,surroundy)
        plt.title('effect of surround angle (180) on center units')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity inhibition rate')
        
        plt.figure()
        for i in range(0,len(resulty)+1,plotStep):
            plt.plot(x,resulty[i])
        plt.title('Tuning curves of center units after surround effect')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity')
         
        plt.figure()
        plt.plot(centSurDif,angShift)
        plt.plot([-70,70],[0,0])
        plt.title('Angular shift relative to center-surround angle difference')
        plt.ylabel('Difference between perceived and real angles')
        plt.xlabel('Center-surround difference')
        return
    
    return plotter()
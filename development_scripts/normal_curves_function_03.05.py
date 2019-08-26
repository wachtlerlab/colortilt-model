# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:12:54 2019

@author: Ibrahim Alperen Tunc
"""
"""
Model function created by Gaussian distribution, can be ignored, population vector decoder is used.
There are bugs regarding the plot outputs, such that color tilt curve has edge effects, which is why von Mises
distribution is used. Debugging process up until creating the colclass.py script
Returns plots of color tilt curve, center unit tuning curves before and after surround modulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath as c

def hue_shifter(stCen,stSur,startAvg=90,plotStep=10):
    """Model test function with linear normal distribution:
        Makes various plots for the model of the given parameters
        
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
    """
    """
    Create normal distribution curve for center population
    """
    avgSur=180#Surround stimulus hie angle is set to 180° for the test
    x=np.linspace(0,360,361)
    centery=[]
    totalavg=[]
    rinh=0.5#maximum inhibition rate of the surround modulation
    for i in range(0,360-startAvg):
        avg=startAvg+i
        totalavg.append(avg)
        y=1/np.sqrt(2*np.pi*stCen**2)*np.e**(-(x-avg)**2/(2*stCen**2))
        centery.append(y)    
    """
    Create the surround effect curve and combine with center population
    """
    unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example
    surroundy=1/np.sqrt(2*np.pi*stSur**2)*np.e**(-(x-avgSur)**2/(2*stSur**2))
    surroundy=surroundy/max(surroundy)*rinh
    resulty=[]
    for i in range(0,len(unitTracker)):
        ymodulated=centery[i]*(1-surroundy[unitTracker[i]])
        resulty.append(ymodulated)   
    """
    Take the population activity readout for the given angle
    """
    def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al.).
        surDecoder=[]
        for i in range(0,len(resulty),unitStep):
            surDecoder.append(resulty[i][stimulusAngle])#surDecoder is symmetric at a given point!
        popSurVector=[]
        for i in range(0,len(surDecoder)):
            popSurVector.append(surDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
            #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
        np.sum(popSurVector)#vector sum of neural population
        decodedAngleSur=np.rad2deg(c.phase(np.sum(popSurVector)))
        if decodedAngleSur<0:
            decodedAngleSur=360+decodedAngleSur #transform the negative angles into angles>180.
        return decodedAngleSur-stimulusAngle , stimulusAngle-avgSur #returns the shift of decoded angle and center-surround difference 
    """
    Induced angular shift against the center-surround discrepancy
    """
    centSurDif=[]
    angShift=[]
    for i in range(0,len(unitTracker)):
       angShift.append(vector_decoder(stimulusAngle=unitTracker[i])[0])
       centSurDif.append(vector_decoder(stimulusAngle=unitTracker[i])[1])
        
    """   
     Plottings all together down here ;)
    """
    def plotter():    
        for i in range(0,len(centery),plotStep):
            plt.plot(x,centery[i])
        plt.title('Tuning curves of center units before surround effect')
        plt.xlabel('Angle')
        plt.ylabel('Neural activity')
        
        plt.figure()
        plt.plot(x,surroundy)
        plt.title('effect of surround angle (180) on center units')
        plt.xlabel('Preferred Unit Angle')
        plt.ylabel('Neural activity inhibition rate')
        
        plt.figure()
        for i in range(0,len(resulty),plotStep):
            plt.plot(x,resulty[i])
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
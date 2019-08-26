# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:25:10 2019

@author: Ibrahim Alperen Tunc
"""
"""
Development script of the von Mises fit decoder, can be ignored, similar codes are also in colclass.py von Mises fit decoder part
Decoder by fitting a von Mises distribution to population activity
Returns the plots of the fit parameters and the color tilt curve.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#!Change the directory where colclass.py is
from colclass import colmod
from scipy.optimize import curve_fit as fit

'''
Call colmod to create center units, surround effect etc.
'''
colMod=colmod(1.5,1,0.5,[60,70],bwType="regular")

'''
To start with the fit first define the von Mises function with parameters to be changed,
than take the population activity for a given stimulus hue angle and use function fit.
'''

def fit_func(ang,amp,kappa,avg):
    """The distribution function:
        This function returns the von Mises distribution values for the given parameters. This will be later used to fit the parameters
        by using the population activity values for the given angle.
        
        Parameters
        ----------
        ang: float. The angle value in degrees. These values are in x-axis. This parameter can also be given as a list.
        amp: float. The amplitude of the distribution.
        kappa: float. The concentration parameter of the distribution.
        avg: float. The global maximum (mode) of the distribution
        
        Returns
        -------
        The von Mises distribution value of the given angle.
    """
    return amp/(2*np.pi)*np.e**(kappa*np.cos(np.deg2rad(ang)-np.deg2rad(avg)))

def von_mises_fit(stimulusAng,avgSur=180):
    """The von Mises fit function:
        This function creates the best von Mises fit on the neuronal population activity by using Levenberg-Marquardt algorithm. The fit
        is done by using the fit_func(ang,amp,kappa,avg) outputs. scipy.optimize.curve_fit() is used for Levenberg-Marquardt algorithm.
        The decoded center hue angle is the angle (x value) of the von Mises fit with global maximum (maximal y-value).
        
        Paramters
        ---------
        stimulusAng: float. The hue angle of the center stimulus in degrees.
        avgSur: float, optional. The hue angle of the surround stimulus in degrees.
        
        Returns
        -------
        popt[2]-stimulusAng: float. The angle difference between the decoded and real center stimulus hues
        stimulusAng-avgSur: float. The angle difference between the center and surround stimuli hues
        popFit: list. All x-values of the von Mises fit to the population activity for different center stimulus conditions.
        surDecoder: list. The ensemble of the population activity for different center stimulus conditions.
        popt: list. The von Mises fit parameter values. These values are amplitude,kappa and global maximum, respectively.
    """
    #stimulusAng=180
    surDecoder=[]#Population activity for a given center-surround stimulus
    for i in range(0,len(colMod.resulty)):
        surDecoder.append(colMod.resulty[i][np.where(colMod.x==stimulusAng)[0][0]])
    x=colMod.x[np.where(colMod.x==0)[0][0]:np.where(colMod.x==360)[0][0]]#angles are restricted to a single cycle for the fit   
    popt, pcov = fit(fit_func, colMod.unitTracker, surDecoder,method='lm')#Levenberg-Marquardt algorithm, popt is the fit parameters, pcov gives the covariance matrix.
    popFit=fit_func(x,*popt)#The fit values to the population activity
    
    """
    Transform the negative angular values to the positive matching angle.
    """
    if popt[2]>360:
        popt[2]=popt[2]-360    
    if popt[2]<-360:
        popt[2]=popt[2]+360
    """
    If the kappa is fit as a negative value, it can be changed to the positive by adding 180° to the maximum value of the fit.
    """
    if popt[1]<0:
        popt[1]=abs(popt[1])
        popt[2]=popt[2]+180
    if popt[2]<-0.1 and popt[1]>0:
        popt[2]=360+popt[2]
#THESE IF STATEMENTS are to solve the irregular parameters problem: due to the fact: cos(x)=cos(-x) and -a*cos(x)=acos(x+180), if kappa is negative it is changed
#to its positive value before adding +180 to the readout angle. Similarly, if kappa is positive but angle is negative, 360 degrees are added (in that part threshold
#is given as -0.1 because in the beginning (stimulus angle=0) there is small error that decoded angle is in -0.000001 range.)
    return popt[2]-stimulusAng, stimulusAng-avgSur, popFit,surDecoder, popt 

angShift=[]
centSurDif=[]
popFit=[]
surDecoder=[]
parameters=[]
for i in range(np.where(colMod.x==0)[0][0],np.where(colMod.x==360)[0][0],10):#Get the von Mises fit values for all center stimuli (spacing 1°)
    vmfit=von_mises_fit(colMod.x[i])
    angShift.append(vmfit[0])
    centSurDif.append(vmfit[1])
    popFit.append(vmfit[2])
    surDecoder.append(vmfit[3])
    parameters.append(vmfit[4])

parameters=list(map(list,zip(*parameters)))#to transpose the list, so that each sublist corresponds to each parameter (amp, kappa, avg)

"""
Plots:
    1) Kappa values of the fits (y-axis) for different center hue angles (x-axis)
    2) Global maximum values of the fits (y-axis) for different center hue angles (x-axis). This value is also the decoded hue angle,
    nonlinearity also hints for the color tilt
    3) The population activity (blue) and the fit curve (orange) for center stimulus with hue angle 170°. Note the surround has 180°
    4) Color tilt curve. x-axis is the center-surround difference, y-axis is the hue shift, all in degrees.
    5) The population activity (blue) and the fit curve (orange) for center stimulus with hue angle 180°, i.e. center=surround. 
"""
plt.plot(parameters[1],'.')
plt.figure()
plt.plot(parameters[2],'.')

plt.figure()
plt.plot(surDecoder[170])
plt.plot(colMod.x[np.where(colMod.x==0)[0][0]:np.where(colMod.x==360)[0][0]],popFit[170])

plt.figure()    
plt.plot(centSurDif,angShift)

plt.figure()
plt.plot(colMod.x[np.where(colMod.x==0)[0][0]:np.where(colMod.x==360)[0][0]],popFit[180])
plt.plot(colMod.unitTracker,surDecoder[180])
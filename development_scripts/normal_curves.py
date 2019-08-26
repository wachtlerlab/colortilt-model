# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:38:01 2019

@author: Ibrahim Alperen Tunc
"""

"""
First step of creating the model, here the linear normal distribution (Gaussian distribution) is used, this whole script can thus be ignored.
Decoder is rather nonsensical and can be ignored.
Returns color tilt curve plot as well as center tuning curve plots before and after surround modulation.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Create normal distribution curve for center population
"""
x=np.linspace(0,360,361)
stCen=20
centery=[]
totalavg=[]
startAvg=90
for i in range(0,181):
    avg=startAvg+i
    totalavg.append(avg)
    y=1/np.sqrt(2*np.pi*stCen**2)*np.e**(-(x-avg)**2/(2*stCen**2))#Gaussian function for center units
    centery.append(y)    
"""
Create the surround effect curve and combine with center population
"""
stSur=50
avgSur=180
surroundy=1/np.sqrt(2*np.pi*stSur**2)*np.e**(-(x-avgSur)**2/(2*stSur**2))*100#Gaussian function for surround inhibition
resulty=[]
for i in range(0,len(centery)):
    ymodulated=centery[i]*(1-surroundy)#as surroundy is the inhibition rate, the activity of the unit at the point is multiplied by 1-inhibition rate
    resulty.append(ymodulated)
"""
Take the population activity readout for the given angle
"""
unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example, each unit is labeled with numbers 1 to N.
def angular_difference(chosenUnit):#this decoding will be substituted by mass of center decoding.
    activityUnit=[]
    popTotal=[]
    for i in range(0,len(centery)):
        activityUnit.append(resulty[i][np.arange(chosenUnit-20,chosenUnit+21)])#array of the unit activity in the 1 sigma interval of stimulus angle
        unitReadout=sum(activityUnit[i][:])#sum of activity for each unit in 1 sigma interval of stimulus angle
        popTotal.append(unitReadout)#the whole unit activity in 1 sigma interval of stimulus angle as an array
    
    popTotalNormalized=popTotal/max(popTotal)#normalize the population activity so that maximal activity is considered as 1.
    
    popReadout=[]
    for i in range(0,len(popTotal)):
        c=popTotalNormalized[i]*(unitTracker[i]-unitTracker[np.where(unitTracker==chosenUnit)[0][0]])#difference between center unit angle and tuning angle of each unit is weighted by unit activity  
        popReadout.append(c)
    
    
    arealSum=sum(popReadout[np.where(unitTracker==chosenUnit)[0][0]-stCen:np.where(unitTracker==chosenUnit)[0][0]+stCen+1])#sum of angle weighted unit activity in 1 sigma interval of stimulus angle 
    return arealSum , chosenUnit-avgSur#gives you the shift and center-surround difference

"""
Instead of taking the unit for decoding, focus on the whole population and use the angle of the unit instead of angular difference between stimulus unit and stdev thing! center of mass decoding 
"""

"""
Induced angular shift against the center-surround discrepancy
"""
centSurDif=[]
angShift=[]
for i in range(stCen,len(unitTracker)-stCen):#
    ydif=angular_difference(chosenUnit=unitTracker[i])[0]
    xdif=angular_difference(chosenUnit=unitTracker[i])[1]
    centSurDif.append(xdif)
    angShift.append(ydif)
    
    
"""   
 Plottings all together down here ;)
"""
plotStep=10

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
        
"""
Short look which st values for center and surround make sense    
"""









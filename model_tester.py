# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:06:49 2019

@author: Ibrahim Alperen Tunc
"""
"""
Model Tester

This script is to test the model with circular Gaussian distribution. Decoding is assessed by using the population vector decoder.
There are also exemplary plots inside showing the population activity and the color tilt prediction of an example model.

Used packages: numpy, scipy, matplotlib, cmath

See also: circular_normal_curves_function.py
"""
import numpy as np
import scipy.special as sp_sp
import matplotlib.pyplot as plt
import cmath as c

"""
Create wrapped normal distribution curve for center population
"""
Kcent=50#Kappa bigger means distribution more dense around peak,1/Kappa ananlogous to variance, Kappa=0 reduces to the uniform distribution
x=np.linspace(-60,420,4801)#center hue angles range from -60° to 420 degrees with 0.1 steps.
centery=[]#List of center hue filter tuning curves, preferred angle starts from 1° and ends at 360° with 1° increment
totalavg=[]
startAvg=1
endAvg=360
for i in range(0,endAvg):
    avg=startAvg+i
    totalavg.append(avg)
    y=1/(2*np.pi*sp_sp.iv(0,Kcent))*np.e**(Kcent*np.cos(np.deg2rad(x)-np.deg2rad(avg)))#von Mises distribution is used for defining the tuning curves
                                                                                       #sp.sp.iv(0,Kcent) is the value of the modified Bessel function of the first kind of the real order for Kcent.  
    centery.append(y/max(y))#normalizing by the maximum value of the tuning curve

"""
Create the surround effect curve and combine with center population
"""
unitTracker=np.arange(startAvg,startAvg+len(totalavg))#Index list to track elements in other lists which correspond to the unit, popReadout[np.where(unitTracker==140)[0][0]] code example
Ksur=10 #Kappa of the surround suppression curve
avgSur=180 #Global maximum of the surround suppression curve
surroundy=1/(2*np.pi*sp_sp.iv(0,Ksur))*np.e**(Ksur*np.cos(np.deg2rad(x)-np.deg2rad(avgSur)))/max(y)#The surround suppression curve, von Mises distributed
resulty=[]#List of center unit tuning curves after surround suppression
for i in range(0,len(unitTracker)):
    ymodulated=centery[i]*(1-surroundy[np.where(x==unitTracker[i])[0][0]])#The modulation is done by rescaling the center unit tuning curve with 1-surroundy.
                                                                          #1-ymodulated value at the preferred hue angle of the center unit is the modulation rate for the center unit, 
                                                                          #the center unit tuning curve will be rescaled accordingly to the 1-ymodulated value.
                                                                          #surroundy[np.where(x==unitTracker[i])[0][0]] this part of the code chooses the ymodulated value for each tuning curve.
    resulty.append(ymodulated)  

"""
Test of decoding by using the population vector decoder.
"""
stimulusAngle=200
unitStep=1
"""
Try first without surround effect to see if decoding works nicely (wokrs fine)
"""
centDecoder=[]
for i in range(0,len(centery),unitStep):
    centDecoder.append(centery[i][np.where(x==stimulusAngle)[0][0]])#select the unit activity for the given stimulus angle for each unit
popCentVector=[]#the vector population of unit activity, each unit as vector is appended to this list in the following lines.
for i in range(0,len(centDecoder)):
    popCentVector.append(centDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
    #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
    #print(np.rad2deg(c.phase(vector)),unitTracker[i])#debug line
np.sum(popCentVector)#vector sum of neural population
decodedAngleCent=np.rad2deg(c.phase(np.sum(popCentVector)))#decoded angle (in degrees), c.phase gives the angle of complex number in radiants
'''
Population vector decoder with surround modulation
'''
def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al. 1990, Georgopoulos et al. 1986, Dayan Theoretical Neuroscience for further info).
    """
    This decoder is documented in colclass.py
    """
    surDecoder=[]
    for i in range(0,len(resulty),unitStep):
        surDecoder.append(resulty[i][np.where(x==stimulusAngle)[0][0]])#select the unit activity for the given stimulus angle for each unit after surround modulation
    popSurVector=[]#the vector population of unit activity, each unit as vector is appended to this list in the following lines.
    for i in range(0,len(surDecoder)):
        popSurVector.append(surDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
        #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
        
    np.sum(popSurVector)#vector sum of neural population
    decodedAngleSur=np.rad2deg(c.phase(np.sum(popSurVector)))#decoded angle (in degrees), c.phase gives the angle of complex number in radiants
    if decodedAngleSur<0:
        decodedAngleSur=360+decodedAngleSur #transform the negative angles into angles>180.
    return decodedAngleSur-stimulusAngle , stimulusAngle-avgSur #decodedAngleSur-stimulusAngle is the shift of decoded angle and stimulusAngle-avgSur is the center-surround difference
"""
Induced angular shift against the center-surround discrepancy
"""
centSurDif=[]#center-surround angle difference
angShift=[]#induced angular shift between the hue angle of real and decoded center stimulus.
for i in range(0,len(unitTracker)):
   ashift,csd=vector_decoder(stimulusAngle=unitTracker[i])
   angShift.append(ashift)
   centSurDif.append(csd)   

"""   
!!!Plots to check the results
"""
"""
Plot of the center units before surround modulation
"""
plotStep=10#number of center tuning curves to be sskipped in plot.

for i in range(0,len(centery),plotStep):
    plt.plot(x,centery[i])
plt.title('Tuning curves of center units before surround effect')
plt.xlabel('Angle')
plt.ylabel('Neural activity')

"""
Plot of the surround suppression curve
"""
plt.figure()
plt.plot(x,surroundy)
plt.title('effect of surround angle (180) on center units')
plt.xlabel('Preferred Unit Angle')
plt.ylabel('Neural activity inhibition rate')

"""
Plot of the center units after surround modulation
"""
plt.figure()
for i in range(0,len(resulty),plotStep):
    plt.plot(x,resulty[i])
plt.title('Tuning curves of center units after surround effect')
plt.xlabel('Angle')
plt.ylabel('Neural activity')


"""
Plot of the color tilt curve by using population vector decoder
"""
plt.figure()
plt.plot(centSurDif,angShift)#for the moment use this given interval, because there are some corner effects (circular variable linearized)
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')
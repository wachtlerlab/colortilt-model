# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:38:01 2019

@author: Ibrahim Alperen Tunc
"""
"""
Development script of the normal_curves_function_03.05, can be totally ignored
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath as c
"""
Create normal distribution curve for center population
"""
stCen=20
x=np.linspace(0,360,361)

centery=[]
totalavg=[]
startAvg=1
endAvg=360
for i in range(0,endAvg):
    avg=startAvg+i
    totalavg.append(avg)
    y=1/np.sqrt(2*np.pi*stCen**2)*np.e**(-(x-avg)**2/(2*stCen**2))
    centery.append(y)    
"""
Create the surround effect curve and combine with center population
"""
unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example
stSur=50
avgSur=180
surroundy=1/np.sqrt(2*np.pi*stSur**2)*np.e**(-(x-avgSur)**2/(2*stSur**2))*100
resulty=[]
for i in range(0,len(unitTracker)):
    ymodulated=centery[i]*(1-surroundy[unitTracker[i]])
    resulty.append(ymodulated)  
    
"""
Take the population activity readout for the given angle, !complex number approach works fine without surround)
"""
stimulusAngle=150
unitStep=1

"""
Try first without surround effect to see if decoding works nicely
"""
centDecoder=[]
for i in range(0,len(centery),unitStep):
    centDecoder.append(centery[i][stimulusAngle])#select the unit activity for the given stimulus angle for each unit
popCentVector=[]
for i in range(0,len(centDecoder)):
    popCentVector.append(centDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
    #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
    #print(np.rad2deg(c.phase(vector)),unitTracker[i])
np.sum(popCentVector)#vector sum of neural population
decodedAngleCent=np.rad2deg(c.phase(np.sum(popCentVector)))#decoded angle (in degrees), c.phase gives the angle of complex number in radiants

"""
Same thing as above but now with +surround
"""
def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al.).
    surDecoder=[]
    for i in range(0,len(resulty),unitStep):
        surDecoder.append(resulty[i][stimulusAngle])
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
plotStep=10

for i in range(0,len(centery),plotStep):#range of 90-270 also possible
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
plt.plot(centSurDif[50:310],angShift[50:310])#for the moment use this given interval, because there are some corner effects (circular variable linearized)
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')
"""
*Comments during development, can be ignored
GITHUB ACCOUNT, MORE COMMENT YOU FORGET WHATS GOIN ON, For the next step adding noise might be clever
von mises (circular gaußian)
"""

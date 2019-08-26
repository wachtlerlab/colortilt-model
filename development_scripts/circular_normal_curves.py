# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:33:28 2019

@author: Ibrahim Alperen Tunc
"""
"""
Redundant version of model_tester.py, can be simply ignored.
"""

'''
Do the whole thing again but in circular manner!
'''
import numpy as np
import matplotlib.pyplot as plt
import cmath as c

"""
Create wrapped normal distribution curve for center population ?WHAT I DO HERE IS ACTUALLY TAKING THE MEAN OF POPULATION IN THE DOMAIN OF CIRCULAR STATISTICS?
"""
Kcent=1.5#Kappa bigger means distribution more dense around peak,1/Kappa ananlogous to variance ?HOW TO IMPLEMENT SIMILAR TO SD
x=np.ndarray.round(np.linspace(-60,420,num=4801),2)#round all these steps to .1 decimals
centery=[]
totalavg=[]
startAvg=1
endAvg=360
for i in range(0,endAvg):
    avg=startAvg+i
    totalavg.append(avg)
    y=1/(2*np.pi)*np.e**(Kcent*np.cos(np.deg2rad(x)-np.deg2rad(avg)))#sp.iv is modified bessel function, first input is order and 2nd is variable
    #leave out the bessel constant, rescale the activity with sum, that total area=1
    centery.append(y/sum(y[np.where(x==0)[0][0]:np.where(x==360)[0][0]]))#normalize the curve by dividing the sum from x=0 to x=359.9 so that total area=1


"""
Create the surround effect curve and combine with center population
"""
unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example
Ksur=0.5
avgSur=180
maxInhRate=0.675
surroundy=1/(2*np.pi)*np.e**(Ksur*np.cos(np.deg2rad(x)-np.deg2rad(avgSur)))#peak of this surround inh. is now maxInhRate
surroundy=surroundy/max(surroundy)*maxInhRate#first divide by max value to get a 1 in peak, then multiply with max inh. rate to get the desired maximum inhibition. 
resulty=[]
for i in range(0,len(unitTracker)):
    ymodulated=centery[i]*(1-surroundy[np.where(x==unitTracker[i])[0][0]])
    resulty.append(ymodulated)  

errorRate=[]
for i in range(0,len(centery)):
        if 1-sum(centery[i][np.where(x==0)[0][0]:np.where(x==360)[0][0]])!=0:#taking the units whose curve area is not exactly 1
            errorRate.append([unitTracker[i],1-sum(centery[i][np.where(x==0)[0][0]:np.where(x==360)[0][0]])])#list of unit ID and error rate
#some units have errors for the total area, approximately 10^-15 (disregard as error or something wrong with code?)

"""
Take the population activity readout for the given angle, !complex number approach works fine without surround)
"""

stimulusAngle=200
unitStep=1
"""
Try first without surround effect to see if decoding works nicely (yup perfecto)
"""
centDecoder=[]
for i in range(0,len(centery),unitStep):
    centDecoder.append(centery[i][np.where(x==stimulusAngle)[0][0]])#select the unit activity for the given stimulus angle for each unit
popCentVector=[]
for i in range(0,len(centDecoder)):
    popCentVector.append(centDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
    #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
    #print(np.rad2deg(c.phase(vector)),unitTracker[i])
np.sum(popCentVector)#vector sum of neural population
decodedAngleCent=np.rad2deg(c.phase(np.sum(popCentVector)))#decoded angle (in degrees), c.phase gives the angle of complex number in radiants
'''
now +surround
'''
def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al.).
    surDecoder=[]
    for i in range(0,len(resulty),unitStep):
        surDecoder.append(resulty[i][np.where(x==stimulusAngle)[0][0]])
    popSurVector=[]
    for i in range(0,len(surDecoder)):
        popSurVector.append(surDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
        #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants
        
    #np.sum(popSurVector)#vector sum of neural population
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
    plt.plot(x,resulty[unitTracker[i]-2])#-2 is to plot maximally inhibited curve as well (resulty[179])
plt.title('Tuning curves of center units after surround effect')
plt.xlabel('Angle')
plt.ylabel('Neural activity')

plt.figure()
plt.plot(centSurDif,angShift)
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')

"""
*The notes taken during the development phase, can be ignored
ORGANISATORY:
    -BA start lets say 03.06 (end is thus 14.08)
    -BRING THE DOCUMENT TO BE SIGNED
THEORETICAL:
    -rescale the curves (done), make a function in the end (done)
    -Another decoding way? (ML (can use von mises MLE in circstats module (astropy))) peak decoder, (literature for decoding types)
    -Look-up table for ML-decoder (matrix for each stimulus the responses before modulation then compare the activity after modulation.)
    -Data fitting? not necessary
    -Anything to change in my coding way? (e.g object oriented with classes and multiple scripts?) ask Nico 
    some questions on how my class definition looks like (in circular_normal_curves_function)
    -...
PRESENTATION:
    -Which topic and when?
    -Bayesian (11.07)
"""
#DIENSTAGS 14.00 Uhr fester Termin T.W
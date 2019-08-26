# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:20:42 2019

@author: Ibrahim Alperen Tunc
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:41:26 2019

@author: Ibrahim Alperen Tunc
"""
'''
Maximum likelihood decoder development script. The decoder has errors, script is kept only to track the project development.
The uniform model is used with the surround stimulus hue angle fixed at 180°
'''
import numpy as np
import matplotlib.pyplot as plt


"""
Create wrapped normal distribution curve for center population
"""
Kcent=5#Kappa bigger means distribution more dense around peak,1/Kappa ananlogous to variance
x=np.ndarray.round(np.linspace(-60,420,num=4801),2)#round all these steps to .1 decimals
centery=[]
totalavg=[]
startAvg=1
endAvg=360
for i in range(0,endAvg):
    avg=startAvg+i
    totalavg.append(avg)
    y=1/(2*np.pi)*np.e**(Kcent*np.cos(np.deg2rad(x)-np.deg2rad(avg)))
    centery.append(y/sum(y[np.where(x==0)[0][0]:np.where(x==359.9)[0][0]]))#normalize the curve by dividing the sum from x=0 to x=359.9 so that total area=1

"""
Create the surround effect curve and combine with center population
"""
unitTracker=np.arange(startAvg,startAvg+len(totalavg))#popReadout[np.where(unitTracker==140)[0][0]] code example
Ksur=2
avgSur=180
maxInhRate=0.5
surroundy=1/(2*np.pi)*np.e**(Ksur*np.cos(np.deg2rad(x)-np.deg2rad(avgSur)))#peak of this surround inh. is now maxInhRate
surroundy=surroundy/max(surroundy)*maxInhRate#first divide by max value to get a 1 in peak, then multiply with max inh. rate to get the desired maximum inhibition. 
resulty=[]
for i in range(0,len(unitTracker)):
    ymodulated=centery[i]*(1-surroundy[np.where(x==unitTracker[i])[0][0]])
    resulty.append(ymodulated)  

errorRate=[]
for i in range(0,len(centery)):
        if 1-sum(centery[i][np.where(x==0)[0][0]:np.where(x==359.9)[0][0]])!=0:#taking the units whose curve area is not exactly 1
            errorRate.append([unitTracker[i],1-sum(centery[i][np.where(x==0)[0][0]:np.where(x==359.9)[0][0]])])#list of unit ID and error rate
#some units have errors for the total area, approximately 10^-15 (disregard as error or something wrong with code?)

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
    for i in range(1,len(unitTracker)+1):
        tableInd.append(centery[np.where(unitTracker==i)[0][0]][np.where(x==stimulusAngle)[0][0]])
    tableML.update({stimulusAngle:tableInd})
    return tableML

"""
Create the lookup table without the population activity entries. Here the keys are the center hue angles, values are not yet assigned,
which will soon be the population activity without surround modulation.
"""
tabLook={}#population activity before surround modulation
for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],10):#this loop is to code more efficiently, that dictionaries are defined outside and only values will be replaced!
    #stimulus angle is from 0 to 359.9 binned 0.1
    tabTemp={x[i]:"Value not yet assigned."}
    tabLook={**tabLook,**tabTemp}#concatanes the table template (dictionary for given stimulus template) with the global lookup table. Another way instead of using tabLook.update(tabTemp)

"""
Fill in the lookup-table with population activity
"""
for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],10):#i here is the stimulusAngle (bin is same as x atm)
    tabLook[x[i]]=np.divide(tab_look(x[i])[x[i]],sum(tab_look(x[i])[x[i]]))#creates a dictionary where key value is stimulus angle and corresponding value to key is population activity in stimulus angle
    #tabLook is normalized with area=1 to enable comparison with thing after modulation. (normalizing where peak=1 better approach?, or no normalization at all?)
    #In colclass.py both with and without normalizing are as options available. In the study without normalization is chosen.
  
def ML_decoder(stimulusAngle):
    """
    Maximum likelihood decoder:
        Choose the most similar lookup table entry for the given population activity after surround modulation. The similarity is quantified 
        by choosing the table entry having the least root mean square difference with the population activity after surround modulation.
        The decoded hue angle is the hue angle of the chosen entry.
        
        Parameters
        ----------
        stimulusAngle: float. The center stimulus hue angle which is desired to be decoded.
        
        Returns
        -------
        stimulusAngle-decodedAng: float. The angular difference between the decoded and the real center stimulus hues in degrees.
        stimulusAngle-avgSur: float. The center-surround angle difference in degrees.
    """  
#stimulusAngle=180
    popInh=[]#The population activity after surround modulation for the given center stimulus angle 
    for i in range(1,len(unitTracker)+1):
        popInh.append(resulty[np.where(unitTracker==i)[0][0]][np.where(x==stimulusAngle)[0][0]])
    popInh=popInh/sum(popInh)#same normalization as tabLook.
    inhRMS=[]#The root mean square difference between the table entries and the population activity after surround modulation.
    for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],10):
        inhRMS.append(np.sqrt(sum((np.subtract(tabLook[x[i]],popInh)**2))))#root mean square to determine most similar curve gives hue shift after normalizing both curves!!!
    decodedAng=x[np.where(x==0)[0][0]:np.where(x==360)[0][0]][np.where(inhRMS==min(inhRMS))[0][0]]#limit x to the interval (0,360) with x[np.where(x==0)[0][0]:np.where(x==360)[0][0]] then take the min RMS value from this array
    return decodedAng-stimulusAngle, stimulusAngle-avgSur #returns hue shift and center-surround difference

"""
Create the lists for ML_decoder() function outputs for a set of center stimuli.
"""
hueShift=[]
centSurDif=[]
for i in range(0,len(np.linspace(0,359,num=360))):
    hueShift.append(ML_decoder(i)[0])
    centSurDif.append(ML_decoder(i)[1])

"""
Plot the color tilt figure
"""
plt.figure()
plt.plot(centSurDif,hueShift,'--')
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')




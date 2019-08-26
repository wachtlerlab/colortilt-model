# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:46:44 2019

@author: Ibrahim Alperen Tunc
"""
'''
Maximum fire rate decoder: look at the population activity in given angle and most active unit ist the decoded angle (winner takes all decoder)
The codes in this script is the further transported to colclass.py. Thus, this script is redundant and is only to check the codes in greater detail/to test
the decoder.
Returns the colortilt plot.
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit")
from colclass import colmod

'''
As always start with modelling the units and stuff with colmod
'''
colMod=colmod(2,0.6,0.7)

def max_fr(stimulusAngle, unitStep=1, avgSur=180):
    """Maximum fire rate decoder:
        This function takes the population activity after surround modulation for a given center-surround stimulus. The decoded center hue angle
        is the preferred hue of the most active center unit.
        
        Parameters
        ----------
        stimulusAngle: float. The center hue stimulus angle in degrees.
        unitStep: integer, optional. The number of units to be skipped while decoding. unitStep=1 means all units are taken into account for the
        decoding.
        avgSur: float, optional. The surround hue stimulus angle in degrees.
        
        Returns
        -------
        colMod.unitTracker[np.where(surDecoder==max(surDecoder))[0][0]]-stimulusAngle: float. Angular difference between the decoded and real hues
        in degrees. Note that the decoded angle is the preferred hue angle of the maximally active unit (colMod.unitTracker[np.where(surDecoder==max(surDecoder))[0][0]])
        stimulusAngle-avgSur: float. The angle difference between and center and surround hues in degrees.
    """
    surDecoder=[]#Population activity after surround modulation
    for i in range(0,len(colMod.resulty),unitStep):
        surDecoder.append(colMod.resulty[i][np.where(colMod.x==stimulusAngle)[0][0]]) 
    return colMod.unitTracker[np.where(surDecoder==max(surDecoder))[0][0]]-stimulusAngle , stimulusAngle-avgSur

"""
Parameter values of the decoder for given center stimulus conditions
"""
angShift=[]
centSurDif=[]
for i in range(0,len(colMod.unitTracker)):
    angShift.append(max_fr(colMod.unitTracker[i])[0]) 
    centSurDif.append(max_fr(colMod.unitTracker[i])[1])

"""
Colortilt plot
"""
plt.figure()
plt.plot(centSurDif,angShift,'--')
plt.plot([-180,180],[0,0])
plt.title('Angular shift relative to center-surround angle difference')
plt.ylabel('Difference between perceived and real angles')
plt.xlabel('Center-surround difference')
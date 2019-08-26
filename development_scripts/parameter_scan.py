# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:05:09 2019

@author: Ibrahim Alperen Tunc
"""
'''
Simple parameter scan for the uniform model, it was used as a starting point of parameter values for the non-uniform model, this whole script
can thus be ignored.
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#Change the directiory accordingly!
import colclass
from supplementary_functions import std2kappa

'''
Below are the final scan intervals found.
'''
Ksur=np.linspace(0.75,1,10)
mod=np.linspace(0.5,0.63,8)
def param_scan(Ksur,mod,angShiftThresh,centSurDifThresh,decoder="vmFit"):
    """Parameter scan function:
        This function makes a parameter scan on the uniform model. Beacuse maximum likelihood and von Mises fit decode the same hues in case
        of the uniform model, here the von Mises fit decoder is used due to its speed.
        
        Parameters
        ---------
        Ksur:list or array. The surround modulation curve kappa values for the scan.
        mod: list or array. Maximum surround suppression values (modulation depth) for the scan.
        angShiftThresh: list. The threshold of the maximum induced hue shift value for the scan to choose the model. It is given in the form of
        [hsb,hsu] where hsb is the lower limit of the threshold interval and hsu is the upper limit. Any model predicting a hue shift in
        this interval is chosen by the scan.
        centSurDifThresh: list. The threshold of the center surround difference value at the maximum induced hue shift for the scan to 
        choose the model. It is given in the form of [csdb,csdu] where csdb is the lower limit of the threshold interval and csdu is 
        the upper limit. Any model predicting a center surround difference in this interval is chosen by the scan.
        decoder: string, optional. The decoder to be used in the scan. von Mises fit "vmFit" is used as default. Other possiblities: 
        "vecSum" for population vector, "mL" for maximum likelihood and "maxFR" for the maximum fire rate decoder.
        
        Returns
        -------
        foundParams: dictionary. The parameter values of the model fitting to the scan criterion (Ksur,mod) as well as the maximum induced hue shift (angShift)
        and its location in form of center-surround difference angle (centSurDif) are included in the dictionary.
    """
    foundParams={"Ksur":[],"mod":[],"centSurDif":[],"angShift":[]}
    for i in range(0,len(Ksur)):   
        for j in range(0,len(mod)):
            colMod=colclass.colmod(std2kappa(60,1.4,1.5),Ksur[i],mod[j])
            
            if decoder=="vecSum":
                dec=colclass.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker)
            
            if decoder=="vmFit":
                dec=colclass.decoder.vmfit(colMod.x,colMod.resulty,colMod.unitTracker)
            
            if decoder=="mL":
                dec=colclass.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker)
            
            if decoder=="maxFR":
                dec=colclass.decoder.maxfr(colMod.x,colMod.resulty,colMod.unitTracker)
            
            if angShiftThresh[0]<=max(dec.angShift)<=angShiftThresh[1] and \
            centSurDifThresh[0]<=dec.centSurDif[np.where(dec.angShift==max(dec.angShift))[0][0]]<=centSurDifThresh[1]:
                print("found parameters matching the requierements")
                print("BINGO YOU HIT THE JACKPOT, Ksur=%s and mod=%s"%(Ksur[i],mod[j]))
                print("Maximum angular shift is %s at the center surround difference of %s"%(max(dec.angShift),
                                                        dec.centSurDif[np.where(dec.angShift==max(dec.angShift))[0][0]]))
                foundParams["Ksur"].append(Ksur[i]),foundParams["mod"].append(mod[j])
                foundParams["centSurDif"].append(dec.centSurDif[np.where(dec.angShift==max(dec.angShift))[0][0]])
                foundParams["angShift"].append(max(dec.angShift))
                plt.figure()
                plt.plot(dec.centSurDif,dec.angShift)
                plt.title("plot with Ksur %s and modulation depth %s"%(Ksur[i],mod[j]))
                 #print("Maximum angular shift is %s at the center surround difference of %s"%(max(vecSum.angShift),
                                                        #vecSum.centSurDif[np.where(vecSum.angShift==max(vecSum.angShift))[0][0]]))
                #if 10<=max(vecSum.angShift)<=18 & 40<=vecSum.centSurDif[np.where(vecSum.angShift==max(vecSum.angShift))[0][0]]<=50:
                #print("BINGO YOU HIT THE JACKPOT, Ksur=%s and mod=%s"%(Ksur[i],mod[j]))
                while True:
                    if plt.waitforbuttonpress(0):
                        break
                plt.close()
                print("waiting for next plot...")
            else:
                print("Parameters do not match the requierements for Ksur=%s and mod=%s" %(Ksur[i],mod[j]))    
                continue
    return foundParams

params=param_scan(Ksur,mod,[14,16],[44,47])# Ksur=1 and mod=0.5371428571428571 seems to make most sense: centSurDif=47 and angShift=14.24795939111496
'''
It would be far better if i used the values for each surround: maximum angular shift and at which angle maximum is, then i could
define a better interval to filter out parameters, but so far this precision is ok for the beginning, because the uniform model
data fit is not of relevance (Kellner manuskript).
'''       


     
        
     
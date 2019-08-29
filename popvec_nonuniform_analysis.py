# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:17:03 2019

@author: Ibrahim Alperen Tunc
"""

"""
Analysis of the population vector decoder in the non-uniform model.
Aim is to eliminat the decoder bias in the no surround case
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#Change the directory accordingly.
import colclass as col
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict, circler
import cmath as c
from scipy import optimize as op
"""
Take the best fit color model to start with
"""
colmod=col.colmod(1,2.3,0.5,stdInt=[1.2,0.9],bwType="gradient/sum",phase=22.5,depInt=[0.2,0.4],depmod=True,stdtransform=False)

"""
Check the model without surround
"""
plt.figure(1)
plt.title("Population activity of the non-uniform model without surround modulation",fontsize=20)
plt.ylabel("Unit activity [a.u.]",fontsize=15)
plt.xlabel("Center hue angle [°]",fontsize=15)
for j in range(23,len(colmod.centery)+1,23):    
    plt.plot(colmod.x[np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]],colmod.centery[j-1][np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]],color="black",linewidth=1)

"""
Population vector decoder without surround
"""
pvd=col.decoder.vecsum(colmod.x,colmod.centery,colmod.unitTracker)
plt.figure(2)
plt.plot(pvd.centSurDif,pvd.angShift,".",markersize=1)#decoding bias +-3
plt.title("Decoding error in the non uniform model",fontsize=20)

vecdict=param_dict(np.linspace(1,360,360),["real","imag","theta","r"])
for i in range(0,len(pvd.popSurVec)):#range between 1-360
    real=[]
    imag=[]
    theta=[]
    r=[]
    for j in range(0,len(pvd.popSurVec[i])):
        real.append(pvd.popSurVec[i][j].real)
        imag.append(pvd.popSurVec[i][j].imag)
        r.append(c.polar(pvd.popSurVec[i][j])[0])
        theta.append(c.polar(pvd.popSurVec[i][j])[1])
        
    vecdict[i+1]["real"]=real;vecdict[i+1]["imag"]=imag;vecdict[i+1]["theta"]=theta;vecdict[i+1]["r"]=r 
      
plt.figure(3)
plt.polar(vecdict[1]["theta"],vecdict[1]["r"],".",markersize=1,color="blue",label="1°")     
plt.polar(vecdict[23]["theta"],vecdict[23]["r"],".",markersize=1,color="red",label="23°")       
plt.polar(vecdict[68]["theta"],vecdict[68]["r"],".",markersize=1,color="green",label="68°")
plt.polar(np.deg2rad([1,1]),[0,0.0008],color="blue")    
plt.polar(np.deg2rad([23,23]),[0,0.0008],color="red")  
plt.polar(np.deg2rad([68,68]),[0,0.0008],color="green")     
plt.title("Vector population examples for the non uniform model",fontsize=20)
plt.legend()
plt.yticks([])

"""
Compare with uniform model polar plots for the population vector decoder
"""
colun=col.colmod(1.5,1,0.5,stdInt=[1.2,0.9],bwType="regular",phase=22.5,depInt=[0.2,0.4],depmod=True,stdtransform=False)
plt.figure(4)
plt.title("Population activity of the uniform model without surround modulation",fontsize=20)
plt.ylabel("Unit activity [a.u.]",fontsize=15)
plt.xlabel("Center hue angle [°]",fontsize=15)
for j in range(23,len(colmod.centery)+1,23):    
    plt.plot(colun.x[np.where(colun.x==0)[0][0]:np.where(colun.x==360)[0][0]],colun.centery[j-1][np.where(colun.x==0)[0][0]:np.where(colun.x==360)[0][0]],color="black",linewidth=1)
pun=col.decoder.vecsum(colun.x,colun.centery,colun.unitTracker)#uniform model decoder without surround

vecdictun=param_dict(np.linspace(1,360,360),["real","imag","theta","r"])
for i in range(0,len(pun.popSurVec)):#range between 1-360
    real=[]
    imag=[]
    theta=[]
    r=[]
    for j in range(0,len(pun.popSurVec[i])):
        real.append(pun.popSurVec[i][j].real)
        imag.append(pun.popSurVec[i][j].imag)
        r.append(c.polar(pun.popSurVec[i][j])[0])
        theta.append(c.polar(pun.popSurVec[i][j])[1])
        
    vecdictun[i+1]["real"]=real;vecdictun[i+1]["imag"]=imag;vecdictun[i+1]["theta"]=theta;vecdictun[i+1]["r"]=r 
plt.figure(5)
plt.polar(vecdictun[1]["theta"],vecdictun[1]["r"],".",markersize=1,color="blue",label="1°")    
plt.polar(vecdictun[23]["theta"],vecdictun[23]["r"],".",markersize=1,color="red",label="23°")      
plt.polar(vecdictun[68]["theta"],vecdictun[68]["r"],".",markersize=1,color="green",label="68°")       
plt.polar(np.deg2rad([1,1]),[0,0.0008],color="blue")    
plt.polar(np.deg2rad([23,23]),[0,0.0008],color="red")  
plt.polar(np.deg2rad([68,68]),[0,0.0008],color="green") 
plt.title("Vector population examples for the uniform model",fontsize=20)
plt.legend()
plt.yticks([])

"""
Both plots seem very similar!
"""

"""
Idea: change the poriton of the summation of each vector so, that the decoding bias vanishes.
"""

"""
Look at the maximum unit activity vector profile for the vecsum uniform and non-uniform model.
"""
plt.figure(6)#to specify each figure
unRealMax=[]#maximum activity of each unit real part uniform model
unImagMax=[]#maximum activity of each unit imaginary part uniform model
plt.figure(7)
nunRealMax=[]#maximum activity of each unit real part non-uniform model
nunImagMax=[]#maximum activity of each unit imaginary part non-uniform model
for i in range(1,len(pun.popSurVec)+1):
    plt.figure(6)#uniform
    plt.plot(vecdictun[i]["real"][list(colmod.unitTracker).index(i)],vecdictun[i]["imag"][list(colmod.unitTracker).index(i)],".",markersize=2.5,color="black")
    unRealMax.append(vecdictun[i]["real"][list(colmod.unitTracker).index(i)])
    unImagMax.append(vecdictun[i]["imag"][list(colmod.unitTracker).index(i)])
    plt.title("Tuning curve maximum activity vectors uniform model",fontsize=20)
    
    plt.figure(7)#non uniform
    plt.plot(vecdict[i]["real"][list(colmod.unitTracker).index(i)],vecdict[i]["imag"][list(colmod.unitTracker).index(i)],".",markersize=1,color="black")
    nunRealMax.append(vecdict[i]["real"][list(colmod.unitTracker).index(i)])
    nunImagMax.append(vecdict[i]["imag"][list(colmod.unitTracker).index(i)])
    plt.title("Tuning curve maximum activity vectors non-uniform model",fontsize=20)

"""
Analyse the circularity of the vector plots for uniform and non-uniform models
"""
circUn=circler((max(unImagMax)-min(unImagMax))/2)#circle radius for the uniform model, instead of imaginary part, real part can also be used
                                                 #as they are completely equal (small error deviance after 8th comma decimal)
circNunr=circler((max(nunRealMax)-min(nunRealMax))/2)#non uniform model vector ellipsis x radius
circNuni=circler((max(nunImagMax)-min(nunImagMax))/2)#non uniform model vector ellipsis y radius

"""
Circle fit for the non-uniform case
"""
def distance_calculator(xc, yc):
    """Calculate the distance of each 2D points from the center (xc, yc) 
    Calculation via pythagorean theorem (np.sqrt(x_d^2+y_d^2))
    """
    return np.sqrt((nunRealMax-xc)**2 + (nunImagMax-yc)**2)

def distance_comparison(c):
    """Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) 
    """
    datDist = distance_calculator(*c)#datapoint distance
    return datDist - datDist.mean()

"""
Small explanation of using * as prefix in functions:
    In the function distance_comparison(c), datDist is calculated by distance_calculator(*c). Here, it means that each element inside c are 
    passed further to the distance_comparison function as input variables. In other words, c is a tuple in form of (xc,yc), where xc and yc
    are numpy.float64. 
    Further info: https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
                  https://treyhunner.com/2018/04/keyword-arguments-in-python/
"""

centEst = np.mean(nunRealMax),np.mean(nunImagMax)#the 1st estimate of the center of the circle fit
centBest, ier = op.leastsq(distance_comparison, centEst)#centBest is the center of the best fit circle, if ier=1,2,3 or 4, solution found, else nothing is found.
                                                        #scipy.optimize.leastsq() takes a function and a starting x value as input for finding the best fit.
                                                        #with least square return is the best fit.
if np.round(centBest[0],decimals=5)!=0 and np.round(centBest[1],decimals=5)!=0:
    raise Exception("the center of the vector population is not at (0,0)")

datDist = distance_calculator(*centBest)#distance between data points and the center of the best circle fit
radBest = datDist.mean()#radius of the best circle fit, calculated as the mean distance between best fit circle center and datapoints.

circNunf=circler(radBest)#non uniform model vector ellipsis best circle fit radius

plt.figure(6) 
plt.plot(*circUn)
plt.figure(7)
plt.plot(*circNunr,color="red",label="x radius")
plt.plot(*circNuni,color="green",label="y radius")
plt.plot(*circNunf,color="blue",label="best fit circle")
plt.legend()
"""
Which one makes most sense? Fitting a circle to the ellipse of non-uniform model or using either the short or long radius of the ellipse 
as the circle radius?
For now circle fit approach is used for normalization.
"""
"""
Normalize the population vectors by using the circle fit
"""
normval=np.array(circNunf[0])/np.array(nunRealMax)#normalization factors for each vector, same result comes for circnunf[1]/nunImagMax
popSurVec2=[]
decodedAng=[]
for i in range(0,len(pvd.popSurVec)):
    popSurVec2.append(np.array(pvd.popSurVec[i])*normval)
    decodedAng.append(np.rad2deg(c.phase(np.sum(popSurVec2[i]))))
    if np.rad2deg(c.phase(np.sum(popSurVec2[i])))<0.5:
        decodedAng[i]=decodedAng[i]+360

plt.figure(8)
plt.title("Decoder errors for different situations",fontsize=20)    
plt.plot(pvd.centSurDif,pvd.angShift,".",markersize=1,color="blue",label="before correction")#decoding bias +-3
plt.plot(pvd.centSurDif,decodedAng-np.linspace(1,360,360),".",markersize=1,color="orange",label="after correction")#max decoder bias is reduced to 0.5264918997367829    
plt.legend()
"""
Alternative idea:
Try to find the portion of vectors to be added around the center hue to get the decoding bias of 0 for the cases of non-symmetric 
population vector activity (center hues not 22.5,112.5,202.5,292.5).
"""
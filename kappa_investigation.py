# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:34:42 2019

@author: Ibrahim Alperen Tunc
"""

"""
Investigation on the relationship between Kappa and standart deviation, there are 3 approaches, each of which is documented below in detail.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as linreg

"""
Define a von Mises distribution wit some kappa and mode at 0 to start with the investigation.
"""
Kcent=2
x=np.linspace(-np.pi,np.pi,num=100*2*np.pi+1)
dist=1/(2*np.pi)*np.e**(Kcent*np.cos(x-0))#von Mises distribution
dist=dist/sum(dist)#normalize the distribution to its total area

"""
Approach 1: Use the curvature (2nd derivative) of the von Mises at mode and approximate it to normal distribution to find a
direct relationship between kappa and std. In other words, the 2nd derivative of both distributions at global maximum are set equal with the 
assumption that they are very similar to each other when they have the same curvature at the mode. This equation gives a formula about the 
relationship between Kappa and standard deviation.   
Derivatives done by hand 
by setting each derivative equal-> sigma^3=sqrt(2pi)/k*e^k
"""
std=np.cbrt(np.sqrt(np.pi)/(Kcent*np.e**Kcent))#!std here in radiant, np.cbrt is the cubic root
distNorm=1/np.sqrt(2*np.pi*std**2)*np.e**(-(x-0)**2/(2*std**2))#The normal distribution with the std value obtained by the formula above.
distNorm=distNorm/sum(distNorm)#normalize to the sum
plt.plot(x,dist,x,distNorm)#seems +- working, if so i can directly write a function where input is std and output is kappa, making life way easier.

"""
Approach 2: Numerically estimate the variance of the von Mises by the formula var(x)=E((x-E(x))^2):
"""
stdvM=np.sqrt(sum(x**2*dist))#as total area is normed to 1, dist gives the probability for each angle point. By that way, the expected value can
                             #calculated by multiplying the angle value by its probability. Furthermore, as the global maximum of the von Mises
                             #distribution is 0, var(x) is simplified as follows: var(x)=E(x^2).
distNorm2=1/np.sqrt(2*np.pi*stdvM**2)*np.e**(-(x-0)**2/(2*stdvM**2))#The normal distribution with the standard deviation of stdvM
distNorm2=distNorm2/sum(distNorm2)#Normalize to total area
plt.figure()
plt.plot(x,dist,x,distNorm2)#Seems to be working better than approach 1.

"""
Approach 3: Plot Kappa against resulting stdvM to see if there is any linear-like relationship in a given interval
"""
kInt=np.linspace(0,10,2001)#Kappa interval with huge binning.
distCom=[]#von Mises comparison distribution with a known Kappa
stdCom=[]#The calculated standard deviation of distCom
for i in range(0,len(kInt)):#Calculate the std for all the distributions with Kappas in kInt.
    distCom.append(1/(2*np.pi)*np.e**(kInt[i]*np.cos(x-0)))
    distCom[i]=distCom[i]/sum(distCom[i])
    stdCom.append(np.sqrt(sum(x**2*distCom[i])))#standard deviation is calculated as in approach 2.

"""
Plot Kappa values against standard deviation.
"""
fig=plt.figure()
plt.title("Relationship between kappa and standard deviation")
plt.xticks([])
plt.yticks([])
plt.box(False)
ax=fig.add_subplot(1,2,1)
ax.set_ylabel("kappa")
ax.set_xlabel("standard deviation")
ax.plot(np.rad2deg(stdCom),kInt,color="black")#relationship seems inverse exponential (coherent with finding in approach 1)
#PROBLEM: even for the maximum BW value of vM (Kappa=0) the BW of Gauss distribution is <105, because the von Mises distribution is reduced
#to uniform distribution for Kappa=0, and the standard deviation of the uniform distribution is calculated as follows:
#std=(b-a)/sqrt(12) where b is the upper limit and a is the lower limit of the interval. Here the limit is +-180, so std is maximum 360/sqrt(12)=103.923

"""
Same plot but for a smaller kappa interval (kappa=1,1.5)
This plot is used in the thesis!
"""
kInt=np.linspace(1,1.5,2001)
distCom=[]
stdCom=[]
for i in range(0,len(kInt)):
    distCom.append(1/(2*np.pi)*np.e**(kInt[i]*np.cos(x-0)))
    distCom[i]=distCom[i]/sum(distCom[i])
    stdCom.append(np.sqrt(sum(x**2*distCom[i])))
ax2=fig.add_subplot(1,2,2)
ax2.set_ylabel("kappa")
ax2.set_xlabel("standard deviation")
ax2.plot(np.rad2deg(stdCom),kInt,color="black")
#Setting a smaller kappa interval causes the relationship to be approximately linear. By this way, linear regression between Kappa and std can
#be done to find out the Kappa value of an std value which we would like to have. 
"""
Fitting a linear regression line to std von Mises and Kappa in interval [0.5;1.5]
The function is transferred to supplementary_functions.py
"""

model=linreg().fit(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)#creating the linear regression model, x value has to be transposed in advance!
model.score(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)#Returns the coefficient of determination R^2 of the prediction.
#0.9978434393176431 is just perfect!
model.intercept_#3.5869855951879352 is the intercept (kappa for stdvM=0), dont take the value serious
model.coef_#-0.03539763 is coefficient, by which x value decays.
# IMPORTANT: this regression is useful if and only if kappa is between 0.5 and 1.5, as the fit is done in that interval!



# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:07:16 2019

@author: Ibrahim Alperen Tunc
"""
"""
This script contains various functions which are further used in different scripts. Each function is described with a documentation.

Necessary modules: numpy, cmath, sklearn, matplotlib
"""
import numpy as np 
import cmath as c   
from sklearn.linear_model import LinearRegression as linreg
import matplotlib.pyplot as plt

def circler(r=1):
    """The circle function:
        This function gives the x-y coordinates of a circle with a given radius. The center is at (0,0).
        
        Parameters
        ----------
        r: float. The radius of the circle.
        
        Returns
        -------
        x: list. The x coordinates of the circle points
        y: list. The y coordinates of the circle points
    """
    x=[]
    y=[]
    angle=np.arange(0,2*np.pi,0.01)#angle in radiants
    for i in range(0,len(angle)):
        x.append(c.rect(r,angle[i]).real)
        y.append(c.rect(r,angle[i]).imag)
        #Here, the complex plane is used to get the x and y values. x value is the real part and the y value is the imaginary part of
        #the complex number r*e^(i*angle) 
    return x,y

def std2kappa(std,Kbel,Kup):
    """The standard deviation transformer:
        This function transforms a given standard deviation value to the Kappa value of the von Mises distribution.
        For this purpose, interpolation with linear regression for a given Kappa interval is done. If the correlation between
        Kappa and standard deviation is smaller than 0.99, or if the estimated Kappa is outside of the Kappa interpolation interval,
        exception is raised.
        
        See also kappa_investigation.py
        
        Parameters
        ----------
        std: float. The standard deviation value which is wished to be transformed to the Kappa value.
        Kbel: float. The lower limit of the Kappa interpolation interval.
        Kup: float. The upper limit of the Kappa interpolation interval.
        
        Returns
        -------
        model.intercept_+std*model.coef_: float. The desired Kappa value of the given standard deviation.
    """
    x=np.linspace(-np.pi,np.pi,num=100*2*np.pi+1)#Create an array spanning from -pi to +pi with a length of 101 (0 in the middle,
                                                 #50 values negative, 50 values positive) 
    kInt=np.linspace(Kbel,Kup,2001)#The kappa interval for interpolation, spanning from Kbel and Kup, with 2001 bins total.
    distCom=[]#List of von Mises distributions with different Kappa values chosen from kInt, normalized by total area
    stdCom=[]#Standard deviation values of the distributions in distCom, calculated with the formula sqrt(sum(x**2*y)) whereas x is 
             #the distribution variable and y is the relative density of the distribution variable.
    for i in range(0,len(kInt)):
        distCom.append(1/(2*np.pi)*np.e**(kInt[i]*np.cos(x-0)))
        distCom[i]=distCom[i]/sum(distCom[i])
        stdCom.append(np.sqrt(sum(x**2*distCom[i])))
    model=linreg().fit(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)#creating the linear regression model, x value has to be transposed in advance!
                                                                         #np.reshape(-1,1) creates an nx1 matrix out of the array.
    model.score(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)#Returns the coefficient of determination R^2 of the prediction.
    model.intercept_#the intercept (kappa for stdvM=0), this value is not to be taken seriously.
    model.coef_#the coefficient coefficient, by which x value decays.
    if model.score(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)<0.99:#Make sure the correlation coefficient>0.99
        raise Exception("The fit is not good enough. Correlation coefficient=%s"%(model.score(np.asarray(np.rad2deg(stdCom)).reshape(-1,1),kInt)))
    if (model.predict(std)>Kbel and model.predict(std)<Kup)==False:#Make sure the estimated Kappa is inside the interpolation interval.
        raise Exception("WARNING! The estimated Kappa is not in given interval. Interval=[%s,%s], estimated Kappa=%s"%(Kbel,Kup,model.intercept_+std*model.coef_))
    return model.intercept_+std*model.coef_  #this function is useful to estimate the Kcent in colmod (colclass.py)!

def depth_modulator(depInt,avgSur,phase=0):
    """
    Surround modulation depth specifier:
        This function is to specify the surround suppression strength for a given surround hue angle. In the non-uniform version of the
        color model, surround suppression depends on the surround hue angle. Here the suppression strength is varied by using a cosine 
        function.
        
        Parameters
        ----------
        depInt: list. The lower and upper values of the maximum inhibition as a function of surround hue angle. Can be between 0 and 1.
        the parameter should be given as [lower,upper]
        avgSur: float. The surround stimulus hue angle of interest. The angle should be given in degrees.
        phase: float, optional. The surround stimulus hue angle, for which the maximum surround inhibition is the upper value of depInt.
        
        Returns
        -------
        depMod[np.where(np.linspace(startAvg,endAvg,719)==avgSur)[0][0]]: float. The modulation depth value for the given surround stimulus hue angle.
    """
    startAvg=0
    endAvg=359.5
    depMod=(depInt[1]-depInt[0])/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,720)-phase))+depInt[0]+(depInt[1]-depInt[0])/2
    #depMod is the cosine function, the y value of this function gives the modulation depth of the chosen surround hue angle (x-axis).
    #Depmod has a full periodicity in 180°, its minimal value is depInt[0] (lower max inhibition value) and maximal value is
    #depInt[1] (upper max inhibition value).
    return depMod[np.where(np.linspace(startAvg,endAvg,720)==avgSur)[0][0]]

def param_dict(dec,params): 
    """Dictionary creator for decoder types and parameters:
        Preallocates the decoder and parameter dictionary for the given decoder and parameter names. The dictionary has then the form
        {dec1:{param1,...},dec2:{param1,...},...}
        
        Parameters
        ----------
        dec: string. The name of the decoder. This parameter can also be given as a list of strings.
        params: string. The name of the parameters. This parameter can also be given as a list of strings.
        
        Returns
        -------
        dicti: dictionary. The preallocated dictionary including empty decoder and parameter dictionaries inside.
    """
    
    dicti={}#the dictionary to preallocate
    for i in range(0,len(dec)):
        dicti.update({dec[i]:{}})#first each decoder is added as empty dictionaries into dicti. 
        for j in range(0,len(params)):
            dicti[dec[i]].update({params[j]:{}})#each decoder sub-dictionary is updated with the empty dictionaries of the parameters.
    return dicti


class plotter:
    """This class contains 2 functions used for the plots with different surround subplots:
        1) plot_template(): Creates the empty figure. 
        2) subplotter(): Creates each of the surround subplot.
    """
    def plot_template(auto=False):
        """Plot template creator:
            Creates the empty figure for the surround subplotting. Here, the x, y labels as well as the plot title can be given manually.
            
            Parameters
            ----------
            auto: boolean, optional. If set to True, the plot title and axis labels have to be given in a separate line, making the naming automatically.
            If False, then each text is asked accordingly.
            
            Returns
            -------
            fig: figure object. The empty figure which is ready for subplotting.
        """
        fig=plt.figure()
        plt.xticks([])#Makes the x-axis ticks off
        plt.yticks([])#Makes the y-axis ticks off
        plt.box(False)#Makes the plot frame off
        ax=plt.gca()#Retrieves the current axes
        ax.xaxis.set_label_coords(0.5, -0.07)#sets the position of x-axis label
        ax.yaxis.set_label_coords(-0.05,0.5)#sets the position of y-axis label
        if auto==False:#Each title and labels have to be written manually
            plt.title(input("plot title    "),y=1.08,fontsize=20)
            plt.xlabel(input("x label    "),fontsize=15)
            plt.ylabel(input("y label    "),fontsize=15,x=-0.08)
        return fig
    
    def subplotter(fig,i):
        """The surround subplot creator:
            Creates each of the subplot according to the surround angle. This is done by firstly dividing the whole figure to the 3x3
            subplots. The subplot in the middle is then used to denote the surround angle of each of the adjacent subplot.
            
            Note that subplots are ordered as follows for the surround hue angle: 135,90,45,180,0,225,270,315
            
            Parameters
            ----------
            fig: figure object. The figure on which the subplots are to be plotted. Best is to use the output of the plot_template() function.
            i: integer. The number of the subplot to be done. Subplotting starts from the uppermost and leftmost subplot, then goes to right
            and down, respectively.
            
            Returns
            -------
            ax1: figure object. The current subplot for the plotting
        """
        if i<=3:
            ax1=fig.add_subplot(3,3,i+1)#First 4 subplots (0:3) are added normally
        elif i==4:#The fifth subplot (i=4) is used for texts to note each surround condition
            ax1=fig.add_subplot(3,3,i+2)
            ax2=fig.add_subplot(3,3,i+1)
            ax2.text(0.5,0.45,"surround angle",size=18,ha="center")
            ax2.text(0,0,"225",size=15)
            ax2.text(0.45,0,"270",size=15)
            ax2.text(0.9,0,"315",size=15)
            ax2.text(0.9,0.45,"0",size=15)
            ax2.text(0,0.45,"180",size=15)
            ax2.text(0.9,0.9,"45",size=15)
            ax2.text(0.45,0.9,"90",size=15)
            ax2.text(0,0.9,"135",size=15)
            ax2.axis("off")
        else:#The remaining subplots are added by skipping the 5th subplot area.
            ax1=fig.add_subplot(3,3,i+2)
        return ax1
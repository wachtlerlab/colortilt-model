# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:52:03 2019

@author: Ibrahim Alperen Tunc
"""
"""
Analysis of the models and parameters filtered in the data_analysis.py file.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#!Change the directory accordingly
import colclass as col
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

paraml=pickle.load(open("paraml_fit_10_errType_rms_2019-08-23.pckl","rb"))#Pickle file of the scanned parameters. This name should be changed according to the file wished to be analyzed.
dictTot=pickle.load(open("dicttot.pckl","rb"))#Dictionary of the subject average.
meanrms=[]
for i in range(0,len(paraml)):
    meanrms.append(np.array(list(paraml[i]["dif"].values())).mean())#The mean root mean square error fo the filtered models

"""
Histogram of the parameter distribution. FIGURE 6
"""
plt.hist(meanrms,bins=100,color="black")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
plt.title("RMS histogram of models with different parameters",fontsize=20)
plt.xlabel("Root mean square error",fontsize=15)
plt.ylabel("Number of models",fontsize=15)
plt.xticks(np.arange(3.7,7.6,0.2),)
plt.tick_params(axis='both', which='major', labelsize=15)


fltrms=[meanrms[np.where(np.array(meanrms)<4)[0][i]] for i in range(0,len(np.where(np.array(meanrms)<4)[0]))]#Take out the parameters which have a RMS >4
fltind=[np.where(np.array(meanrms)<4)[0][i] for i in range(0,len(np.where(np.array(meanrms)<4)[0]))]#index list of the fltrms
outind=[np.where(np.array(meanrms)>=4)[0][i] for i in range(0,len(np.where(np.array(meanrms)>=4)[0]))]#index list of the parameters not included in fltrms
fltrms, fltind=zip(*sorted(zip(fltrms,fltind)))#sort fltind and fltrms values in the same ascending order 

"""
Calculate average center BW, min-max center BW difference, average sur mod depth, min-max sur mod depth by using following parameters:
Parameters: surround modulation width, average center BW, min-max center BW difference, average mod depth sur, min-max mod depth sur (phase all equal in this case)
"""
smw=[]#surround modulation width
difc=[]#center min-max BW difference
difsm=[]#surround mod depth min-max difference
cabw=[]#center average BW
smad=[]#surround average mod depth

"""
calculation for fltrms/fltind parameters
"""
for i in range(0,len(fltind)):
    smw.append(paraml[fltind[i]]["ksi"])#ksi is the surround kappa values in the dictionary.
    difc.append(paraml[fltind[i]]["ku"]-paraml[fltind[i]]["kb"])#the center min-max difference
    difsm.append(paraml[fltind[i]]["depu"]-paraml[fltind[i]]["depb"])#surround modulation min-max difference
    cabw.append(difc[i]/2+paraml[fltind[i]]["kb"])#take the min max difference, divide by 2 and add minimal value to get the average bandwidth.
    smad.append(difsm[i]/2+paraml[fltind[i]]["depb"])#take the min max difference, divide by 2 and add below value the average mod depth.

"""
calculation for outind parameters
"""
smwf=[]#surround mod width list for filtered out models
difcf=[]#center min-max BW difference list for filtered out models
difsmf=[]#surround mod depth min-max difference list for filtered out models
cabwf=[]#center average BW list for filtered out models
smadf=[]#surround average mod depth list for filtered out models
for i in range(0,len(outind)):#same loop as above for all model parameters with rms>4
    smwf.append(paraml[outind[i]]["ksi"])
    difcf.append(paraml[outind[i]]["ku"]-paraml[outind[i]]["kb"])
    difsmf.append(paraml[outind[i]]["depu"]-paraml[outind[i]]["depb"])
    cabwf.append(difcf[i]/2+paraml[outind[i]]["kb"])#take the min max difference, divide by 2 and add below value
    smadf.append(difsmf[i]/2+paraml[outind[i]]["depb"])#take the min max difference, divide by 2 and add below value

"""
Plot the parameter triplets in 3D, make also 2D variant of the same (same number of plots) FIGURE 7
"""
"""
Plot formatting
"""
plotdict={"smw":smw,"cBWd":difc,"smd":difsm,"caBW":cabw,"sma":smad}#to assign the parameters to an axes while plotting
plotdictf={"smwf":smwf,"difcf":difcf,"difsmf":difsmf,"cabwf":cabwf,"smadf":smadf}#to assign the parameters to an axes while plotting
params=["smw","cBWd","smd","caBW","sma"]#to name the parameters in the plot title
paramsf=["smwf","difcf","difsmf","cabwf","smadf"]#to name the parameters in the plot title
colMap=["gray","red","blue"]#color map
labmap=["rms>4","rms<4","best 10 models"]#label map
fig = plt.figure()
plt.title("Parameter distribution of models",fontsize=18,y=1.08)
plt.xticks([])#main plot figure ticks are off
plt.yticks([])#main plot figure ticks are off
plt.box(False)#the frame of the main plot is off

"""
Plot all triple combinations of the 5 parameters in a total of 10 subplots. FIGURE SUPPLEMENT 3
"""
i=0    
for j in range(0,3):#combinations including smw
    x=plotdict[params[j]]#parameters with rms<4
    x1=plotdictf[paramsf[j]]#parameters with rms>4
    x2=plotdict[params[j]][0:10]#best 10 model parameters
    for k in range(1,4):#difc combinations
        if j>=k:#stop if j>k to avoid repetition of the same parameter triplets.
            continue
        y=plotdict[params[k]]
        y1=plotdictf[paramsf[k]]
        y2=plotdict[params[k]][0:10]
        for l in range(2,5):#remaining combinations
            if j>=l:#stop if j>l to avoid repetition of the same parameter triplets.
                continue
            if k>=l:#stop if k>l to avoid repetition of the same parameter triplets.
                continue
            z=plotdict[params[l]]
            z1=plotdictf[paramsf[l]]
            z2=plotdict[params[l]][0:10]
            i=i+1#in each iteration next subplot figure is chosen
            ax=fig.add_subplot(3,4,i,projection='3d')#3D subplots.
            ax.plot3D(x1,y1,z1,'o',color=colMap[0],label=labmap[0])
            ax.plot3D(x,y,z,'o',color=colMap[1],label=labmap[1])
            ax.plot3D(x2,y2,z2,'o',color=colMap[2],label=labmap[2])
            ax.set_title('x=%s , y=%s , z=%s'%(params[j],params[k],params[l]),fontdict={'fontsize': 15, 'fontweight': 'medium'})#subplot title serves as the axis naming for each subplot.
            ax.tick_params(axis='both', which='major', labelsize=13)#axes ticks are made bigger.
ax.legend(loc="best", bbox_to_anchor=(1.2,1),fontsize=15)

"""
Plot all pair combinations of the 5 parameters in a total of 10 subplots. The loop is structured the same as the triplet version.
FIGURE 7
"""
i=0
fig2 = plt.figure()
plt.title("Parameter distribution of models",y=1.08,fontsize=18)
plt.xticks([])
plt.yticks([])
plt.box(False)
for j in range(0,4):
    x=plotdict[params[j]]
    x1=plotdictf[paramsf[j]]
    x2=plotdict[params[j]][0:10]
    for k in range(1,5):
        if j>=k:
            continue
        y=plotdict[params[k]]
        y1=plotdictf[paramsf[k]]
        y2=plotdict[params[k]][0:10]
        i=i+1
        ax=fig2.add_subplot(3,4,i)
        ax.plot(x1,y1,'o',color=colMap[0],label=labmap[0])
        ax.plot(x,y,'o',color=colMap[1],label=labmap[1])
        ax.plot(x2,y2,'o',color=colMap[2],label=labmap[2])
        ax.set_title('x=%s , y=%s , z=%s'%(params[j],params[k],params[l]),fontdict={'fontsize': 15, 'fontweight': 'medium'})
        ax.tick_params(axis='both', which='major', labelsize=15)

ax.legend(loc="best", bbox_to_anchor=(1.2,1),fontsize=15)

"""
Look at the filtered model fits with waitforbuttonpress 
NOTE: this code should be stopped after a while, as it will go through each and every model in the uploaded file.
TO DO: Axis lines (x=0,y=0) and model estimates without psychophysics data
"""
q=0#To track the number of the model fit.
labmap=["data","model"]#label map
surrInt=(135,90,45,180,0,225,270,315)#surround subplot order should be like this (see help(plotter.subplotter))
for i in fltind:
    q=q+1#in each iteration the model fit number goes up by 1.
    fig= plt.figure()
    plt.title("Model fit for induced hue shift, model #%s"%(q),fontsize=20)#Figure formatting the usual way from here up until for loop.
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    ax=plt.gca()
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.yaxis.set_label_coords(-0.05,0.5)
    plt.xlabel("center surround difference",fontsize=20)
    plt.ylabel("angular shift",fontsize=20)
    
    for j in range(0,len(dictTot)):#Create the model and decoder objects for each surround by using the parameter sets. Dicttot has each surround condition as element inside.
        #colMod=col.colmod(Kcent,Ksur,maxInh,stdInt,bwType="gradient/sum",phase=phase,avgSur=surrAvg[i],depInt=depInt,depmod=True,stdtransform=False)
        colMod=col.colmod(1,paraml[i]["ksi"],1,stdInt=[paraml[i]["ku"],paraml[i]["kb"]],bwType="gradient/sum",\
                          phase=22.5,avgSur=surrInt[j],depInt=[paraml[i]["depb"],paraml[i]["depu"]],depmod=True,stdtransform=False)#The model      
        dec=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=surrInt[j],dataFit=True)#the decoder
        ax=plotter.subplotter(fig,j)#subplot the color tilt for each surround
        ax.errorbar(dictTot[surrInt[j]]["angshi"].keys(),dictTot[surrInt[j]]["angshi"].values(),dictTot[surrInt[j]]["se"].values(),fmt='.',capsize=3,label=labmap[0],ecolor="gray",color="black")
        #data plot with errorbars (above line)
        ax.plot(dec.centSurDif,dec.angShift,color=colMap[1],label=labmap[1])#model plot
        ax.set_ylim(bottom=-25,top=25)#y axis limit +-25
        ax.set_xticks(np.linspace(-180,180,9))#x ticks between +-180 and ticks are at cardinal and oblique angles.
        ax.tick_params(axis='both', which='major', labelsize=15)#major ticks are increazed in label size
        ax.xaxis.set_major_locator(MultipleLocator(90))#major ticks at cardinal angles.
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(45))#minor ticks at obliques.
        print("1 subplot done")
    ax.legend(loc="best", bbox_to_anchor=(1,1),fontsize=20)#add the legend to the subplot in the very end, after all surrounds are plotted.
    while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
        if plt.waitforbuttonpress(0):
            break
    plt.close()

"""
Kellner angle vs my SE rms comparison
"""
"""
kelang=[3.83,3.60,1.81,3.21,2.95,3.05,2.63,3.83]
se=[]
for i in np.linspace(0,315,8):
    print(np.mean(kelang[int(i/45)]/np.array(list(dictTot[i]["se"].values()))))
    se.append(np.mean(kelang[int(i/45)]/np.array(list(dictTot[i]["se"].values()))))
"""

'''
*Development notes
MAKE ALSO THE HISTOGRAM BETTER, WITH AXIS NAMES ETC!
GIVE MEAN AND VARIANCE VALUES FOR EACH PARAMETER BASED ON BEST 10 FITS
'''
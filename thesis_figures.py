# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:06:49 2019

@author: Ibrahim Alperen Tunc
"""

"""
Figure reproduction for thesis (cooler more formal etc with legends for all figures, also each figure is named accordingly)
Each figure in the thesis is commented in the beginning. Figures are commented out, so the codes of the figure which is wished to be created
should be first chosen to run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#Change the directory accordingly.
import colclass as col
from scipy import stats as st
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


"""
decoder labels/colors, this labeling is used for all figures for consistency.
"""
labs=["population vector","von Mises fit","maximum likelihood","maximum fire rate"]
colors=["red","blue","black","green"]

"""
figure supplement 2
Uniform model surround modulation and tuning curves before and after surround supression
"""
"""
a=col.colmod(1.5,1,0.5,[1,10],bwType="regular")
fig=plt.figure()
plt.title("Center unit tuning curves and surround modulation in the uniform model",fontsize=20,y=1.08)
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(1,3,1)
for i in range(45,len(a.centery)+1,45):#range of 90-270 also possible
    print(i)
    ax1.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1.0)
ax1.set_xlabel('Hue angle of center stimulus',fontsize=15)
ax1.set_xticks(np.linspace(0,360,9))
ax1.set_ylabel('Neuronal activity (arbitrary units)',fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))

ax2=fig.add_subplot(1,3,2)
ax2.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],1-a.surroundy[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]])
ax2.set_xlabel('Preferred hue angle of center filter',fontsize=15)
ax2.set_xticks(np.linspace(0,360,9))
ax2.set_yticks(np.linspace(0.4,1,7))
ax2.set_ylabel('Modulation rate',fontsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.xaxis.set_major_locator(MultipleLocator(90))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_minor_locator(MultipleLocator(45))

ax3=fig.add_subplot(1,3,3)
for i in range(45,len(a.centery)+1,45):
    ax3.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="gray",linewidth=0.5,label="before surround modulation")
    ax3.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.resulty[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1,label="after surround modulation")
ax3.set_xlabel('Hue angle of center stimulus',fontsize=15)
ax3.set_xticks(np.linspace(0,360,9))
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_ylabel('Neuronal activity (arbitrary units)',fontsize=15)
ax3.legend(["before surround modulation","after surround modulation"],loc="best", bbox_to_anchor=(1,1),fontsize=15)
ax3.xaxis.set_major_locator(MultipleLocator(90))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_minor_locator(MultipleLocator(45))
plt.tight_layout()


"""

"""
figure supplement 1
The realtionship between kappa nad standard deviation for a huge and small kappa interval.
"""
"""
x=np.linspace(-np.pi,np.pi,num=100*2*np.pi+1)
kInt=np.linspace(0,10,2001)
distCom=[]
distComNor=[]
stdCom=[]
for i in range(0,len(kInt)):
    distCom.append(1/(2*np.pi)*np.e**(kInt[i]*np.cos(x-0)))
    distCom[i]=distCom[i]/sum(distCom[i])
    stdCom.append(np.sqrt(sum(x**2*distCom[i])))

fig=plt.figure()
plt.title("Relationship between kappa and standard deviation",fontsize=20,y=1.08)
plt.xticks([])
plt.yticks([])
plt.box(False)
ax=fig.add_subplot(1,2,1)
ax.set_ylabel("kappa",fontsize=15)
ax.set_xlabel("standard deviation",fontsize=15)
ax.plot(np.rad2deg(stdCom),kInt,color="black")#relationship seems inverse exponential (coherent with finding in approach 1)
ax.tick_params(axis='both', which='major', labelsize=15)

kInt=np.linspace(1,1.5,2001)
distCom=[]
distComNor=[]
stdCom=[]
for i in range(0,len(kInt)):
    distCom.append(1/(2*np.pi)*np.e**(kInt[i]*np.cos(x-0)))
    distCom[i]=distCom[i]/sum(distCom[i])
    stdCom.append(np.sqrt(sum(x**2*distCom[i])))
ax2=fig.add_subplot(1,2,2)
ax2.set_ylabel("kappa",fontsize=15)
ax2.set_xlabel("standard deviation",fontsize=15)
ax2.plot(np.rad2deg(stdCom),kInt,color="black")
plt.tight_layout()
ax2.tick_params(axis='both', which='major', labelsize=15)

"""

"""
figure 2
Effecst of changing the parameters in the uniform model
"""
"""
col1=col.colmod(1.5,1,0.5,[1,10],bwType="regular")
dec1v=col.decoder.vecsum(col1.x,col1.resulty,col1.unitTracker)
dec1vm=col.decoder.vmfit(col1.x,col1.resulty,col1.unitTracker)
dec1m=col.decoder.ml(col1.x,col1.centery,col1.resulty,col1.unitTracker)
dec1mf=col.decoder.maxfr(col1.x,col1.resulty,col1.unitTracker)

col2=col.colmod(2,1,0.5,[1,10],bwType="regular")
dec2v=col.decoder.vecsum(col2.x,col2.resulty,col2.unitTracker)
dec2vm=col.decoder.vmfit(col2.x,col2.resulty,col2.unitTracker)
dec2m=col.decoder.ml(col2.x,col2.centery,col2.resulty,col2.unitTracker)

col3=col.colmod(1.5,2,0.5,[1,10],bwType="regular")
dec3v=col.decoder.vecsum(col3.x,col3.resulty,col3.unitTracker)
dec3vm=col.decoder.vmfit(col3.x,col3.resulty,col3.unitTracker)
dec3m=col.decoder.ml(col3.x,col3.centery,col3.resulty,col3.unitTracker)

col4=col.colmod(1.5,1,0.7,[1,10],bwType="regular")
dec4v=col.decoder.vecsum(col4.x,col4.resulty,col4.unitTracker)
dec4vm=col.decoder.vmfit(col4.x,col4.resulty,col4.unitTracker)
dec4m=col.decoder.ml(col4.x,col4.centery,col4.resulty,col4.unitTracker)

fig=plt.figure()
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(1,3,1)
ax1.set_title("population vector",fontsize=15)
ax1.set_ylabel("Induced hue shift [°]",fontsize=15)
ax1.set_xticks(np.linspace(-180,180,9))
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))
ax1.plot(dec1v.centSurDif,dec1v.angShift,label="reference curve",color="black")
ax1.plot(dec2v.centSurDif,dec2v.angShift,label="narrower center tuning",color="red")
ax1.plot(dec3v.centSurDif,dec3v.angShift,label="narrower surround modulation",color="blue")
ax1.plot(dec4v.centSurDif,dec4v.angShift,label="stronger surround modulation",color="green")


ax2=fig.add_subplot(1,3,2)
ax2.set_title("von Mises fit",fontsize=15)
ax2.set_xlabel("Hue difference between center and surround [°]",fontsize=15)
ax2.set_xticks(np.linspace(-180,180,9))
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.xaxis.set_major_locator(MultipleLocator(90))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_minor_locator(MultipleLocator(45))
ax2.plot(dec1vm.centSurDif,dec1vm.angShift,label="reference curve",color="black")
ax2.plot(dec2vm.centSurDif,dec2vm.angShift,label="narrower center tuning",color="red")
ax2.plot(dec3vm.centSurDif,dec3vm.angShift,label="narrower surround modulation",color="blue")
ax2.plot(dec4vm.centSurDif,dec4vm.angShift,label="stronger surround modulation",color="green")

ax3=fig.add_subplot(1,3,3)
ax3.set_title("maximum likelihood",fontsize=15)
ax3.set_xticks(np.linspace(-180,180,9))
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.xaxis.set_major_locator(MultipleLocator(90))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_minor_locator(MultipleLocator(45))
ax3.plot(dec1m.centSurDif,dec1m.angShift,label="reference curve",color="black")
ax3.plot(dec2m.centSurDif,dec2m.angShift,label="narrower center tuning",color="red")
ax3.plot(dec3m.centSurDif,dec3m.angShift,label="narrower surround modulation",color="blue")
ax3.plot(dec4m.centSurDif,dec4m.angShift,label="stronger surround modulation",color="green")
ax3.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
plt.tight_layout()
"""
"""
Supplementary figure 5
The effect of surround modulation curve bandwidth on the suppression strength
"""
"""
try:
    col1;col3
except NameError:
    col1=col.colmod(1.5,1,0.5,[1,10],bwType="regular")
    col3=col.colmod(1.5,2,0.5,[1,10],bwType="regular")
plt.figure()
plt.title("surround modulation for different bandwidths",fontsize=20)
plt.xlabel("center unit preferred hue angle",fontsize=15)
plt.ylabel("surround modulation",fontsize=15)
plt.plot(col1.x[np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],1-col1.surroundy[np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],color="black",label="Ks=1")
plt.plot(col3.x[np.where(col3.x==0)[0][0]:np.where(col3.x==360)[0][0]],1-col3.surroundy[np.where(col3.x==0)[0][0]:np.where(col3.x==360)[0][0]],color="red",label="Ks=2")
plt.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(np.linspace(0,360,9))

"""

"""
figure 1 
all decoders in once
"""
"""
try:
    col1;dec1v
except NameError:
    col1=col.colmod(1.5,1,0.5,[1,10],bwType="regular")
    dec1v=col.decoder.vecsum(col1.x,col1.resulty,col1.unitTracker)
    dec1vm=col.decoder.vmfit(col1.x,col1.resulty,col1.unitTracker)
    dec1m=col.decoder.ml(col1.x,col1.centery,col1.resulty,col1.unitTracker)
    dec1mf=col.decoder.maxfr(col1.x,col1.resulty,col1.unitTracker)

plt.figure()
plt.title("Color tilt predictions of decoders",fontsize=20)
plt.ylabel("Induced hue shift [°]",fontsize=15)
plt.xlabel("Hue difference between center and surround [°]",fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(np.linspace(-180,180,9))


plt.plot(dec1v.centSurDif,dec1v.angShift,label=labs[0],color=colors[0])
plt.plot(dec1vm.centSurDif,dec1vm.angShift,label=labs[1],color=colors[1])
plt.plot(dec1m.centSurDif,dec1m.angShift,label=labs[2],color=colors[2])
plt.plot(dec1mf.centSurDif,dec1mf.angShift,label=labs[3],color=colors[3])
plt.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
"""

"""
figures 3,4: decoding error for gradient center bw
done in circular_model_different_BW.py
3 is maxfr, 4 is totalsum normalized
"""

"""
Figure 5: Qualitative data reproduction
analysis_gradientvsnromalBW_changingsurround was used
"""

"""
Figure 6: Histogram of RMS
Figure 7: Parameter pairs for different RMS
Figure 8: The best model fit
data_params.py was used
"""

"""
Figure 9: tuning curves of the best model as a function of surround
TO DO: Add in the middle of S4 the pop act. without surround for comparison. DONE
"""
"""
fig=plotter.plot_template(auto=True)
plt.title("Tuning curves of the best model",y=1.08,fontsize=20)#1st plot is the color tilt curve, subplots with different surrounds
plt.xlabel("Center stimulus hue angle [°]",fontsize=15)
plt.ylabel("Unit activity [a.u.]",fontsize=15,x=-0.1)
surr=[135,90,45,180,0,225,270,315]
for i in range(0,len(surr)):
    a=col.colmod(1,2.3,0.5,[1.2,0.9],bwType="gradient/sum",phase=22.5,avgSur=surr[i],depInt=[0.2,0.4],depmod=True,stdtransform=False)
    if i==4:
        ax=fig.add_subplot(3,3,i+2)
        ax2=fig.add_subplot(3,3,i+1)
    else:
        ax=plotter.subplotter(fig,i)
    for j in range(23,len(a.centery)+1,23):
        ax.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.resulty[j-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1)
        if i==4:
            ax2.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[j-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1)
            ax2.set_xticks(np.linspace(0,360,9))
            ax2.tick_params(axis='both', which='major', labelsize=15)
            ax2.xaxis.set_major_locator(MultipleLocator(90))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax2.xaxis.set_minor_locator(MultipleLocator(45))
            ax2.set_xlim([0,360])

    ax.set_xticks(np.linspace(0,360,9))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(45))
    ax.set_xlim([0,360])
"""

"""
Supplementary figure 4: the best fit model population activity without surround
Added into the middle subplot of figure 9
"""
"""
plt.figure()
plt.title("Population activity of the best fit model without surround modulation",fontsize=20)
plt.ylabel("Unit activity [a.u.]",fontsize=15)
plt.xlabel("Center hue angle [°]",fontsize=15)
for j in range(23,len(a.centery)+1,23):    
    plt.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[j-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(np.linspace(0,360,9))
"""
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
import colclass as col
from colclass import pathes
from scipy import stats as st
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

path=pathes.figpath

"""
decoder labels/colors, this labeling is used for all figures for consistency.
"""
labs=["population vector","von Mises fit","maximum likelihood","maximum fire rate"]
colors=["green","teal","magenta","brown"]
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
plt.subplots_adjust(left=0.08, bottom=0.1, right=0.75, top=0.88, wspace=0.47, hspace=0)

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
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.73, top=0.96, wspace=0.23, hspace=0)
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
plt.subplots_adjust(left=0.05, bottom=0.09, right=0.9, top=0.90, wspace=0.20, hspace=0.20)

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
    dec1m=col.decoder.ml(col1.x,col1.centery,col1.resulty,col1.unitTracker,tabStep=1)
    dec1mf=col.decoder.maxfr(col1.x,col1.resulty,col1.unitTracker)

plt.figure()
plt.ylabel("Induced hue shift [°]",fontsize=30)
plt.xlabel("Hue difference between center and surround [°]",fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(np.linspace(-180,180,9))


plt.plot(dec1v.centSurDif,dec1v.angShift,label=labs[0],color=colors[0])
plt.plot(dec1vm.centSurDif,dec1vm.angShift,label=labs[1],color=colors[1])
plt.plot(dec1m.centSurDif,dec1m.angShift,label=labs[2],color=colors[2])
plt.plot(dec1mf.centSurDif,dec1mf.angShift,label=labs[3],color=colors[3])
plt.legend(loc="lower right",fontsize=25)
plt.subplots_adjust(left=0.08, bottom=0.12, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
plt.savefig(path+"\\all_decoders_together.pdf")

"""

"""
figures 3,4: decoding error for gradient center bw
done in circular_model_different_BW.py
3 is maxfr, 4 is totalsum normalized
"""

"""
Figure 5: Qualitative data reproduction
analysis_gradientvsnromalBW_changingsurround was used !Some debugging necessary
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
fig.text(0.355,0.39,"225",size=20)
fig.text(0.504,0.39,"270",size=20)
fig.text(0.675,0.39,"315",size=20)
fig.text(0.675,0.53,"0",size=20)
fig.text(0.355,0.53,"180",size=20)
fig.text(0.675,0.7,"45",size=20)
fig.text(0.504,0.7,"90",size=20)
fig.text(0.355,0.7,"135",size=20)
plt.xlabel("Center stimulus hue angle [°]",fontsize=30)
plt.ylabel("Unit activity [a.u.]",fontsize=30)
surr=[135,90,45,180,0,225,270,315]
for i in range(0,len(surr)):
    a=col.colmod(1,2.3,0.5,[1.2,0.9],bwType="gradient/sum",phase=22.5,avgSur=surr[i],depInt=[0.2,0.4],depmod=True,stdtransform=False)
    if i==4:
        ax=fig.add_subplot(3,3,i+2)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax2=fig.add_subplot(3,3,i+1)
        
        ax2.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
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
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(45))
    ax.set_xlim([0,360])
plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=1, wspace=0.16, hspace=0.20)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.savefig(path+"\\tuning_curves_best_model.pdf")
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

"""
Manuscript figure 1:  Contextual modulation of population code units
col1=col.colmod(1.5,1,0.5,[1,10],bwType="regular")
fig=plt.figure()
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
for i in range(29,360,30):
    if i==179:
        col="blue"
    else:
        col="black"
    ax1.plot(col1.x[np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],col1.centery[i][np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],color=col,linewidth=1)
    ax2.plot(col1.x[np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],col1.resulty[i][np.where(col1.x==0)[0][0]:np.where(col1.x==360)[0][0]],color=col,linewidth=1)
ax1.axis("off")
ax2.set_xticks(np.linspace(0,360,13))
ax2.set_yticks([])
ax2.tick_params(axis='both', which='major', labelsize=30)
ax2.set_xlabel("Preferred hue angle [°]",fontsize=30)
ax2.set_ylabel("Unit activity [a.u.]",fontsize=30)
plt.gca().set(frame_on=False)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.04, bottom=0.14, right=1, top=1, wspace=0, hspace=0.02)
plt.savefig(path+"\\tuning_curves_unimod.pdf")
"""

"""
Manuscript figure 2:  Population activity thingy (results first figure)
stimang=180
tabStep=10
col1=col.colmod(1.5,1,0.5,[1,10],bwType="regular",avgSur=135)
dec1m=col.decoder.ml(col1.x,col1.centery,col1.resulty,col1.unitTracker,avgSur=135,tabStep=tabStep)#!TABSTEP here 0.5, so indexes are to be doubled
popNS=col.decoder.nosurround(stimang,col1.x,col1.centery)#Population activity without surround
popS=col.decoder.nosurround(stimang,col1.x,col1.resulty)#Population activity without surround
fig=plt.figure()
plt.ylabel("Activation [a.u.]",fontsize=30)
plt.xlabel("Hue angle [°]",fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
ax1=plt.gca()
ax1.xaxis.set_major_locator(MultipleLocator(45))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(22.5))
surmod=(1-col1.surroundy[np.where(col1.x==1)[0][0]:np.where(col1.x==361)[0][0]:10])*max(popNS.noSur)
plt.plot(col1.unitTracker,popNS.noSur,color="purple",label="Population response")
plt.plot(col1.unitTracker,popS.noSur,color="red",label="Modulated response")
plt.plot(col1.unitTracker,surmod,color="blue",label="Surround modulation")
plt.plot(col1.unitTracker,dec1m.surDecoder[int(list(col1.unitTracker).index(stimang)/(tabStep/10))],color="black",linestyle="dashed",label="Decoder's best fit")
plt.yticks([])
plt.legend(prop={'size':20})

plt.gca().set(frame_on=True)#puts the frame of the plot if True, removes if False.
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.04, bottom=0.13, right=0.99, top=1, wspace=0, hspace=0.02)
fig.text(0.49,0.965,"180°",size=20,color="purple")
fig.text(0.08,0.89,"135°",size=20,color="blue")
plt.savefig(path+"\\popact_fig.pdf")
"""

"""
Barplot figure for best model RMS, done in data_params.py
(after loading the json file)
"""

"""
Kappa and depmod distribution for different surround conditions.
Done in data_params.py both linear and polar versions
"""


"""
Tuning curves of the best model as a function of surround (same as manuscript tuning curves of the best fitting model which is for ml,
but this one is for popvec)
Do this for uniform & center-only uniform ml and popvec plots. 
"""
"""
def tuning_curve_plotter(*args,**kwargs):
    fig=plotter.plot_template(auto=True)
    fig.text(0.355,0.39,"225",size=20)
    fig.text(0.504,0.39,"270",size=20)
    fig.text(0.675,0.39,"315",size=20)
    fig.text(0.675,0.53,"0",size=20)
    fig.text(0.355,0.53,"180",size=20)
    fig.text(0.675,0.7,"45",size=20)
    fig.text(0.504,0.7,"90",size=20)
    fig.text(0.355,0.7,"135",size=20)
    plt.xlabel("Center stimulus hue angle [°]",fontsize=30)
    plt.ylabel("Unit activity [a.u.]",fontsize=30)
    surr=[135,90,45,180,0,225,270,315]
    for i in range(0,len(surr)):
        a=col.colmod(*args,**kwargs,avgSur=surr[i])
        if i==4:
            ax=fig.add_subplot(3,3,i+2)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax2=fig.add_subplot(3,3,i+1)
            
            ax2.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)
        else:
            ax=plotter.subplotter(fig,i)
        for j in (1,22,45,67,90,112,135,157,180,202,225,247,270,292,315,337):
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
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(45))
        ax.set_xlim([0,360])
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=1, wspace=0.16, hspace=0.20)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    return
def saver():
    q1=input("Wanna save?\n")
    while q1=="yes":
        q2=input("Please type the figure name.\n")
        plt.savefig(path+"\\"+q2+".pdf")
        print("saved.")
        q1="no"
    return
#mluni
tuning_curve_plotter(1.5,1.7,0.4)
#run one by one 
saver()
#mlcuni
tuning_curve_plotter(1,1.322,1,stdInt=[2,2],bwType="gradient/sum",phase=22.5,depInt=[0.4,0.6],depmod=True,stdtransform=False)
saver()
#pvuni
tuning_curve_plotter(2.3,2.0999999999999996,0.6000000000000001)
saver()
#pvcuni
tuning_curve_plotter(1,2.3,1,stdInt=[1.8,1.8],bwType="gradient/sum",phase=22.5,depInt=[0.4,0.6],depmod=True,stdtransform=False)
saver()
#mlbestmod
tuning_curve_plotter(1,2.3,1,stdInt=[1.2,0.9],bwType="gradient/sum",phase=22.5,depInt=[0.2,0.3999999999999999],depmod=True,stdtransform=False)
saver()
#pvbestmod
tuning_curve_plotter(1,2.3,1,stdInt=[2,1.9],bwType="gradient/sum",phase=22.5,depInt=[0.4,0.6],depmod=True,stdtransform=False)
saver()
"""


"""
Manuscript figure for decoding errors, do as 2x2 subplot where left is sum normalized, right is maxfr normalized with decoder bias 
above and tuning curves below

modsum=col.colmod(1,1,1,stdInt=[2,1],bwType="gradient/sum",depInt=[0.2,0.8],depmod=True,stdtransform=False)

mlsum=col.decoder.ml(modsum.x,modsum.centery,modsum.centery,modsum.unitTracker)
vssum=col.decoder.vecsum(modsum.x,modsum.centery,modsum.unitTracker)
#mfrsum=col.decoder.maxfr(modsum.x,modsum.centery,modsum.unitTracker)
vmsum=col.decoder.vmfit(modsum.x,modsum.centery,modsum.unitTracker)

modfr=col.colmod(1,1,1,stdInt=[2,1],bwType="gradient/max",depInt=[0.2,0.8],depmod=True,stdtransform=False)

mlfr=col.decoder.ml(modfr.x,modfr.centery,modfr.centery,modfr.unitTracker)
vsfr=col.decoder.vecsum(modfr.x,modfr.centery,modfr.unitTracker)
#mfrfr=col.decoder.maxfr(modfr.x,modfr.centery,modfr.unitTracker)
vmfr=col.decoder.vmfit(modfr.x,modfr.centery,modfr.unitTracker)

fig=plt.figure()

ax1=fig.add_subplot(2,2,1)
ax1.set_title("Total area normalized",fontsize=25)
ax1.set_ylabel("Decoding Error [°]",fontsize=20)
ax1.set_xlabel("Center stiumulus hue angle [°]",fontsize=20)
ax1.set_xticks(np.linspace(0,360,9))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))
ax1.set_ylim([-6,6])

ax2=fig.add_subplot(2,2,2)
ax2.set_title("Maximum fire rate normalized",fontsize=25)
ax2.set_xticks(np.linspace(0,360,9))
ax2.set_yticks([])
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.xaxis.set_major_locator(MultipleLocator(90))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_minor_locator(MultipleLocator(45))
ax2.set_ylim([-6,6])


labs=["population vector","von Mises fit","maximum likelihood"]
colors=["green","teal","magenta"]
sumdecs=["vssum","vmsum","mlsum"]
frdecs=["vsfr","vmfr","mlfr"]
for i in range(0,3):
    ax1.plot(np.array(eval(sumdecs[i]).centSurDif)+180,eval(sumdecs[i]).angShift,label=labs[i],color=colors[i])
    ax2.plot(np.array(eval(frdecs[i]).centSurDif)+180,eval(frdecs[i]).angShift,label=labs[i],color=colors[i])
ax2.legend(loc="best",prop={'size':20},bbox_to_anchor=(1,1))

ax3=fig.add_subplot(2,2,3)
ax3.set_ylabel("Unit activity [a.u.]",fontsize=20)
ax3.set_xlabel("Center stiumulus hue angle [°]",fontsize=20)
ax3.set_xticks(np.linspace(0,360,9))
ax3.xaxis.set_major_locator(MultipleLocator(90))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_minor_locator(MultipleLocator(45))
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.set_yticks([])
ax3.set_xlim([0,360])

ax4=fig.add_subplot(2,2,4)
ax4.set_xticks(np.linspace(0,360,9))
ax4.set_yticks([])
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.xaxis.set_major_locator(MultipleLocator(90))
ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax4.xaxis.set_minor_locator(MultipleLocator(45))
ax4.set_xlim([0,360])

for j in range(23,len(modsum.centery)+1,23):
    ax3.plot(modsum.x[np.where(modsum.x==0)[0][0]:np.where(modsum.x==360)[0][0]],modsum.centery[j-1][np.where(modsum.x==0)[0][0]:np.where(modsum.x==360)[0][0]],color="black",linewidth=1)
    ax4.plot(modfr.x[np.where(modfr.x==0)[0][0]:np.where(modfr.x==360)[0][0]],modfr.centery[j-1][np.where(modfr.x==0)[0][0]:np.where(modfr.x==360)[0][0]],color="black",linewidth=1)

mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.74, top=0.95, wspace=0.1, hspace=0.28)

plt.savefig(path+"\\decoding_error_different_models.pdf")
"""

"""
Phase effect of center and surround Kappa
in colclass incorporated as the functions centphase and surphase
"""

"""
FIGURES TO PUT:
Parameter distribution DONE BUT MANUAL
Best fit of both models in same plot without data DONE BUT MANUAL
Non-uniform tuning curves best fit ML DONE
Parameter scan description (methods oder first sentence of the results scan subset.)
Skype Freitag 14.00 (skype thomas.wachtler)
"""
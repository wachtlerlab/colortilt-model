# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:57:48 2019

@author: Ibrahim Alperen Tunc
"""
'''
The analysis of color tilt in different surrounds for given surround angle in all model types.
Model types are uniform-non uniform model. Non-uniform model can be normalized by maximum unit activity or the total unit activity,
each of these variants can also have the uniform or non-uniform surround modulation. All non-uniformities have the phase of 22.5, that 
strongest surround suppression and narrowest tuning curve is in 22.5°
Small issues with figure 3, which can be sorted out later.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import colclass as col
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

surrInt=(135,90,45,180,0,225,270,315)#The order of the surround stimuli angles is important for the subplotting.
dec=["vm","vs","ml"]#decoders list, vm=von Mises fit, vs=population vector, ml=maximum likelihood
params=["maanshi","csdmaanshi","mianshi","csdmianshi","decprec"]#color tilt curve parameters. maanshi=maximum angular shift, mianshi=minimum (negative) angular shift, csd parameters give the center-surround difference angle at maanshi/mianshi

dictPar=param_dict(dec,params)#preallocate the dictionary of decoders and parameters

def plot_adjuster():
    """Adjust the plot size automatically for each plot in successive order, atm it is optimized only for figure 5, this would but also work
    for other models.
    """
    while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
        if plt.waitforbuttonpress(0):
            plt.close()
            break
    
    plt.subplots_adjust(left=0.16, bottom=0.09, right=0.74, top=0.88, wspace=0.05, hspace=0.08)
    while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
        if plt.waitforbuttonpress(0):
            plt.close()
            break
    
    plt.subplots_adjust(left=0.14, bottom=0.09, right=0.8, top=0.88, wspace=0.1, hspace=0.1)
    while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
        if plt.waitforbuttonpress(0):
            plt.close()
            break
    return

def surr_analysis(surr,grType,depmod):#Make plot titles for each condition (bwType,depmod)!
    """Color tilt plotter function for the given model type:
        This function creates the plots of exemplary population activity at 100° center stimulus hue, tuning curves of the model units 
        after surround modulation as well as the color tilt curves for all decoders.
        
        Parameters
        ----------
        surr: float. The surround stimulus hue angle in degrees. This parameter can also be given as tuple/list.
        grType: string. The model type. Can be "norm" (uniform model), "gf" (non-uniform maximum unit activity normalized model), 
        or "gs" (non-uniform total unit activity normalized model).
        depmod: booelan. If true, surround modulation is also non-uniform with different suppression strength depending on the surround
        stimulus hue angle
        
        Note that phase of the non-uniform model is 22.5°, meaning that the strongest suppression is at surround angle=22.5° and the center
        unit with preferred hue angle of 22.5° has the narrowest tuning. These parameters change then as a gradient within a 180° period. 
        
        Returns
        -------
        dictPar: dictionary. The dictionary with important color tilt curve values for each decoder and surround.
    """
    titmap=["uniform model","non-uniform model maximum activity normalized","non-uniform model total activity normalized","uniform surround modulation","non-uniform surround modulation"]#Plot title map for different grType and depMod
    if grType=="norm":
        tit=titmap[0]
    if grType=="gf":
        tit=titmap[1]
    if grType=="gs": 
        tit=titmap[2]
    if depmod==False:
        tit2=titmap[3]
    if depmod==True:
        tit2=titmap[4]
    fig1=plotter.plot_template(auto=True)
    plt.title("Hue shift for the %s with %s"%(tit,tit2),y=1.08,fontsize=20)#1st plot is the color tilt curve, subplots with different surrounds
    plt.xlabel("Hue difference between center and surround [°]",fontsize=15)
    plt.ylabel("Induced hue shift [°]",fontsize=15)
    
    fig2=plotter.plot_template(auto=True)#2nd plot is the population activity for center stimulus=100°, subplots with different surrounds
    plt.xlabel("Center stimulus angle [°]",fontsize=15)
    plt.ylabel("Population activity [a.u.]",fontsize=15,x=-0.1)
    plt.title("Pop. activity (100°) %s with %s"%(tit,tit2),x=0.55,y=1.08,fontsize=20)
    
    fig3=plotter.plot_template(auto=True)#Tuning curves of the chromatic filters, subplots with different surrounds
    plt.xlabel("Center stimulus angle [°]",fontsize=15)
    plt.ylabel("Unit activity [a.u.]",fontsize=15,x=-0.1)
    plt.title("Tuning curves of the %s with %s"%(tit,tit2),y=1.08,fontsize=20)
    
    colMap=("blue","red","black")#color map for each decoder in fig 1.
    labMap=("von Mises fit","vector sum","maximum likelihood")#label map for legend
    for i in range(0,len(surr)):#start of the plotting for each surround
        if i==0:
            print("computing 1st surround angle")
        """
        Create the chosen model
        """
        if grType=="norm":
            colMod=col.colmod(std2kappa(60,1,1.5),1,0.5,[60,70],avgSur=surr[i],bwType="regular",depmod=depmod,phase=22.5)
        if grType=="gf":
            colMod=col.colmod(std2kappa(60,1,1.5),1,0.5,[60,70],avgSur=surr[i],bwType="gradient/max",depmod=depmod,phase=22.5)    
        if grType=="gs":
            colMod=col.colmod(std2kappa(60,1,1.5),1,0.5,[60,70],avgSur=surr[i],bwType="gradient/sum",depmod=depmod,phase=22.5)
        
        """
        Different decoders to be used in fig. 1
        """
        decvm=col.decoder.vmfit(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=surr[i])#von Mises fit
        decvs=col.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=surr[i])#population vector
        print("takes some time")
        decml=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=surr[i])#maximum likelihood
        
        ax1=plotter.subplotter(fig1,i)#Create the passing subplot for the surround hue for the moment.
        """
        Plottings in the subplot for all decoders
        """
        ax1.plot(decvm.centSurDif,decvm.angShift,color=colMap[0],label=labMap[0])
        ax1.plot(decvs.centSurDif,decvs.angShift,color=colMap[1],label=labMap[1])
        ax1.plot(decml.centSurDif,decml.angShift,color=colMap[2],label=labMap[2])
        """
        Subplot formatting
        """
        ax1.set_xticks(np.linspace(-180,180,9))#x ticks are between +-180, in obliques and cardinal angles
        ax1.tick_params(axis='both', which='major', labelsize=15)#major ticks are bigger labeled
        ax1.xaxis.set_major_locator(MultipleLocator(90))#major ticks are set at 0,90,180,...
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.xaxis.set_minor_locator(MultipleLocator(45))#minor ticks are set at 45,135,...
        ax1.set_ylim(bottom=-30,top=30)#y axis is between +-30
        
        """
        Same manipulations for the fig 2.
        """
        ax2=plotter.subplotter(fig2,i)
        ax2.plot(np.linspace(0,359,360),decvm.surDecoder[np.where(colMod.unitTracker==100)[0][0]],color="black",label="with surround modulation")
        ax2.plot(np.linspace(0,359,360),col.decoder.nosurround(100,colMod.x,colMod.centery).noSur,color="grey",label="without surround modulation")
        ax2.set_xticks(np.linspace(0,360,9))
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.xaxis.set_major_locator(MultipleLocator(90))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax2.xaxis.set_minor_locator(MultipleLocator(45))
        
        """
        Legends of the figures 1/2
        """
        if i==7:
            print("putting legends")
            ax1.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
            ax2.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
        
        """
        Same work done for fig 3. Problem with this figure, the figure is but unnecessary, so can be discarded for the time being
        """
        ax3=plotter.subplotter(fig3,i)
        for l in range(0,len(surr)):#This for loops plots for each subplot all tuning curves with preferred hue angle as in surr list!   
            if l==4:#As there is no unit with preferred hue angle of 0, for surround=0° the activity in 360° is given
                ax3.plot(np.linspace(0,359.9,3600),colMod.resulty[np.where(colMod.unitTracker==360)[0][0]][np.where(colMod.x==0)[0][0]:np.where(colMod.x==360)[0][0]],color="black",linewidth=1.0)
            else:
                ax3.plot(np.linspace(0,359.9,3600),colMod.resulty[np.where(colMod.unitTracker==surr[l])[0][0]][np.where(colMod.x==0)[0][0]:np.where(colMod.x==360)[0][0]],color="black",linewidth=1.0)
        ax3.set_xticks(np.linspace(0,360,9))
        ax3.xaxis.set_major_locator(MultipleLocator(90))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax3.xaxis.set_minor_locator(MultipleLocator(45))
        ax3.tick_params(axis='x', which='major', labelsize=15)
        ax3.tick_params(axis='y', which='major', labelsize=15)
        
        if i in (0,3):
            ax1.axes.get_xaxis().set_visible(False)            
            ax2.axes.get_xaxis().set_visible(False)

        elif i in (6,7):
            ax1.axes.get_yaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
        elif i in (1,2):
            ax1.axes.get_yaxis().set_visible(False)
            ax1.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)            
        elif i==5:
            ax1.axes.get_yaxis().set_visible(True)
            ax1.axes.get_xaxis().set_visible(True)
            ax2.axes.get_yaxis().set_visible(True)
            ax2.axes.get_xaxis().set_visible(True)
        
        """
        Mayhem is the collection of all important color tilt values for each decoder and each surround condition.
        """
        mayhem=(surr[i],max(decvm.angShift),max(decvs.angShift),max(decml.angShift),\
              decvm.centSurDif[np.where(decvm.angShift==max(decvm.angShift))[0][0]],decvs.centSurDif[np.where(decvs.angShift==max(decvs.angShift))[0][0]],decml.centSurDif[np.where(decml.angShift==max(decml.angShift))[0][0]],\
              min(decvm.angShift),min(decvs.angShift),min(decml.angShift),\
              decvm.centSurDif[np.where(decvm.angShift==min(decvm.angShift))[0][0]],decvs.centSurDif[np.where(decvs.angShift==min(decvs.angShift))[0][0]],decml.centSurDif[np.where(decml.angShift==min(decml.angShift))[0][0]],\
              decvm.angShift[decvm.centSurDif.index(0)],decvs.angShift[decvs.centSurDif.index(0)],decml.angShift[decml.centSurDif.index(0)])
        
        """
        Print the summary values to give feedback that function is up&working, these summary values also cause the waiting time to go faster, as reading the summary is a nice hobby inbetween.
        """
        print("Summary: surround angle=%s \
            maximum angular shift: vm=%s, vs=%s, ml=%s\
            csd at max ang. shift: vm=%s, vs=%s, ml=%s\
            minimum angular shift: vm=%s, vs=%s, ml=%s\
            csd at min ang. shift: vm=%s, vs=%s, ml=%s\
            decoding precision stim=surround: vm=%s, vs=%s, ml=%s"\
            %(mayhem))
        
        
        """
        Create the colortilt values dictionary
        """
        for j in range(0,len(dec)):
            for k in range(0,len(params)):
                dictPar[dec[j]][params[k]].update({surr[i]:mayhem[j+3*k+1]})
        
        print("computing next surround angle")
        print("")
    return dictPar

"""
The colortilt plotter function for each possible model combination
"""
dictParNorm=surr_analysis(surrInt,grType="norm",depmod=False)
dictParNormDep=surr_analysis(surrInt,grType="norm",depmod=True)    
    
dictPargf=surr_analysis(surrInt,grType="gf",depmod=False)    
dictPargfDep=surr_analysis(surrInt,grType="gf",depmod=True)    

dictPargs=surr_analysis(surrInt,grType="gs",depmod=False) 
dictPargsDep=surr_analysis(surrInt,grType="gs",depmod=True)#FIGURE 5 
plot_adjuster()

"""  
These are the artifacts from last working session, where each title and axis labels were given manually. 
decoded hue shift for center population gradient BW maximum fire rate normalized gradient surround modulation depth
center surround difference
angular shift
exemplary population activity in 100 degrees center population gradient BW maximum fire rate normalized gradient surround modulation depth
stimulus angle 
population activity
activity of center units with gradient BW maximum fire rate normalized after gradient surround modulation
stimulus angle
unit activity
"""
"""
*Development notes
Make a huge results plot (unit activity after surround modulation, population activity in a given angle, 8 subplots for all orientations)
by writing a function which takes as input the decoder type and model type etc.
Include also the ml model to the analysis

Subplots same x and y scala! done
There is a problem with decoders when center=surround angle, shifts are too huge! compare with circular_model_different_BW.py! no prob we were looking rather at angShift=0!
Send subplot mails for all cases (normal BW, and also gradient version)
Make unit activity plot lines thinner, make the 0 and 90 degrees units in a different color (all others grey, these two black e.g)

analyze the gsdepmod case closer, change the phase of center BW and modulation depth, so that the maximum of depmod and minimum BW is at 22.5 grad and periodicity is again 180 degrees!
try to put the middle subplot angle text in the middle, make angshift y axis limits +-29. The data will be sent as e-mail, so that i can quantitatively find out the max min values for each
and make a parameter scan for gsdepmod (parameters ksur, bw gradient(upper,lower limits), modulation depth(upper,lower limits), phase of bw and moddepth (assumption in first glance, that it is at 22.5 degrees (blue-yellow axis)))
"""


"""
Codes to be salvaged.
    plt.figure()
    plt.plot(vmFitNorm.centSurDif,vmFitNorm.angShift,vecSumNorm.centSurDif,vecSumNorm.angShift)
    plt.title("Normal BW induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitNorm.surDecoder[100],"r")
    #plt.plot(vmFitGraFR.surDecoder[100],"b")
    plt.plot(vmFitGraSum.surDecoder[100],"g")
    plt.plot(col.decoder.nosurround(100,colModNorm.x,colModNorm.centery).noSur,"k")
    plt.title("vm Fit population response to 100° at different situations")
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
        
    plt.figure()
    plt.plot(vecSumNorm.surDecoder[100],"r")
    #plt.plot(vecSumGraFR.surDecoder[100],"b")
    plt.plot(vecSumGraSum.surDecoder[100],"g")
    plt.plot(col.decoder.nosurround(100,colModNorm.x,colModNorm.centery).noSur,"k")
    plt.title("Vector sum population response to 100° at different situations")
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitGraFR.centSurDif,vmFitGraFR.angShift,vecSumGraFR.centSurDif,vecSumGraFR.angShift)
    plt.title("Gradient BW with FR normalization induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitGraSum.centSurDif,vmFitGraSum.angShift,vecSumGraSum.centSurDif,vecSumGraSum.angShift)
    plt.title("Gradient BW with sum normalization induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    
    if i!=4:
        ax1=fig1.add_subplot(2,3,i+1)
        ax1.plot(vmFitGraSum.centSurDif,vmFitGraSum.angShift,vecSumGraSum.centSurDif,vecSumGraSum.angShift)
        ax1.set_title("%s"%(surrAng[i]))
        ax1.set_ylim(bottom=-20,top=20)
        ax1.set_xticks(np.linspace(-180,180,9))
        ax2=fig2.add_subplot(2,3,i+1)
        ax2.plot(vmFitGraFR.centSurDif,vmFitGraFR.angShift,vecSumGraFR.centSurDif,vecSumGraFR.angShift)
        ax2.set_title("%s"%(surrAng[i]))
        ax2.set_ylim(bottom=-20,top=20)
        ax2.set_xticks(np.linspace(-180,180,9))
    else:
        ax1=fig1.add_subplot(2,3,i+2)
        ax1.plot(vmFitGraSum.centSurDif,vmFitGraSum.angShift,vecSumGraSum.centSurDif,vecSumGraSum.angShift)
        ax1.set_title("%s"%(surrAng[i]))
        ax1.set_ylim(bottom=-20,top=20)
        ax1.set_xticks(np.linspace(-180,180,9))
        ax2=fig2.add_subplot(2,3,i+2)
        ax2.plot(vmFitGraFR.centSurDif,vmFitGraFR.angShift,vecSumGraFR.centSurDif,vecSumGraFR.angShift)
        ax2.set_title("%s"%(surrAng[i]))       
        ax2.set_ylim(bottom=-20,top=20)
        ax2.set_xticks(np.linspace(-180,180,9))
    """
"""
fig1.tight_layout()
fig2.tight_layout()
fig1.show()
fig2.show()
"""
#Different BW are rather shifting the maximum hue value in csd-axis!

"""
Change modulation depth!

dictPar=param_dict()

surrAng=[135,90,45,180,0]
fig3=plt.figure()
fig3.legend(handles=[bPatch,oPatch])
plt.axis("off")
plt.title("different depmod",y=1.08)   

for i in range(0,len(surrAng)):#should i pickle dump all variables here?
    if i==0:
        print("computing 1st surround angle")
    colModNorm=col.colmod(std2kappa(60,1,1.5),1,0.537,[60,70],avgSur=surrAng[i],bwType="regular",depmod=True)
    colModGraMFR=col.colmod(std2kappa(60,1,1.5),1,0.537,[60,70],avgSur=surrAng[i],bwType="gradient/max",depmod=True)
    colModGraSum=col.colmod(std2kappa(60,1,1.5),1,0.537,[60,70],avgSur=surrAng[i],bwType="gradient/sum",depmod=True)
    vmFitNorm=col.decoder.vmfit(colModNorm.x,colModNorm.resulty,colModNorm.unitTracker,avgSur=surrAng[i])
    vecSumNorm=col.decoder.vecsum(colModNorm.x,colModNorm.resulty,colModNorm.unitTracker,avgSur=surrAng[i])
    vmFitGraFR=col.decoder.vmfit(colModGraMFR.x,colModGraMFR.resulty,colModGraMFR.unitTracker,avgSur=surrAng[i])    
    vecSumGraFR=col.decoder.vecsum(colModGraMFR.x,colModGraMFR.resulty,colModGraMFR.unitTracker,avgSur=surrAng[i])
    vmFitGraSum=col.decoder.vmfit(colModGraSum.x,colModGraSum.resulty,colModGraSum.unitTracker,avgSur=surrAng[i])    
    vecSumGraSum=col.decoder.vecsum(colModGraSum.x,colModGraSum.resulty,colModGraSum.unitTracker,avgSur=surrAng[i])
    
    plt.figure()
    plt.plot(vmFitNorm.centSurDif,vmFitNorm.angShift,vecSumNorm.centSurDif,vecSumNorm.angShift)
    plt.title("Normal BW different depmod induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitNorm.surDecoder[100],"r")
    #plt.plot(vmFitGraFR.surDecoder[100],"b")
    plt.plot(vmFitGraSum.surDecoder[100],"g")
    plt.plot(col.decoder.nosurround(100,colModNorm.x,colModNorm.centery).noSur,"k")
    plt.title("vm Fit population response to 100° at different situations")
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
        
    plt.figure()
    plt.plot(vecSumNorm.surDecoder[100],"r")
    #plt.plot(vecSumGraFR.surDecoder[100],"b")
    plt.plot(vecSumGraSum.surDecoder[100],"g")
    plt.plot(col.decoder.nosurround(100,colModNorm.x,colModNorm.centery).noSur,"k")
    plt.title("Vector sum population response to 100° at different situations")
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitGraFR.centSurDif,vmFitGraFR.angShift,vecSumGraFR.centSurDif,vecSumGraFR.angShift)
    plt.title("Gradient BW with FR normalization + different depmod induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    plt.figure()
    plt.plot(vmFitGraSum.centSurDif,vmFitGraSum.angShift,vecSumGraSum.centSurDif,vecSumGraSum.angShift)
    plt.title("Gradient BW with sum normalization + different depmod induced hue shift for surround angle %s"%(surrAng[i]))
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    
    if i!=4:
        ax3=fig3.add_subplot(2,3,i+1)
        ax3.plot(vmFitNorm.centSurDif,vmFitNorm.angShift,vecSumNorm.centSurDif,vecSumNorm.angShift)
        ax3.set_title("surround agle=%s, modulation depth=%s"%(surrAng[i],np.round(depth_modulator([0.2,0.6],surrAng[i]),2)))
        ax3.set_ylim(bottom=-20,top=20)
        ax3.set_xticks(np.linspace(-180,180,9))
    else:
        ax3=fig3.add_subplot(2,3,i+2)
        ax3.plot(vmFitNorm.centSurDif,vmFitNorm.angShift,vecSumNorm.centSurDif,vecSumNorm.angShift)
        ax3.set_title("surround agle=%s, modulation depth=%s"%(surrAng[i],np.round(depth_modulator([0.2,0.6],surrAng[i]),2)))
        ax3.set_ylim(bottom=-20,top=20)
        ax3.set_xticks(np.linspace(-180,180,9))
    print("Summary: surround angle=%s \
        maximum angular shift: vmNorm=%s, vmGraSum=%s, vmGraFR=%s, vsNorm=%s, vsGraSum=%s, vsGraFR=%s\
        csd at max ang. shift: vmNorm=%s, vmGraSum=%s, vmGraFR=%s, vsNorm=%s, vsGraSum=%s, vsGraFR=%s\
        minimum angular shift: vmNorm=%s, vmGraSum=%s, vmGraFR=%s, vsNorm=%s, vsGraSum=%s, vsGraFR=%s\
        csd at min ang. shift: vmNorm=%s, vmGraSum=%s, vmGraFR=%s, vsNorm=%s, vsGraSum=%s, vsGraFR=%s"\
        %(surrAng[i],max(vmFitNorm.angShift),max(vmFitGraSum.angShift),max(vmFitGraFR.angShift),max(vecSumNorm.angShift),max(vecSumGraSum.angShift),max(vecSumGraFR.angShift),\
          vmFitNorm.centSurDif[np.where(vmFitNorm.angShift==max(vmFitNorm.angShift))[0][0]],vmFitGraSum.centSurDif[np.where(vmFitGraSum.angShift==max(vmFitGraSum.angShift))[0][0]],\
          vmFitGraFR.centSurDif[np.where(vmFitGraFR.angShift==max(vmFitGraFR.angShift))[0][0]],vecSumNorm.centSurDif[np.where(vecSumNorm.angShift==max(vecSumNorm.angShift))[0][0]],\
          vecSumGraSum.centSurDif[np.where(vecSumGraSum.angShift==max(vecSumGraSum.angShift))[0][0]],vecSumGraFR.centSurDif[np.where(vecSumGraFR.angShift==max(vecSumGraFR.angShift))[0][0]],\
          min(vmFitNorm.angShift),min(vmFitGraSum.angShift),min(vmFitGraFR.angShift),min(vecSumNorm.angShift),min(vecSumGraSum.angShift),min(vecSumGraFR.angShift),\
          vmFitNorm.centSurDif[np.where(vmFitNorm.angShift==min(vmFitNorm.angShift))[0][0]],vmFitGraSum.centSurDif[np.where(vmFitGraSum.angShift==min(vmFitGraSum.angShift))[0][0]],\
          vmFitGraFR.centSurDif[np.where(vmFitGraFR.angShift==min(vmFitGraFR.angShift))[0][0]],vecSumNorm.centSurDif[np.where(vecSumNorm.angShift==min(vecSumNorm.angShift))[0][0]],\
          vecSumGraSum.centSurDif[np.where(vecSumGraSum.angShift==min(vecSumGraSum.angShift))[0][0]],vecSumGraFR.centSurDif[np.where(vecSumGraFR.angShift==min(vecSumGraFR.angShift))[0][0]]))
    print("computing next surround angle")
    if i==4:
        print("all done!")
    print("")

colorMap=["red","blue","black","green","purple"]   
fig=plt.figure()
plt.axis("off")
for i in range(0,len(dec)):
    for j in range(0,len(params)):
        x=dictPar[dec[i]][params[j]].keys()
        y=dictPar[dec[i]][params[j]].values()
        ax=fig.add_subplot(2,3,i+1)
        ax.plot(x,y,'.',color=colorMap[j],label=params[j])
        ax.set_title("params of the decoder %s"%(dec[i]))
        ax.set_ylim(bottom=-80,top=80)
        ax.set_xticks(np.linspace(0,180,5))
ax.legend(loc="upper left", bbox_to_anchor=(1,1))    
"""    


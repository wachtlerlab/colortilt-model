# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:23:30 2019

@author: Ibrahim Alperen Tunc
"""
"""
The plots of the decoding errors for each decoder in the non-uniform model
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import sys
sys.path.insert(0,r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python")#!Change the directory where colclass.py and supplementary_functions.py are
from supplementary_functions import std2kappa, param_dict, plotter
import colclass
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

'''
The Bandwiths of the center units vary: for 0° BW 60°, then increasing gradually to 70° until 90° unit then same thing again!
This code is transferred to colclass.py
'''
"""
x=np.ndarray.round(np.linspace(-60,420,num=4801),2)#round all these steps to .1 decimals
kappaDown=std2kappa(60,1,1.5)[0]
kappaUp=std2kappa(70,1,1.5)[0]
'''
Write a cosine function with maximum at bwDown and minimum at bwUp with double periodicity, by this way the 0 unit has bwdown value,
90 unit has bwup value!
'''
startAvg=1
endAvg=360
kapMod=(kappaDown-kappaUp)/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,360)))+kappaUp+(kappaDown-kappaUp)/2#Kappa Modulator
#Logic behind: (kappaDown-kappaUp)/2 is to change the difference of max and min to interval [kappaDown;KappaUp]; np.cos(2*...) 
#is to increase the frequency to 2, so that at 90 unit function already in minimum. then +kappaUp+(kappaDown-kappaUp)/2 to push
#minimum of function to the kappaUp value (with +(kappaDown-kappaUp)/2 value at 0 and then +kappaUp)

totalAvg=[]
centery=[]
for i in range(0,endAvg):
    avg=startAvg+i
    totalAvg.append(avg)
    y=1/(2*np.pi)*np.e**(kapMod[i]*np.cos(np.deg2rad(x)-np.deg2rad(avg)))
    centery.append(y/max(y))
unitTracker=np.arange(startAvg,startAvg+len(totalAvg))

#!!! Normalizing the units to the area causes the maxima to be different. should i normalize regarding the max value? I guess normalizing
#to max value makes more sense. Also the plots have a sine-like wave underneath (plt.plot(centery))
"""

"""
Wrapper function, where inputs are decoder type, surround etc is and output is decoder classes.
"""
def col_BW_wrapper(avgS,bwType,dec,depmod=False):
    """The wrapper function for different decoders in the non-uniform model:
        This function returns the decoder variables for the given model parameters as well as the decoder type.
        
        Parameters
        ----------
        avgS: float. The surround stimulus hue angle in degrees.
        bwType: string. The type of the non-uniform model normalization. "gf" gives the maximum activity normalized model,
        "gs" gives the total activity normalized model.
        dec: string. The type of the decoder. "vs" is the population vector decoder, "vm" is the von Mises fit decoder,
        "ml" is the maximum likelihood decoder, "mf" is the maximum fire rate decoder.
        depmod: boolean, optional. If this parameter is True, then the modulation depth is different as a function of 
        the surround hue angle. Otherwise, the modulation depth is the same for all surround conditions.
        
        Returns
        -------
        decNoSur: class object. Returns the chosen decoder object created by the given non-uniform model.
    """
    colMod=colclass.colmod(std2kappa(60,1.4,1.5),1,0.5,[60,70],avgSur=avgS,bwType=bwType,depmod=depmod)
    if dec=="vs":
        #decSur=colclass.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=avgS)
        decNoSur=colclass.decoder.vecsum(colMod.x,colMod.centery,colMod.unitTracker,avgSur=avgS)
    if dec=="vm":
        #decSur=colclass.decoder.vmfit(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=avgS)
        decNoSur=colclass.decoder.vmfit(colMod.x,colMod.centery,colMod.unitTracker,avgSur=avgS)
    if dec=="ml":
        #decSur=colclass.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=avgS)
        decNoSur=colclass.decoder.ml(colMod.x,colMod.centery,colMod.centery,colMod.unitTracker,avgSur=avgS)        
    if dec=="mf":
        #decSur=colclass.decoder.maxfr(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=avgS)
        decNoSur=colclass.decoder.maxfr(colMod.x,colMod.centery,colMod.unitTracker,avgSur=avgS)
    #return decSur,decNoSur
    return decNoSur

'''
Now the analysis, how well decoders perform:
'''
def surround_plotter(surr,grType,depmod=False):
    """The plotter of the decoding error:
        This function plots the decoding errors of each surround condition for all decoders. Important values of the decodings are also returned
        as output
        
        Parameters
        ----------
        surr: float. The surround hue angle in degrees.
        grType: string. The normalization type of the non-uniform model. "gf" is for the maximum activity normalization, "gs" is for the total
        activity normalization.
        depmod: boolean, optional. If this parameter is True, then the modulation depth is different as a function of 
        the surround hue angle. Otherwise, the modulation depth is the same for all surround conditions.
        
        Returns
        -------
        surrPar: dictionary. Returns the color tilt values for each decoder and each surround condition. "maanshi"=maximum angular shift, 
        "csdmaanshi"=center-surround hue angle difference in maximum angular shift, "mianshi"=minimum angular shift,
        "csdmianshi"=center-surround hue angle difference in minimum angular shift, "vs"=population vector decoder, "vm"=von Mises fit decoder,
        "ml"=maximum likelihood decoder, "mf"=maximum fire rate decoder.
    """
    fig=plotter.plot_template(auto=True)
    plt.xlabel("Hue difference between center and surround [°]",fontsize=15)
    plt.ylabel("Decoding error [°]",fontsize=15)
    surrPar={}
    if grType=="gf":
        plt.title("Decoding errors of the non-uniform model maximum activity normalized",y=1.08,fontsize=20)
    if grType=="gs":
        plt.title("Decoding errors of the non-uniform model total activity normalized",y=1.08,fontsize=20)
    for i in range(0,len(surr)):
        if grType=="gf":
            vsgfns=col_BW_wrapper(surr[i],"gradient/max","vs")#vector sum gradient/max no surround
            vmgfns=col_BW_wrapper(surr[i],"gradient/max","vm")#same as above but vmFit
            mfgfns=col_BW_wrapper(surr[i],"gradient/max","mf")#same as above but maxfr
            print("it will take some time...")
            mlgfns=col_BW_wrapper(surr[i],"gradient/max","ml")
            
            ax1=plotter.subplotter(fig,i)
            ax1.set_xticks(np.linspace(-180,180,9))#sets x ticks between +-180 with 45° spacing
            ax1.tick_params(axis='both', which='major', labelsize=15)#makes tick label size in both axes 15
            ax1.xaxis.set_major_locator(MultipleLocator(90))#major ticks at cardinal angles
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax1.xaxis.set_minor_locator(MultipleLocator(45))#minor ticks at oblique angles
            ax1.plot(vsgfns.centSurDif,vsgfns.angShift,color="red",label="vector sum")#plot all decoder values in the following 4 lines
            ax1.plot(vmgfns.centSurDif,vmgfns.angShift,color="blue",label="von Mises fit")
            ax1.plot(mlgfns.centSurDif,mlgfns.angShift,color="black",label="maximum likelihood")
            ax1.plot(mfgfns.centSurDif,mfgfns.angShift,color="black",label="maximum fire rate")
            if i==7:
                ax1.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)#insert the legend in the last subplot
               
            decoders=["vs","vm","ml","mf"]
            params=["maanshi","csdmaanshi","mianshi","csdmianshi"]
            dictPar=param_dict(decoders,params)
            mayhem=(max(vsgfns.angShift),vsgfns.centSurDif[vsgfns.angShift.index(max(vsgfns.angShift))],min(vsgfns.angShift),vsgfns.centSurDif[vsgfns.angShift.index(min(vsgfns.angShift))],\
                    max(vmgfns.angShift),vmgfns.centSurDif[vmgfns.angShift.index(max(vmgfns.angShift))],min(vmgfns.angShift),vmgfns.centSurDif[vmgfns.angShift.index(min(vmgfns.angShift))],\
                    max(mlgfns.angShift),mlgfns.centSurDif[mlgfns.angShift.index(max(mlgfns.angShift))],min(mlgfns.angShift),mlgfns.centSurDif[mlgfns.angShift.index(min(mlgfns.angShift))],\
                    max(mfgfns.angShift),mfgfns.centSurDif[mfgfns.angShift.index(max(mfgfns.angShift))],min(mfgfns.angShift),mfgfns.centSurDif[mfgfns.angShift.index(min(mfgfns.angShift))])
            for j in range(0,len(decoders)):
                for k in range(0,len(params)):
                 dictPar[decoders[j]][params[k]].update({surr[i]:mayhem[k+4*j]})   
            surrPar.update({surr[i]:dictPar})
         
        if grType=="gs":
            vsgsns=col_BW_wrapper(surr[i],"gradient/sum","vs")#same as before but gradient/sum
            vmgsns=col_BW_wrapper(surr[i],"gradient/sum","vm")
            mfgsns=col_BW_wrapper(surr[i],"gradient/sum","mf")
            print("it will take some time...")
            mlgsns=col_BW_wrapper(surr[i],"gradient/sum","ml")
            
            ax1=plotter.subplotter(fig,i)
            ax1.set_xticks(np.linspace(-180,180,9))
            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax1.xaxis.set_major_locator(MultipleLocator(90))
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax1.xaxis.set_minor_locator(MultipleLocator(45))
            ax1.plot(vsgsns.centSurDif,vsgsns.angShift,color="red",label="vector sum")
            ax1.plot(vmgsns.centSurDif,vmgsns.angShift,color="blue",label="von Mises fit")
            ax1.plot(mlgsns.centSurDif,mlgsns.angShift,color="black",label="maximum likelihood")
            ax1.plot(mfgsns.centSurDif,mfgsns.angShift,color="green",label="maximum fire rate")
            if i==7:
                ax1.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
                #fig.tight_layout()
            decoders=["vs","vm","ml","mf"]
            params=["maanshi","csdmaanshi","mianshi","csdmianshi"]
            dictPar=param_dict(decoders,params)
            mayhem=(max(vsgsns.angShift),vsgsns.centSurDif[vsgsns.angShift.index(max(vsgsns.angShift))],min(vsgsns.angShift),vsgsns.centSurDif[vsgsns.angShift.index(min(vsgsns.angShift))],\
                    max(vmgsns.angShift),vmgsns.centSurDif[vmgsns.angShift.index(max(vmgsns.angShift))],min(vmgsns.angShift),vmgsns.centSurDif[vmgsns.angShift.index(min(vmgsns.angShift))],\
                    max(mlgsns.angShift),mlgsns.centSurDif[mlgsns.angShift.index(max(mlgsns.angShift))],min(mlgsns.angShift),mlgsns.centSurDif[mlgsns.angShift.index(min(mlgsns.angShift))],\
                    max(mfgsns.angShift),mfgsns.centSurDif[mfgsns.angShift.index(max(mfgsns.angShift))],min(mfgsns.angShift),mfgsns.centSurDif[mfgsns.angShift.index(min(mfgsns.angShift))])
            for j in range(0,len(decoders)):
                for k in range(0,len(params)):
                 dictPar[decoders[j]][params[k]].update({surr[i]:mayhem[k+4*j]})   
            surrPar.update({surr[i]:dictPar})
        print("next surround")
    #fig.tight_layout()
    return surrPar


surrInt=(135,90,45,180,0,225,270,315)
surrParfr=surround_plotter(surr=surrInt,grType="gf")#Maximum activity normalized model. FIGURE 3
plt.subplots_adjust(left=0.06, bottom=0.09, right=0.8, top=0.88, wspace=0.14, hspace=0.15)
surrParSum=surround_plotter(surr=surrInt,grType="gs")#Total activity normalized model FIGURE 4
plt.subplots_adjust(left=0.06, bottom=0.09, right=0.8, top=0.88, wspace=0.14, hspace=0.15)
"""
Important parameters of no surround modulation: max/min ang shift and corresponding csds
for fr normalized:
These plots are not used further
"""
"""
fig=plotter.plot_template()
decoders=["vsgfns","vmgfns","mlgfns","mfgfns"]
params=["maanshi","csdmaanshi","mianshi","csdmianshi"]
colorMap=["red","blue","black","black"]
labelMap=["vector sum","von Mises fit","maximum likelihood","maximum fire rate"]
surrAngs=(135,90,45,180,0,225,270,315)
csdmamidiff=param_dict(decoders,sorted(["csdmamidiff"]*4)) #csd difference between maanshi and mianshi for fr normalized
mamidiff=param_dict(decoders,sorted(["mamidiff"]*4)) #difference between maanshi and mianshi for fr normalized
for i in range(0,len(surrParfr)):
   ax1=plotter.subplotter(fig,i)
   for j in range(0,len(surrParfr[surrAngs[i]])):
       ax1.plot(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["maanshi"].values(),'.',color=colorMap[j],label=labelMap[j]+" max")
       #print(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["maanshi"].values())
       ax1.plot(surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["mianshi"].values(),'x',color=colorMap[j],label=labelMap[j]+" min")
       #print(surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["mianshi"].values())
       ax1.set_xticks(np.linspace(-180,180,9))
       if i==2:
           ax1.legend(loc="best", bbox_to_anchor=(1,1))
       csdmamidiff[decoders[j]]["csdmamidiff"].update({surrAngs[i]:abs(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])})
       if abs(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])>180:
           csdmamidiff[decoders[j]]["csdmamidiff"].update({surrAngs[i]:360-abs(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])})
       mamidiff[decoders[j]]["mamidiff"].update({surrAngs[i]:abs(surrParfr[surrAngs[i]][decoders[j]]["maanshi"][surrAngs[i]]-surrParfr[surrAngs[i]][decoders[j]]["mianshi"][surrAngs[i]])})


plt.figure()
plt.title("Distance between maximum and minimum coding error without surround modulation for gradient center BW maximum fire rate normalized")
plt.xlabel("surround angle")
plt.ylabel("angular distance")
plt.plot(csdmamidiff["vsgfns"]["csdmamidiff"].keys(),csdmamidiff["vsgfns"]["csdmamidiff"].values(),".",color="red",label="vector sum")
plt.plot(csdmamidiff["vmgfns"]["csdmamidiff"].keys(),csdmamidiff["vmgfns"]["csdmamidiff"].values(),".",color="blue",label="von Mises fit")
plt.xticks(np.linspace(0,315,8))
plt.legend(loc="best", bbox_to_anchor=(1,1))

plt.figure()
plt.title("Difference of maximum and minimum coding error without surround modulation for gradient center BW maximum fire rate normalized")
plt.xlabel("surround angle")
plt.ylabel("Coding error difference")
plt.plot(mamidiff["vsgfns"]["mamidiff"].keys(),mamidiff["vsgfns"]["mamidiff"].values(),".",color="red",label="vector sum")
plt.plot(mamidiff["vmgfns"]["mamidiff"].keys(),mamidiff["vmgfns"]["mamidiff"].values(),".",color="blue",label="von Mises fit")
plt.xticks(np.linspace(0,315,8))
plt.legend(loc="best", bbox_to_anchor=(1,1))
"""
"""
Same as above for sum normalized
"""
"""
fig=plotter.plot_template()
decoders=["vsgsns","vmgsns","mlgsns","mfgsns"]
params=["maanshi","csdmaanshi","mianshi","csdmianshi"]
colorMap=["red","blue","black","green"]
labelMap=["vector sum","von Mises fit","maximum likelihood","maximum fire rate"]
surrAngs=(135,90,45,180,0,225,270,315)
csdmamidifs=param_dict(decoders,sorted(["csdmamidifs"]*4)) #csd difference between maanshi and mianshi for sum normalized
mamidifs=param_dict(decoders,sorted(["mamidifs"]*4)) #difference between maanshi and mianshi for sum normalized
for i in range(0,len(surrParfr)):
   ax1=plotter.subplotter(fig,i)
   for j in range(0,len(surrParfr[surrAngs[i]])):
       ax1.plot(surrParSum[surrAngs[i]][decoders[j]]["csdmaanshi"].values(),surrParSum[surrAngs[i]][decoders[j]]["maanshi"].values(),'.',color=colorMap[j],label=labelMap[j]+" max")
       #print(surrParfr[surrAngs[i]][decoders[j]]["csdmaanshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["maanshi"].values())
       ax1.plot(surrParSum[surrAngs[i]][decoders[j]]["csdmianshi"].values(),surrParSum[surrAngs[i]][decoders[j]]["mianshi"].values(),'x',color=colorMap[j],label=labelMap[j]+" min")
       #print(surrParfr[surrAngs[i]][decoders[j]]["csdmianshi"].values(),surrParfr[surrAngs[i]][decoders[j]]["mianshi"].values())
       ax1.set_xticks(np.linspace(-180,180,9))
       if i==2:
           ax1.legend(loc="best", bbox_to_anchor=(1,1))
       csdmamidifs[decoders[j]]["csdmamidifs"].update({surrAngs[i]:abs(surrParSum[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParSum[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])})
       if abs(surrParSum[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParSum[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])>180:
           csdmamidifs[decoders[j]]["csdmamidifs"].update({surrAngs[i]:360-abs(surrParSum[surrAngs[i]][decoders[j]]["csdmaanshi"][surrAngs[i]]-surrParSum[surrAngs[i]][decoders[j]]["csdmianshi"][surrAngs[i]])})
       mamidifs[decoders[j]]["mamidifs"].update({surrAngs[i]:abs(surrParSum[surrAngs[i]][decoders[j]]["maanshi"][surrAngs[i]]-surrParSum[surrAngs[i]][decoders[j]]["mianshi"][surrAngs[i]])})


plt.figure()
plt.title("Distance between maximum and minimum coding error without surround modulation for gradient center BW maximum total neuronal activity normalized")
plt.xlabel("surround angle")
plt.ylabel("angular distance")
plt.plot(csdmamidifs["vsgsns"]["csdmamidifs"].keys(),csdmamidifs["vsgsns"]["csdmamidifs"].values(),".",color="red",label="vector sum")
plt.plot(csdmamidifs["vmgsns"]["csdmamidifs"].keys(),csdmamidifs["vmgsns"]["csdmamidifs"].values(),".",color="blue",label="von Mises fit")
plt.plot(csdmamidifs["mfgsns"]["csdmamidifs"].keys(),csdmamidifs["mfgsns"]["csdmamidifs"].values(),".",color="green",label="maximum fire rate")
plt.xticks(np.linspace(0,315,8))
plt.legend(loc="best", bbox_to_anchor=(1,1))

plt.figure()
plt.title("Difference of maximum and minimum coding error without surround modulation for gradient center BW total neuronal activity normalized")
plt.xlabel("surround angle")
plt.ylabel("Coding error difference")
plt.plot(mamidifs["vsgsns"]["mamidifs"].keys(),mamidifs["vsgsns"]["mamidifs"].values(),".",color="red",label="vector sum")
plt.plot(mamidifs["vmgsns"]["mamidifs"].keys(),mamidifs["vmgsns"]["mamidifs"].values(),".",color="blue",label="von Mises fit")
plt.plot(mamidifs["mfgsns"]["mamidifs"].keys(),mamidifs["mfgsns"]["mamidifs"].values(),".",color="green",label="maximum fire rate")
plt.xticks(np.linspace(0,315,8))
plt.legend(loc="best", bbox_to_anchor=(1,1))
"""

#maximum and minimum coding error without surround modulation for gradient center BW total neuronal activity normalized

"""
rPatch=mp.Patch(color="red",label="von Mises fit")
bPatch=mp.Patch(color="blue",label="vector sum")
blPatch=mp.Patch(color="black",label="maximum likelihood&maximum fire rate")
fig1=plt.figure()
fig1.legend(handles=[rPatch,bPatch,blPatch])
plt.axis("off")
plt.title("coding error without surround modulation for gradient center BW maximum fire rate normalized",y=1.08)

for i in range(0,len(surrInt)):
    dictPar=surround_plotter(surrInt[i],analysisType="noSur",grType="gf")
    surrPar.update({surrInt[i]:dictPar})
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.close()
    print("next surround")
"""

"""
colModReg=colclass.colmod(std2kappa(60,1.4,1.5),1,0.5371428571428571,[60,70],bwType="regular")
colModGra=colclass.colmod(std2kappa(60,1.4,1.5),1,0.5371428571428571,[60,70],bwType="gradient/sum",avgSur=135)

vecSumGra1=colclass.decoder.vecsum(colModGra.x,colModGra.centery,colModGra.unitTracker,avgSur=135)#normalisieren auf max bringt biased activity for wide bandwidths!
vecSumGra2=colclass.decoder.vecsum(colModGra.x,colModGra.resulty,colModGra.unitTracker)
mLGra=colclass.decoder.ml(colModGra.x,colModGra.centery,colModGra.resulty,colModGra.unitTracker)
maxFrGra1=colclass.decoder.maxfr(colModGra.x,colModGra.centery,colModGra.unitTracker)
maxFrGra2=colclass.decoder.maxfr(colModGra.x,colModGra.resulty,colModGra.unitTracker)
vmFitGra1=colclass.decoder.vmfit(colModGra.x,colModGra.centery,colModGra.unitTracker,avgSur=135)
vmFitGra2=colclass.decoder.vmfit(colModGra.x,colModGra.resulty,colModGra.unitTracker)

plt.plot(vecSumGra1.angShift)#max deviance of 0.6, mean is but 0
plt.figure()
plt.plot(maxFrGra1.angShift)#all ok
plt.figure()
plt.plot(vmFitGra1.angShift)#problem here, max deviance 2, mean is but 0

plt.figure()
plt.plot(vecSumGra2.centSurDif,vecSumGra2.angShift,mLGra.centSurDif,mLGra.angShift,maxFrGra2.centSurDif,maxFrGra2.angShift,vmFitGra2.centSurDif,vmFitGra2.angShift)

for i in range(0,len(colModGra.centery),10):
    plt.plot(colModGra.centery[i])
"""

"""
Wrapper function of the above, where inputs are decoder type, surround etc is and output is plot and different variables of interest
"""

#Here mL and vmFit dont match each other for bigger centSurDif

#Plots get worse when units normalized by area, there is huge maximum deviance. Also surround modulation in maxFr funny! 








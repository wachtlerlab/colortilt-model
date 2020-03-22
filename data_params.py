# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:52:03 2019

@author: Ibrahim Alperen Tunc
"""
"""
Analysis of the models and parameters filtered in the data_analysis.py file.
"""
#Null model

import pickle
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import colclass as col
import sys
sys.path.insert(0,col.pathes.runpath)#!Change the directory accordingly
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
path=col.pathes.figpath
from datetime import date
from scipy.stats import chi2 as chisq

dictTot=pickle.load(open("dicttot.pckl","rb"))#Dictionary of the subject average.
date=date.today()

def file_opener(dataName,rmsThres):
    scpath=col.pathes.scanpath
    paraml=pickle.load(open(scpath+"\\%s.pckl"%(dataName),"rb"))#Pickle file of the scanned parameters. This name should be changed according to the file wished to be analyzed.
    meanrms=[]
    #rmsThres=4.5#the RMS threshold for filtering the scan parameters
    for i in range(0,len(paraml)):
        meanrms.append(np.array(list(paraml[i]["dif"].values())).mean())#The mean root mean square error fo the filtered models
    fltrms=[meanrms[np.where(np.array(meanrms)<rmsThres)[0][i]] for i in range(0,len(np.where(np.array(meanrms)<rmsThres)[0]))]#Take out the parameters which have a RMS >4
    fltind=[np.where(np.array(meanrms)<rmsThres)[0][i] for i in range(0,len(np.where(np.array(meanrms)<rmsThres)[0]))]#index list of the fltrms
    outind=[np.where(np.array(meanrms)>=rmsThres)[0][i] for i in range(0,len(np.where(np.array(meanrms)>=rmsThres)[0]))]#index list of the parameters not included in fltrms
    fltrms, fltind=zip(*sorted(zip(fltrms,fltind)))#sort fltind and fltrms values in the same ascending order 
    return paraml, meanrms, fltrms, fltind, outind


"""
Histogram of the parameter distribution. FIGURE 6
"""
popVecParams,popVecRms,popvecflt,popvecind,popvecout=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr",4.5)
mlParams,mlRms,mlflt,mlind,mlout=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-08-23",4)
popVecParamsmf,popVecRmsmf,popvecfltmf,popvecindmf,popvecoutmf=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr_maxnorm",4.5)

mlParamsskm,mlRmsskm,mlfltskm,mlindskm,mloutskm=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-22_surkap_modulated_ksurphase_112.5",4)#surround kappa modulated scan
mlParamsskm2,mlRmsskm2,mlfltskm2,mlindskm2,mloutskm2=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-20_surkap_modulated",4)#surround kappa modulated scan

mlParamsskm3,mlRmsskm3,mlfltskm3,mlindskm3,mloutskm3=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=22.5_kcentphase=112.5_2020-03-17",5)#not worked out well
mlParamsskm4,mlRmsskm4,mlfltskm4,mlindskm4,mloutskm4=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well

mlParamsskm5,mlRmsskm5,mlfltskm5,mlindskm5,mloutskm5=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well

plt.figure()
plt.hist(mlRms,bins=100,color="black")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
plt.title("RMS histogram of models with different parameters maximum likelhood decoder",fontsize=20)
plt.xlabel("Root mean square error",fontsize=15)
plt.ylabel("Number of models",fontsize=15)
plt.xticks(np.arange(3.7,7.6,0.2),)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.figure()
plt.hist(popVecRms,bins=100,color="black")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
plt.title("RMS histogram of models with different parameters population vector decoder",fontsize=20)
plt.xlabel("Root mean square error",fontsize=15)
plt.ylabel("Number of models",fontsize=15)
plt.xticks(np.arange(3.7,7.6,0.2),)
plt.tick_params(axis='both', which='major', labelsize=15)

def param_calculator(paraml,fltind,outind,rmslist,rmsThres,dataPlot=False,deco="ml",paraml2=None,fltind2=None,datDict=None,speplot=False,unimod=False,bwType="gradient/sum",bwType2="gradient/sum"):
    #paraml2 for deco=="both", that 2nd colmod model is created for the other decoder (1st ml, 2nd vecsum)
    #speplot: use to specify the model you want to plot (based on the index entry)
    #unimod: use to specify if plotted model is uniform or non-uniform (for uniform model the dictionary has to be slightly different)
    #bwType is for 1st model, bwType2 only when bothdec 
    """
    Calculate average center BW, min-max center BW difference, average sur mod depth, min-max sur mod depth by using following parameters:
    Parameters: surround modulation width, average center BW, min-max center BW difference, average mod depth sur, min-max mod depth sur (phase all equal in this case)
    """
    
    """
    Due to colormapping, some parts will be unnecessary, like filtering data (out ind) etc.
    
    scheme:
    fig=plt.figure()
    ax=plt.subplot(4,1,1)
    ax.scatter(mlflt,mlind,c=np.linspace(0,1,35),cmap="inferno")
    plot = ax.pcolor([mlflt,mlind]); fig.colorbar(plot)
    ax.set_xlim([3,5,4])
    """
    if unimod==True:
        kci=[]
        ksi=[]
        depval=[]
    else:
        smw=[]#surround modulation width
        difc=[]#center min-max BW difference
        difsm=[]#surround mod depth min-max difference
        cabw=[]#center average BW
        smad=[]#surround average mod depth
            
        
    if deco=="ml":
        decName="maximum likelihood"
    elif deco=="vecsum":
        decName="population vector"
    elif deco=="both":
        decName=None
    else:
        raise Exception("False decoder name given")
    """
    calculation for fltrms/fltind parameters
    """
    for i in range(0,len(paraml)):
        if unimod==True:
            kci.append(paraml[i]["kci"])
            ksi.append(paraml[i]["ksi"])
            depval.append(paraml[i]["depval"])
        else:
            smw.append(paraml[i]["ksi"])#ksi is the surround kappa values in the dictionary.
            difc.append(paraml[i]["ku"]-paraml[i]["kb"])#the center min-max difference
            difsm.append(paraml[i]["depu"]-paraml[i]["depb"])#surround modulation min-max difference
            cabw.append(difc[i]/2+paraml[i]["kb"])#take the min max difference, divide by 2 and add minimal value to get the average bandwidth.
            smad.append(difsm[i]/2+paraml[i]["depb"])#take the min max difference, divide by 2 and add below value the average mod depth.
    
    """
    calculation for outind parameters
    """
    if unimod==True:
        kcif=[]
        ksif=[]
        depvalf=[]
    else:
        smwf=[]#surround mod width list for filtered out models
        difcf=[]#center min-max BW difference list for filtered out models
        difsmf=[]#surround mod depth min-max difference list for filtered out models
        cabwf=[]#center average BW list for filtered out models
        smadf=[]#surround average mod depth list for filtered out models
    for i in range(0,len(outind)):#same loop as above for all model parameters with rms>4
        if unimod==True:
            kcif.append(paraml[outind[i]]["kci"])
            ksif.append(paraml[outind[i]]["ksi"])
            depvalf.append(paraml[outind[i]]["depval"])
        else:
            smwf.append(paraml[outind[i]]["ksi"])
            difcf.append(paraml[outind[i]]["ku"]-paraml[outind[i]]["kb"])
            difsmf.append(paraml[outind[i]]["depu"]-paraml[outind[i]]["depb"])
            cabwf.append(difcf[i]/2+paraml[outind[i]]["kb"])#take the min max difference, divide by 2 and add below value
            smadf.append(difsmf[i]/2+paraml[outind[i]]["depb"])#take the min max difference, divide by 2 and add below value
    
    if unimod==True:
        rmslist,kci,ksi,depval=zip(*sorted(zip(rmslist,kci,ksi,depval),reverse=True))
    else:
        rmslist,smw,difc,difsm,cabw,smad=zip(*sorted(zip(rmslist,smw,difc,difsm,cabw,smad),reverse=True))
    
    
    """
    Plot the parameter triplets in 3D, make also 2D variant of the same (same number of plots) FIGURE 7
    Leave it out for the time being
    """
    """
    Plot formatting
    """
    if unimod==True:
        colMap=["gray","red","blue"]#color map
    else:
        plotdict={"smw":smw,"cBWd":difc,"smd":difsm,"caBW":cabw,"sma":smad}#to assign the parameters to an axes while plotting
        plotdictf={"smwf":smwf,"difcf":difcf,"difsmf":difsmf,"cabwf":cabwf,"smadf":smadf}#to assign the parameters to an axes while plotting
        params=["smw","cBWd","smd","caBW","sma"]#to name the parameters in the plot title
        paramsf=["smwf","difcf","difsmf","cabwf","smadf"]#to name the parameters in the plot title
        colMap=["gray","red","blue"]#color map
        labmap=["rms>%s"%(rmsThres),"rms<%s"%(rmsThres),"best 10 models"]#label map
        fig = plt.figure()
        plt.title("Parameter distribution of models %s decoder"%(decName),fontsize=18,y=1.08)
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
        """
        Due to colormapping, some parts will be unnecessary, like filtering data (out ind) etc.
        
        scheme:
        fig=plt.figure()
        ax=plt.subplot(4,1,1)
        ax.scatter(mlflt,mlind,c=np.linspace(0,1,35),cmap="inferno")
        plot = ax.pcolor([mlflt,mlind]); fig.colorbar(plot)
        ax.set_xlim([3,5,4])
        """    
        i=0
        fig2 = plt.figure()
        plt.title("Parameter distribution of models %s decoder"%(decName),y=1.08,fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        for j in range(0,4):
            x=plotdict[params[j]]
            for k in range(1,5):
                if j>=k:
                    continue
                y=plotdict[params[k]]
                i=i+1
                ax=fig2.add_subplot(3,4,i)
                a=ax.scatter(x,y,c=np.array(rmslist),cmap='jet')
                ax.set_title('x=%s , y=%s'%(params[j],params[k]),fontdict={'fontsize': 15, 'fontweight': 'medium'})
                ax.tick_params(axis='both', which='major', labelsize=15)
        colbar=plt.colorbar(a,extend="max")
        a.set_clim(np.round(min(rmslist)-0.05,1),np.round(min(rmslist)+0.6,1))
        colbar.ax.tick_params(labelsize=15)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.89, wspace=0.2, hspace=0.39)

    """
    Look at the filtered model fits with waitforbuttonpress 
    NOTE: this code should be stopped after a while, as it will go through each and every model in the uploaded file.
    TO DO: Axis lines (x=0,y=0) and model estimates without psychophysics data DONE
    """
    symDict={}
    q=0#To track the number of the model fit.
    labmap=["data","model"]#label map
    surrInt=(135,90,45,180,0,225,270,315)#surround subplot order should be like this (see help(plotter.subplotter))
    if speplot==False:
        ran=range(0,len(fltind))
    else:
        ran=[eval(input("Enter the model index you want to plot in terms of fltind"))]#GIVE 0 for first entry of list
        q=ran[0]
        while q>=len(fltind):
            print("The given index exceeds the number of filtered models")
            ran=[eval(input("Enter the model index you want to plot"))]
        print("Following model is plotted: \n%s"%(paraml[fltind[ran[0]]]))
        
    for i in ran:
        q=q+1#in each iteration the model fit number goes up by 1.
        fig= plt.figure()
        plt.title("Model fit for induced hue shift, model #%s"%(q),fontsize=20)#Figure formatting the usual way from here up until for loop.
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        ax=plt.gca()
        ax.xaxis.set_label_coords(0.5, -0.07)
        ax.yaxis.set_label_coords(-0.05,0.5)
        plt.xlabel("Center-surround hue difference [°]",fontsize=20)
        plt.ylabel("Induced hue shift [°]",fontsize=20)
        symDict.update({q:{}})
        for j in range(0,len(dictTot)):#Create the model and decoder objects for each surround by using the parameter sets. Dicttot has each surround condition as element inside.
            #colMod=col.colmod(Kcent,Ksur,maxInh,stdInt,bwType="gradient/sum",phase=phase,avgSur=surrAvg[i],depInt=depInt,depmod=True,stdtransform=False)
            if unimod==True:
                print("uniform model")
                colMod=col.colmod(paraml[fltind[i]]["kci"],paraml[fltind[i]]["ksi"],paraml[fltind[i]]["depval"],bwType="regular",\
                                  avgSur=surrInt[j])
            else:
                colMod=col.colmod(1,paraml[fltind[i]]["ksi"],1,stdInt=[paraml[fltind[i]]["ku"],paraml[fltind[i]]["kb"]],bwType=bwType,\
                                  phase=paraml[fltind[i]]["phase"],avgSur=surrInt[j],depInt=[paraml[fltind[i]]["depb"],paraml[fltind[i]]["depu"]],depmod=True,stdtransform=False)#The model for ml so gradient/sum     
                print("model phase is %s"%(paraml[fltind[i]]["phase"]))
            if deco=="vecsum":
                print("population vector decoder")
                dec=col.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=surrInt[j],errNorm=True,centery=colMod.centery)#the decoder
            elif deco=="ml":
                print("maximum likelihood decoder")
                dec=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=surrInt[j],tabStep=1)#the decoder
            elif deco=="both":
                print("both decoders at once")
                if unimod==True:
                    print("uniform model")
                    colMod2=col.colmod(paraml2[fltind2[i]]["kci"],paraml2[fltind2[i]]["ksi"],paraml2[fltind2[i]]["depval"],bwType="regular",\
                                      avgSur=surrInt[j])
                else: 
                    colMod2=col.colmod(1,paraml2[fltind2[i]]["ksi"],1,stdInt=[paraml2[fltind2[i]]["ku"],paraml2[fltind2[i]]["kb"]],bwType=bwType2,\
                                  phase=paraml2[fltind2[i]]["phase"],avgSur=surrInt[j],depInt=[paraml2[fltind2[i]]["depb"],paraml2[fltind2[i]]["depu"]],depmod=True,stdtransform=False)#The model      
                    print("phases of models are %s (ml) and %s (vecsum)"%(paraml[fltind[i]]["phase"],paraml2[fltind2[i]]["phase"]))

                dec1=col.decoder.vecsum(colMod2.x,colMod2.resulty,colMod2.unitTracker,avgSur=surrInt[j],errNorm=True,centery=colMod2.centery)#the decoder vecsum
                dec2=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=surrInt[j],tabStep=1)#the decoder ml
                
            else:
                pass

            ax=plotter.subplotter(fig,j)#subplot the color tilt for each surround
            if dataPlot==True:
                if datDict==None:
                    ax.errorbar(dictTot[surrInt[j]]["angshi"].keys(),dictTot[surrInt[j]]["angshi"].values(),dictTot[surrInt[j]]["se"].values(),fmt='.',capsize=3,label=labmap[0],ecolor="gray",color="black")
                else:
                    ax.errorbar(datDict[surrInt[j]]["angshi"].keys(),datDict[surrInt[j]]["angshi"].values(),datDict[surrInt[j]]["se"].values(),fmt='.',capsize=3,label=labmap[0],ecolor="gray",color="black")
            #data plot with errorbars (above line)
            if deco=="both":
                ax.plot(dec1.centSurDif,dec1.angShift,color="green",label="population vector")#model plot
                ax.plot(dec2.centSurDif,dec2.angShift,color="magenta",label="maximum likelihood")#model plot
                ax.plot([0,0],[-25,25],color="black",linewidth=0.8)
                ax.plot([-185,185],[0,0],color="black",linewidth=0.8)
                ax.set_ylim(bottom=-25,top=25)#y axis limit +-25
                ax.set_xlim([-185,185])
                ax.set_xticks(np.linspace(-180,180,9))#x ticks between +-180 and ticks are at cardinal and oblique angles.
                ax.set_yticks(np.linspace(-20,20,5))#x ticks between +-180 and ticks are at cardinal and oblique angles.
                ax.tick_params(axis='both', which='major', labelsize=15)#major ticks are increazed in label size
                ax.xaxis.set_major_locator(MultipleLocator(90))#major ticks at cardinal angles.
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.xaxis.set_minor_locator(MultipleLocator(45))#minor ticks at obliques.
                print("1 subplot done")
                
            else:
                symDict[q].update({surrInt[j]:dec})
                ax.plot(dec.centSurDif,dec.angShift,color=colMap[1],label=labmap[1])#model plot
                ax.plot([0,0],[-25,25],color="black",linewidth=0.8)
                ax.plot([-185,185],[0,0],color="black",linewidth=0.8)
                ax.set_ylim(bottom=-25,top=25)#y axis limit +-25
                ax.set_xlim([-185,185])
                ax.set_xticks(np.linspace(-180,180,9))#x ticks between +-180 and ticks are at cardinal and oblique angles.
                ax.set_yticks(np.linspace(-20,20,5))#x ticks between +-180 and ticks are at cardinal and oblique angles.
                ax.tick_params(axis='both', which='major', labelsize=15)#major ticks are increazed in label size
                ax.xaxis.set_major_locator(MultipleLocator(90))#major ticks at cardinal angles.
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.xaxis.set_minor_locator(MultipleLocator(45))#minor ticks at obliques.
                print("1 subplot done")
        ax.legend(loc="best", bbox_to_anchor=(1,1),fontsize=20)#add the legend to the subplot in the very end, after all surrounds are plotted.
        if deco!="both":
            plt.subplots_adjust(left=0.07, bottom=0.09, right=0.86, top=0.95, wspace=0.14, hspace=0.1)
        else:
            plt.subplots_adjust(left=0.07, bottom=0.11, right=0.73, top=0.95, wspace=0.12, hspace=0.13)
            
        while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
            if plt.waitforbuttonpress(0):
                break
        plt.close()
        a=input("continue?   ")#if written yes to input, iteration goes on, if not, it stops :)
        numa=1
        while numa==1:
            if a=="yes":
                numa=0
                pass
            elif a=="no":
                numa=0
                return symDict
            else:
                numa=1
                a=input("continue?   ")            
    if deco!="both":
        return  symDict
    else:
        return

bothdec=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParams,fltind2=popvecind)
bothdec2=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParamsmf,fltind2=popvecindmf,bwType2="gradient/max")

mldec=param_calculator(mlParams,mlind,mlout,mlRms,4,deco="ml",dataPlot=True)
pvsum=param_calculator(popVecParams,popvecind,popvecout,popVecRms,4.5,deco="vecsum",dataPlot=True)
pvmf=param_calculator(popVecParamsmf,popvecindmf,popvecoutmf,popVecRmsmf,4.5,deco="vecsum",dataPlot=True,bwType="gradient/max")



#Same trick as before, try to open the pckl file, if not create one
try: 
    with open('dictionaries_pvtotarea_2020-02-11.json',"rb") as file:
        datDicts=json.load(file)
except FileNotFoundError:    
    print("forming the dictionaries")
    mlSymDict=param_calculator(mlParams,mlind,mlout,4)
    popvecSymDict=param_calculator(popVecParams,popvecind,popvecout,4.5,deco="vecsum")
    jsondict={"ml":{},"vecsum":{}}
    for i in list(mlSymDict[1].keys()):
        csdml=mlSymDict[1][i].centSurDif
        angsml=np.concatenate([*mlSymDict[1][i].angShift]).tolist()#to merge all values to a single array
        csdvs=[float(i) for i in popvecSymDict[1][i].centSurDif] #one liner to convert integers to float (json creates problems)
        angsvs=popvecSymDict[1][i].angShift
        jsondict["ml"].update({i:{}})
        jsondict["vecsum"].update({i:{}})
        jsondict["ml"][i].update({"csd":csdml})
        jsondict["ml"][i].update({"angs":angsml})
        jsondict["vecsum"][i].update({"csd":csdvs})
        jsondict["vecsum"][i].update({"angs":angsvs})
    print("creating the json file")
    with open('dictionaries_pvtotarea_%s.json'%(date), 'w') as f:
        json.dump(jsondict, f)
    print("json file is created")
except EOFError:    
    mlSymDict=param_calculator(mlParams,mlind,mlout,4)
    popvecSymDict=param_calculator(popVecParams,popvecind,popvecout,4.5,deco="vecsum")    
    jsondict={"ml":{},"vecsum":{}}
    for i in list(mlSymDict[1].keys()):
        csdml=mlSymDict[1][i].centSurDif
        angsml=np.concatenate([*mlSymDict[1][i].angShift]).tolist()#to merge all values to a single array
        csdvs=[float(i) for i in popvecSymDict[1][i].centSurDif] #one liner to convert integers to float (json creates problems)
        angsvs=popvecSymDict[1][i].angShift
        jsondict["ml"].update({i:{}})
        jsondict["vecsum"].update({i:{}})
        jsondict["ml"][i].update({"csd":csdml})
        jsondict["ml"][i].update({"angs":angsml})
        jsondict["vecsum"][i].update({"csd":csdvs})
        jsondict["vecsum"][i].update({"angs":angsvs})
    with open('dictionariess_%s.json'%(date), 'w') as f:
        json.dump(jsondict, f)
    print("json file is filled")

"""
load json file:
with open('dictionariess_2019-11-17.json',"rb") as file:
data=json.load(file)
"""

"""
Best model different decoders RMS plots relative to different surrounds
"""
mlbestRMS=mlParams[mlind[0]]["dif"]
pvbestRMS=popVecParams[popvecind[0]]["dif"]
plt.figure()
ax=plt.subplot(111)
ax.bar(np.array(list(mlbestRMS.keys()))-2.5,list(mlbestRMS.values()), width=5, color="magenta", align="center",label="Maximum likelihood")
ax.bar(np.array(list(mlbestRMS.keys()))+2.5,list(pvbestRMS.values()), width=5, color="green", align="center",label="Population vector")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("RMS between data and model",fontsize=30)
plt.legend(loc="best",fontsize=20)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.99, wspace=0, hspace=0)
plt.savefig(path+"\\decoder_models_comparison.pdf")
#mlSymDict=param_calculator(mlParams,mlind,mlout,4)
#popvecSymDict=param_calculator(popVecParams,popvecind,popvecout,4.5,deco="vecsum")


"""
Check if the both decoders plot is now correct
fig=plotter.plot_template(auto=True)
angles=(135,90,45,180,0,225,270,315)
for i in range(0,8):
    ax=plotter.subplotter(fig,i)
    print(i)
    while True:#This line ensures the next color tilt curve is plotted when a keyboard button is pressed.
            if plt.waitforbuttonpress(0):
                break
    ax.plot(datDicts["ml"]["%s"%(angles[i])]["csd"],datDicts["ml"]["%s"%(angles[i])]["angs"],color="magenta")
    ax.plot(datDicts["vecsum"]["%s"%(angles[i])]["csd"],datDicts["vecsum"]["%s"%(angles[i])]["angs"],color="green")
    ax.errorbar(dictTot[angles[i]]["angshi"].keys(),dictTot[angles[i]]["angshi"].values(),dictTot[angles[i]]["se"].values(),fmt=".",capsize=3,ecolor="gray",color="black")
    ax.set_xticks(np.linspace(-180,180,9))#x ticks between +-180 and ticks are at cardinal and oblique angles.
    ax.set_yticks(np.linspace(-20,20,5))#x ticks between +-180 and ticks are at cardinal and oblique angles.
"""

"""
Look at best fitting parameter margins to make the interval smaller
depb10=[]
depu10=[]
kb10=[]
ku10=[]
ksi10=[]
depb10v=[]
depu10v=[]
kb10v=[]
ku10v=[]
ksi10v=[]
for i in range(0,len(mlind)):
    depb10.append(mlParams[mlind[i]]["depb"])
    depu10.append(mlParams[mlind[i]]["depu"])
    kb10.append(mlParams[mlind[i]]["kb"])
    ku10.append(mlParams[mlind[i]]["ku"])
    ksi10.append(mlParams[mlind[i]]["ksi"])
for i in range(0,len(popvecind)):    
    depb10v.append(popVecParams[popvecind[i]]["depb"])
    depu10v.append(popVecParams[popvecind[i]]["depu"])
    kb10v.append(popVecParams[popvecind[i]]["kb"])
    ku10v.append(popVecParams[popvecind[i]]["ku"])
    ksi10v.append(popVecParams[popvecind[i]]["ksi"])

print(min(depb10),max(depu10),min(kb10),max(ku10),min(ksi10),max(ksi10),"ml")
print(min(depb10v),max(depu10v),min(kb10v),max(ku10v),min(ksi10v),max(ksi10v),"pv")
"""

def symmetry_analyzer(dicti):
    symVal={"ml":{},"vecsum":{}}
    angles=(135,90,45,180,0,225,270,315)
    if dicti==dictTot:
        name="Empirical data"#The data has to be reshaped, that RMS is calculated for
                             #each individual plot, then the average is taken for the total
        symVal={"rms":{},"std":{}}#avgse is averaged standart error
        
        fig=plotter.plot_template(auto=True)
        plt.title("Symmetry analysis of %s"%(name),y=1.08,fontsize=30)
        plt.xlabel("Absolute center surround angle difference [°]",fontsize=30)
        plt.ylabel("Absolute hue shift [°]",fontsize=30)
        for i in range(0,len(dicti)):
            csdneg=np.flip(abs(np.array(list(dicti[angles[i]]["angshi"].keys())))[0:7],0)
            angsneg=np.flip(abs(np.array(list(dicti[angles[i]]["angshi"].values())))[0:7],0)
            negse=np.flip(abs(np.array(list(dicti[angles[i]]["se"].values())))[0:7],0)#standard error values at negative side., dont use 
                                                                              #it for the time being, ask Thomas first how to get along 
                                                                              #with it
            
            csdpos=np.array(list(dicti[angles[i]]["angshi"].keys()))[8:-1]
            angspos=np.array(list(dicti[angles[i]]["angshi"].values()))[8:-1]
            posse=np.array(list(dicti[angles[i]]["se"].values()))[8:-1]
            
            rms=np.sqrt(((angsneg-angspos)**2).mean())
            sd=np.std(abs(angsneg-angspos))#the standart deviation of the angle difference between halves (all positive)
            symVal["rms"].update({angles[i]:rms})
            symVal["std"].update({angles[i]:sd})
            
            ax=plotter.subplotter(fig,i)
            ax.errorbar(csdneg,angsneg,negse,fmt='x',capsize=3,markersize=5,label="negative",ecolor="blue",color="blue")
            ax.errorbar(csdpos,angspos,posse,fmt='.',capsize=3,markersize=5,label="positive",ecolor="red",color="red")

            #ax.plot(csdneg,angsneg,"x",color="blue",label="negative",markersize=10)
            #ax.plot(csdpos,angspos,".",color="red",label="positive",markersize=10)
            ax.set_xticks(np.linspace(0,180,9))#x ticks are between 0-180, in obliques and cardinal angles
            ax.tick_params(axis='both', which='major', labelsize=20)#major ticks are bigger labeled
            ax.xaxis.set_major_locator(MultipleLocator(45))#major ticks are set at 0,90,180,...
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(22.5))#minor ticks are set at 45,135,..
            ax.set_ylim(bottom=-3,top=25)#y axis is between +-30
            if i==7:            
                ax.legend(loc="best",bbox_to_anchor=(1,1),fontsize=20)
                mng = plt.get_current_fig_manager()
                mng.window.state("zoomed")
                plt.pause(0.1)
        return symVal
    
    else:
        symVal={"ml":{},"vecsum":{}}
        for i in ("ml","vecsum"):
            if i=="ml":
                name="maximum likelihood decoder"
            else:
                name="population vector decoder"
            fig=plotter.plot_template(auto=True)
            plt.title("Symmetry analysis of %s"%(name),y=1.08,fontsize=30)
            plt.xlabel("Absolute center surround angle difference [°]",fontsize=30)
            plt.ylabel("Absolute hue shift [°]",fontsize=30)
            
            for j in (135,90,45,180,0,225,270,315):
                if j<=180 and i=="vecsum" or j<180:
                    csdneg=np.flip(abs(np.array(dicti[i]["%s"%(j)]["csd"][0:dicti[i]["%s"%(j)]["csd"].index(0)])),0)#center surround differences are transformed into absolute values.
                    angsneg=np.flip(abs(np.array(dicti[i]["%s"%(j)]["angs"][0:dicti[i]["%s"%(j)]["csd"].index(0)])),0)#angular shifts are transformed into absolute values.
                    csdpos=np.array(dicti[i]["%s"%(j)]["csd"][dicti[i]["%s"%(j)]["csd"].index(0)+1:-1])
                    angspos=np.array(dicti[i]["%s"%(j)]["angs"][dicti[i]["%s"%(j)]["csd"].index(0)+1:-1])   
                else:
                    csdneg=np.flip(abs(np.array(dicti[i]["%s"%(j)]["csd"][1:dicti[i]["%s"%(j)]["csd"].index(0)])),0)#center surround differences are transformed into absolute values.
                    angsneg=np.flip(abs(np.array(dicti[i]["%s"%(j)]["angs"][1:dicti[i]["%s"%(j)]["csd"].index(0)])),0)#angular shifts are transformed into absolute values.
                    csdpos=np.array(dicti[i]["%s"%(j)]["csd"][dicti[i]["%s"%(j)]["csd"].index(0)+1:])
                    angspos=np.array(dicti[i]["%s"%(j)]["angs"][dicti[i]["%s"%(j)]["csd"].index(0)+1:])   
                rms=np.sqrt(((angsneg-angspos)**2).mean())
                symVal[i].update({j:rms})
                ax=plotter.subplotter(fig,angles.index(j))
                ax.plot(csdneg,angsneg,"x",color="blue",label="negative",markersize=5)
                ax.plot(csdpos,angspos,".",color="red",label="positive",markersize=5)
                ax.set_xticks(np.linspace(0,180,9))#x ticks are between 0-180, in obliques and cardinal angles
                ax.tick_params(axis='both', which='major', labelsize=20)#major ticks are bigger labeled
                ax.xaxis.set_major_locator(MultipleLocator(45))#major ticks are set at 0,90,180,...
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.xaxis.set_minor_locator(MultipleLocator(22.5))#minor ticks are set at 45,135,..
                ax.set_ylim(bottom=0,top=20)#y axis is between +-20
                if angles.index(j)==7:            
                    ax.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
                    mng = plt.get_current_fig_manager()
                    mng.window.state("zoomed")
                    plt.pause(0.1)
        return symVal
          
    """            
    else:
        try: 
            str(dicti).index("vecsum")
            name="population vector decoder"
        except ValueError:
            name="maximum likelihood decoder"
    for i in range(1,len(dicti)+1):
        symVal.update({i:{}})
        fig=plotter.plot_template(auto=True)
        plt.title("Symmetry analysis of %s"%(name),y=1.08,fontsize=30)
        plt.xlabel("Absolute center surround angle difference [°]",fontsize=20)
        plt.ylabel("Absolute hue shift [°]",fontsize=20)

        for j in (135,90,45,180,0,225,270,315):
            if j>=180 and name=="maximum likelihood decoder" or j>180:
                csdneg=np.flip(abs(np.array(dicti[i][j].centSurDif[1:dicti[i][j].centSurDif.index(0)])),0)#center surround differences are transformed into absolute values.
                angsneg=np.flip(abs(np.array(dicti[i][j].angShift[1:dicti[i][j].centSurDif.index(0)])),0)#angular shifts are transformed into absolute values.
                csdpos=np.array(dicti[i][j].centSurDif[dicti[i][j].centSurDif.index(0)+1:])
                angspos=np.array(dicti[i][j].angShift[dicti[i][j].centSurDif.index(0)+1:])
            else:
                csdneg=np.flip(abs(np.array(dicti[i][j].centSurDif[0:dicti[i][j].centSurDif.index(0)])),0)#center surround differences are transformed into absolute values.
                angsneg=np.flip(abs(np.array(dicti[i][j].angShift[0:dicti[i][j].centSurDif.index(0)])),0)#angular shifts are transformed into absolute values.
                csdpos=np.array(dicti[i][j].centSurDif[dicti[i][j].centSurDif.index(0)+1:-1])
                angspos=np.array(dicti[i][j].angShift[dicti[i][j].centSurDif.index(0)+1:-1])
            rms=np.sqrt(((angsneg-angspos)**2).mean())
            symVal[i].update({j:rms})
            ax=plotter.subplotter(fig,angles.index(j))
            ax.plot(csdneg,angsneg,"x",color="blue",label="negative",markersize=10)
            ax.plot(csdpos,angspos,".",color="red",label="positive",markersize=10)
            ax.set_xticks(np.linspace(0,180,9))#x ticks are between 0-180, in obliques and cardinal angles
            ax.tick_params(axis='both', which='major', labelsize=15)#major ticks are bigger labeled
            ax.xaxis.set_major_locator(MultipleLocator(45))#major ticks are set at 0,90,180,...
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(22.5))#minor ticks are set at 45,135,..
            ax.set_ylim(bottom=0,top=20)#y axis is between +-20
            if angles.index(j)==7:            
                ax.legend(loc="best", bbox_to_anchor=(1,1),fontsize=15)
                mng = plt.get_current_fig_manager()
                mng.window.state("zoomed")
                plt.pause(0.1)
                plt.savefig(path+"\\symmetry_analysis_%s.pdf"%(name))
    return symVal
    """
datsym=symmetry_analyzer(dictTot)#data symmetry (no change necessary)
decsym=symmetry_analyzer(datDicts)
mlsym=decsym["ml"]
popvecsym=decsym["vecsum"]
plt.figure()
plt.title("Symmetry comparsion between decoders and data",y=1.08,fontsize=30)
plt.xlabel("Absolute center surround angle difference [°]",fontsize=30)
plt.ylabel("Average symmetry difference",fontsize=30)
#plt.ylim(0.1,2.5)
plt.plot(mlsym.keys(),mlsym.values(),".",color="magenta",label="maximum likelihood",markersize=15)
plt.plot(popvecsym.keys(),popvecsym.values(),".",color="green",label="population vector decoder",markersize=15)
plt.errorbar(datsym["rms"].keys(),datsym["rms"].values(),datsym["std"].values(),fmt='.',capsize=3,markersize=15,label="empirical data",ecolor="black",color="black")
plt.xticks(np.linspace(0,315,8))
plt.tick_params(axis='both', which='major', labelsize=20)#major ticks are bigger labeled
plt.legend(loc="best",fontsize=20)
plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.88, wspace=0, hspace=0)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.savefig(path+"\\symmetry_decoder_comparison.pdf")

"""
Analysis of single individuals
"""
"""
Individual data dictionaries
"""
"""
Import the data
!These directories should be changed accordingly to the directory where data csv file is. 
"""
coldatTW=pd.read_csv(r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\data\tiltdataTW.csv",header=None)
coldatSU=pd.read_csv(r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\data\tiltdataSU.csv",header=None)
coldatMH=pd.read_csv(r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\data\tiltdataMH.csv",header=None)
coldatLH=pd.read_csv(r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\data\tiltdataLH.csv",header=None)
coldatHN=pd.read_csv(r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\data\tiltdataHN.csv",header=None)


"""
Preallocate dictionaries for each surround.
"""
params=["angshi","se"]#csd=center-surround difference, angshi= induced hue shift, se=standard error
dictTW=param_dict(np.linspace(0,315,8),params)
dictSU=param_dict(np.linspace(0,315,8),params)
dictMH=param_dict(np.linspace(0,315,8),params)
dictLH=param_dict(np.linspace(0,315,8),params)
dictHN=param_dict(np.linspace(0,315,8),params)

for i in range(0,360,45):
    for j in range(1,17):
        dictTW[i]["angshi"].update({coldatTW[0][j]:coldatTW[1][j+i/45*17]})
        dictTW[i]["se"].update({coldatTW[0][j]:coldatTW[2][j+i/45*17]})
        dictSU[i]["angshi"].update({coldatSU[0][j]:coldatSU[1][j+i/45*17]})
        dictSU[i]["se"].update({coldatSU[0][j]:coldatSU[2][j+i/45*17]})
        dictMH[i]["angshi"].update({coldatMH[0][j]:coldatMH[1][j+i/45*17]})
        dictMH[i]["se"].update({coldatMH[0][j]:coldatMH[2][j+i/45*17]})
        dictLH[i]["angshi"].update({coldatLH[0][j]:coldatLH[1][j+i/45*17]})
        dictLH[i]["se"].update({coldatLH[0][j]:coldatLH[2][j+i/45*17]})
        dictHN[i]["angshi"].update({coldatHN[0][j]:coldatHN[1][j+i/45*17]})
        dictHN[i]["se"].update({coldatHN[0][j]:coldatHN[2][j+i/45*17]})


"""
Load scanned decoder dictionaries and form datas:
"""
popVecParamsHN,popVecRmsHN,popvecfltHN,popvecindHN,popvecoutHN=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-02_dictHN",4.5)
mlParamsHN,mlRmsHN,mlfltHN,mlindHN,mloutHN=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-11-26_dictHN",4)
mlbestRMSHN=mlParamsHN[mlindHN[0]]["dif"]
pvbestRMSHN=popVecParamsHN[popvecindHN[0]]["dif"]


popVecParamsLH,popVecRmsLH,popvecfltLH,popvecindLH,popvecoutLH=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-02_dictLH",4.5)
mlParamsLH,mlRmsLH,mlfltLH,mlindLH,mloutLH=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-11-27_dictLH",4)
mlbestRMSLH=mlParamsLH[mlindLH[0]]["dif"]
pvbestRMSLH=popVecParamsLH[popvecindLH[0]]["dif"]

popVecParamsMH,popVecRmsMH,popvecfltMH,popvecindMH,popvecoutMH=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-02_dictMH",4.5)
mlParamsMH,mlRmsMH,mlfltMH,mlindMH,mloutMH=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-11-29_dictMH",4)
mlbestRMSMH=mlParamsMH[mlindMH[0]]["dif"]
pvbestRMSMH=popVecParamsMH[popvecindMH[0]]["dif"]

popVecParamsTW,popVecRmsTW,popvecfltTW,popvecindTW,popvecoutTW=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-02_dictTW",5)
mlParamsTW,mlRmsTW,mlfltTW,mlindTW,mloutTW=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-12-02_dictTW",4.5)
mlbestRMSTW=mlParamsTW[mlindTW[0]]["dif"]
pvbestRMSTW=popVecParamsTW[popvecindTW[0]]["dif"]

popVecParamsSU,popVecRmsSU,popvecfltSU,popvecindSU,popvecoutSU=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-02_dictSU",4.5)
mlParamsSU,mlRmsSU,mlfltSU,mlindSU,mloutSU=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-12-01_dictSU",4)
mlbestRMSSU=mlParamsSU[mlindSU[0]]["dif"]
pvbestRMSSU=popVecParamsSU[popvecindSU[0]]["dif"]

popVecParamsAbs,popVecRmsAbs,popvecfltAbs,popvecindAbs,popvecoutAbs=file_opener("paraml_fit_15_decoder_vecsum_errType_rms_2019-12-06_nose",4.5)
mlParamsAbs,mlRmsAbs,mlfltAbs,mlindAbs,mloutAbs=file_opener("paraml_fit_15_decoder_ml_errType_rms_2019-12-06_nose",4)
mlbestRMSAbs=mlParamsAbs[mlindAbs[0]]["dif"]
pvbestRMSAbs=popVecParamsAbs[popvecindAbs[0]]["dif"]


"""
Create the RMS comparison plots
"""
names=["HN","LH","MH","TW","SU"]
mlvals=[mlbestRMSHN,mlbestRMSLH,mlbestRMSMH,mlbestRMSTW,mlbestRMSSU]
pvvals=[pvbestRMSHN,pvbestRMSLH,pvbestRMSMH,pvbestRMSTW,pvbestRMSSU]
plt.figure()
for i in range(0,5):
    ax=plt.subplot(1,5,i+1)
    plt.title(names[i],fontsize=30)
    ax.bar(np.array(list(mlbestRMS.keys()))-2.5,list(mlvals[i].values()), width=5, color="magenta", align="center",label="ml")
    ax.bar(np.array(list(mlbestRMS.keys()))+2.5,list(pvvals[i].values()), width=5, color="green", align="center",label="pv")
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.set_xticks(np.linspace(0,315,8))
    ax.set_yticks(np.linspace(0,7,8))
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])
    #ax.set_xlabel("Surround hue angle [°]",fontsize=30)
    #ax.set_ylabel("RMS between data and model",fontsize=30)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.89, top=0.94, wspace=0, hspace=0)
plt.legend(loc="best",fontsize=20,bbox_to_anchor=[1,1])
mng.window.state("zoomed")
plt.pause(0.1)
plt.savefig(path+"\\rms_individual.pdf")
"""
RMS histograms for individuals
"""
mlRMSs=[mlRmsHN,mlRmsLH,mlRmsMH,mlRmsTW,mlRmsSU,mlRmsAbs]
pvRMSs=[popVecRmsHN,popVecRmsLH,popVecRmsMH,popVecRmsTW,popVecRmsSU,popVecRmsAbs]
for i in range(0,6):
    plt.figure()
    plt.suptitle("RMS of %s"%(names[i]),fontsize=25)
    ax=plt.subplot(1,2,1)
    ax.set_title("ml decoder",fontsize=25)
    plt.hist(mlRMSs[i],bins=100,color="magenta")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
    plt.xlabel("Root mean square error",fontsize=25)
    plt.ylabel("Number of models",fontsize=25)
    plt.xticks(np.linspace(2.5,8.5,7))
    plt.yticks(np.linspace(0,140,15))
    plt.tick_params(axis='both', which='major', labelsize=15)
    ax=plt.subplot(1,2,2)
    plt.hist(pvRMSs[i],bins=100,color="green")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
    plt.xticks(np.linspace(2.5,8.5,7))
    plt.yticks(np.linspace(0,140,15))
    ax.set_title("pv decoder",fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=15)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.pause(0.1)
    while True:
        if plt.waitforbuttonpress(0):
            break
    plt.savefig(path+"\\individual_hist_%s.pdf"%(names[i]))
    plt.close()
 
    
"""
Best decoder plots for individuals and non SE normalized scan
IF YOU RUN THESE SIMULATIONS AGAIN; CONSIDER CHANGING DATDICT FOR EACH INDIVIDUAL TO THE CORRESPONDING
DATA DICTIONARY (dictTW etc.). THIS WAS MANUALLY DONE FOR THE FIRST RUN.
"""   
bothdecHN=param_calculator(mlParamsHN,mlindHN,mloutHN,4.5,deco="both",dataPlot=True,paraml2=popVecParamsHN,fltind2=popvecindHN)
bothdecLH=param_calculator(mlParamsLH,mlindLH,mloutLH,4.5,deco="both",dataPlot=True,paraml2=popVecParamsLH,fltind2=popvecindLH)
bothdecMH=param_calculator(mlParamsMH,mlindMH,mloutMH,4.5,deco="both",dataPlot=True,paraml2=popVecParamsMH,fltind2=popvecindMH)
bothdecTW=param_calculator(mlParamsTW,mlindTW,mloutTW,4.5,deco="both",dataPlot=True,paraml2=popVecParamsTW,fltind2=popvecindTW)
bothdecSU=param_calculator(mlParamsSU,mlindSU,mloutSU,4.5,deco="both",dataPlot=True,paraml2=popVecParamsSU,fltind2=popvecindSU)
bothdecAbs=param_calculator(mlParamsAbs,mlindAbs,mloutAbs,4.5,deco="both",dataPlot=True,paraml2=popVecParamsAbs,fltind2=popvecindAbs)


"""
Compare se normalized with non se normalized
"""
plt.figure()
ax=plt.subplot(121)
ax.bar(np.array(list(mlbestRMS.keys()))-2.5,list(mlbestRMS.values()), width=5, color="magenta", align="center",label="Maximum likelihood")
ax.bar(np.array(list(mlbestRMS.keys()))+2.5,list(pvbestRMS.values()), width=5, color="green", align="center",label="Population vector")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("RMS between data and model",fontsize=30)
ax=plt.subplot(122)
ax.bar(np.array(list(mlbestRMS.keys()))-2.5,list(mlbestRMSAbs.values()), width=5, color="magenta", align="center",label="Maximum likelihood")
ax.bar(np.array(list(mlbestRMS.keys()))+2.5,list(pvbestRMSAbs.values()), width=5, color="green", align="center",label="Population vector")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("Absolute model error",fontsize=30)
plt.legend(loc="best",fontsize=20)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")

"""
Scan with 30 degrees 45 degrees phase and a specific phase interval 
"""
popVecParams30,popVecRms30,popvecflt30,popvecind30,popvecout30=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-13_[30]°",4.5)
mlParams30,mlRms30,mlflt30,mlind30,mlout30=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-12-13_[30]°",4)

popVecParams45,popVecRms45,popvecflt45,popvecind45,popvecout45=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2019-12-15_[45]°",5)
mlParams45,mlRms45,mlflt45,mlind45,mlout45=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-12-15_[45]°",5)


plt.figure()
plt.hist(mlRms30,bins=100,color="black")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
plt.title("RMS histogram of models with different parameters maximum likelhood decoder",fontsize=20)
plt.xlabel("Root mean square error",fontsize=15)
plt.ylabel("Number of models",fontsize=15)
plt.xticks(np.arange(3.7,7.6,0.2),)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.figure()
plt.hist(popVecRms30,bins=100,color="black")#min mean val is 3.74, taking mean rms=4 as threshold to look at parameter properties    
plt.title("RMS histogram of models with different parameters population vector decoder",fontsize=20)
plt.xlabel("Root mean square error",fontsize=15)
plt.ylabel("Number of models",fontsize=15)
plt.xticks(np.arange(3.7,7.6,0.2),)
plt.tick_params(axis='both', which='major', labelsize=15)

bothdec30=param_calculator(mlParams30,mlind30,mlout30,4.5,deco="both",dataPlot=True,paraml2=popVecParams30,fltind2=popvecind30)

mlbestRMS30=mlParams30[mlind30[0]]["dif"]
pvbestRMS30=popVecParams30[popvecind30[0]]["dif"]
plt.figure()
ax=plt.subplot(111)
ax.bar(np.array(list(mlbestRMS30.keys()))-2.5,list(mlbestRMS30.values()), width=5, color="magenta", align="center",label="Maximum likelihood")
ax.bar(np.array(list(mlbestRMS30.keys()))+2.5,list(pvbestRMS30.values()), width=5, color="green", align="center",label="Population vector")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("RMS between data and model",fontsize=30)
plt.legend(loc="best",fontsize=20)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.99, wspace=0, hspace=0)


bothdec45=param_calculator(mlParams45,mlind45,mlout45,5,deco="both",dataPlot=True,paraml2=popVecParams45,fltind2=popvecind45)
mlbestRMS45=mlParams45[mlind45[0]]["dif"]
pvbestRMS45=popVecParams45[popvecind45[0]]["dif"]
plt.figure()
ax=plt.subplot(111)
ax.bar(np.array(list(mlbestRMS45.keys()))-2.5,list(mlbestRMS45.values()), width=5, color="magenta", align="center",label="Maximum likelihood")
ax.bar(np.array(list(mlbestRMS45.keys()))+2.5,list(pvbestRMS45.values()), width=5, color="green", align="center",label="Population vector")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("RMS between data and model",fontsize=30)
plt.legend(loc="best",fontsize=20)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.99, wspace=0, hspace=0)


popVecParamsph,popVecRmsph,popvecfltph,popvecindph,popvecoutph=file_opener("paraml_fit_7_decoder_vecsum_errType_rms_2020-01-06_[  0.   22.5  45.   67.5  90.  112.5 135.  157.5]°",4.5)
mlParamsph,mlRmsph,mlfltph,mlindph,mloutph=file_opener("paraml_fit_7_decoder_ml_errType_rms_2019-12-24_[11.25  16.875 22.5   28.125 33.75 ]°",4)

#Analyze this further.
pvph=[]
mlph=[]
for i in range(0,20):
  pvph.append(popVecParamsph[popvecindph[i]]["phase"])
  mlph.append(mlParamsph[mlindph[i]]["phase"])  

"""
Look at the best fit 16.875° model for ml graphically
"""
mlphspec=param_calculator(mlParamsph,mlindph,mloutph,4,deco="ml",dataPlot=True,speplot=True)
#write to input mlph.index(16.875)

"""
Popvec parameter scan no error correction:
"""
popVecParamsnc,popVecRmsnc,popvecfltnc,popvecindnc,popvecoutnc=file_opener("paraml_fit_15_decoder_vecsum_errType_rms_2020-01-09_nose_nocorr",4.5)
#Fit better, as error is in this case smaller duh!

"""
Uniform scan, total uniform or only non uniform surround drive
"""
popVecParamsuni,popVecRmsuni,popvecfltuni,popvecinduni,popvecoutuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-02-14_nocorr_uni",4.5)
mlParamsuni,mlRmsuni,mlfltuni,mlinduni,mloutuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-14_uni",4.5)

popVecParamscuni,popVecRmscuni,popvecfltcuni,popvecindcuni,popvecoutcuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-02-14_nocorr_unicent",4.5)
mlParamscuni,mlRmscuni,mlfltcuni,mlindcuni,mloutcuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-15_unicent",4.5)

mlbestRMS=mlParams[mlind[0]]["dif"]
mlbestRMSskm=mlParamsskm2[mlindskm2[0]]["dif"]

pvbestRMS=popVecParams[popvecind[0]]["dif"]

mlbestRMSuni=mlParamsuni[mlinduni[0]]["dif"]
pvbestRMSuni=popVecParamsuni[popvecinduni[0]]["dif"]

mlbestRMScuni=mlParamscuni[mlindcuni[0]]["dif"]
pvbestRMScuni=popVecParamscuni[popvecindcuni[0]]["dif"]


"""
Uniform model plots
"""
bothdecuni=param_calculator(mlParamsuni,mlinduni,mloutuni,mlRmsuni,4.5,deco="both",dataPlot=True,paraml2=popVecParamsuni,fltind2=popvecinduni,unimod=True)#both uniform models

bothdecCuni=param_calculator(mlParamscuni,mlindcuni,mloutcuni,mlRmscuni,4.5,deco="both",dataPlot=True,paraml2=popVecParamscuni,fltind2=popvecindcuni)#both uniform models


plt.figure()
ax=plt.subplot(121)
ax.set_title("Maximum Likelihood",fontsize=30)
ax.bar(np.array(list(mlbestRMS.keys()))-12,list(mlbestRMS.values()), width=8, color="magenta", align="center",hatch="*",label="Non-uniform avg 3.74")
ax.bar(np.array(list(mlbestRMS.keys())),list(mlbestRMScuni.values()), width=8, color="magenta", align="center",hatch="+",label="Center Uniform avg 4.054")
ax.bar(np.array(list(mlbestRMS.keys()))+12,list(mlbestRMSuni.values()), width=8, color="magenta", align="center",hatch="o",label="Uniform avg 4.374")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_yticks(np.linspace(0,7,8))
ax.set_ylim([0,8])
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
ax.set_ylabel("RMS error",fontsize=30)
plt.legend(loc="best",fontsize=18)

ax=plt.subplot(122)
ax.set_title("Population vector",fontsize=30)
ax.bar(np.array(list(pvbestRMS.keys()))-12,list(pvbestRMS.values()), width=8, color="green", align="center",hatch="*",label="Non-uniform avg 4.196")
ax.bar(np.array(list(pvbestRMS.keys())),list(pvbestRMScuni.values()), width=8, color="green", align="center",hatch="+",label="Center Uniform avg 4.157")
ax.bar(np.array(list(pvbestRMS.keys()))+12,list(pvbestRMSuni.values()), width=8, color="green", align="center",hatch="o",label="Uniform avg 4.382")
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xticks(np.linspace(0,315,8))
ax.set_ylim([0,8])
ax.set_yticks([])
ax.set_xlabel("Surround hue angle [°]",fontsize=30)
plt.legend(loc="best",fontsize=18)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.06, bottom=0.12, right=0.99, top=0.94, wspace=0, hspace=0)
plt.savefig(path+"\\rms_error_ml_pv_uni_nonuni_cuni.pdf")

"""
Kappa and depmod distribution for different surround conditions for best model
Do after looking at the error correction stuff, as this could lead to change in vecsum object etc.
"""
#Kappa
def kappa_vals(kdown,kup,phase):
    kapmod=(kdown-kup)/2*np.cos(2*np.deg2rad(np.linspace(0,359,360)-phase))+kup+(kdown-kup)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py
    return kapmod

#Suppression strength
def dep_vals(depInt,phase):
    depval=[]
    for i in np.linspace(0,359,360):
        depval.append(depth_modulator(depInt,i,phase))
    return depval

mlbest=mlParams[mlind[0]]
pvbest=popVecParams[popvecind[0]]
pvbestmf=popVecParamsmf[popvecindmf[0]]

mlkap=kappa_vals(mlbest["ku"],mlbest["kb"],mlbest["phase"])
pvkap=kappa_vals(pvbest["ku"],pvbest["kb"],pvbest["phase"])
pvkapmf=kappa_vals(pvbestmf["ku"],pvbestmf["kb"],pvbestmf["phase"])

mldep=dep_vals([mlbest["depb"],mlbest["depu"]],mlbest["phase"])
pvdep=dep_vals([pvbest["depb"],pvbest["depu"]],pvbest["phase"])
pvdepmf=dep_vals([pvbestmf["depb"],pvbestmf["depu"]],pvbestmf["phase"])

fig=plt.figure()
plt.title("Kappa distribution",fontsize=30,y=1.08)
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(1,2,1,projection='polar')
ax1.plot(np.deg2rad(np.linspace(0,359,360)),mlkap,".",color="magenta",label="Maximum likelihood")
ax1.set_ylim(0.8,1.3)
ax1.set_yticks(np.arange(0.8,1.3,0.1))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title("Maximum likelihood",fontsize=30,y=1.08)


ax1=fig.add_subplot(1,2,2,projection='polar')
ax1.plot(np.deg2rad(np.linspace(0,359,360)),pvkap,".",color="green",label="Population vector decoder")
ax1.set_ylim(1.8,2.1)
ax1.set_yticks(np.arange(1.8,2.1,0.1))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title("Population vector decoder",fontsize=30,y=1.08)

fig=plt.figure()
plt.title("Surround modulation strength",fontsize=30,y=1.08)
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(1,2,1,projection='polar')
ax1.plot(np.deg2rad(np.linspace(0,359,360)),mldep,".",color="magenta",label="Maximum likelihood")
ax1.set_ylim(0.1,0.5)
ax1.set_yticks(np.arange(0.1,0.5,0.1))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title("Maximum likelihood",fontsize=30,y=1.08)


ax1=fig.add_subplot(1,2,2,projection='polar')
ax1.plot(np.deg2rad(np.linspace(0,359,360)),pvdep,".",color="green",label="Population vector decoder")
ax1.set_ylim(0.3,0.7)
ax1.set_yticks(np.arange(0.4,0.7,0.1))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title("Population vector decoder",fontsize=30,y=1.08)


"""
Same as above but linear coordinate systems:
"""
fig=plt.figure()
plt.xticks([])
plt.yticks([])
plt.box(False)

ax1=fig.add_subplot(1,2,1)
ax1.plot(np.linspace(0,359,360),mlkap,color="magenta",label="Maximum likelihood",linewidth=3)
ax1.plot(np.linspace(0,359,360),pvkap,color="green",label="Population vector decoder",linewidth=3)
ax1.set_xticks(np.linspace(0,360,9))
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_ylim([0,2.1])
ax1.set_title("Kappa distribution",fontsize=30)

ax2=fig.add_subplot(1,2,2)
ax2.plot(np.linspace(0,359,360),mldep,color="magenta",label="Maximum likelihood",linewidth=3)
ax2.plot(np.linspace(0,359,360),pvdep,color="green",label="Population vector decoder",linewidth=3)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_xticks(np.linspace(0,360,9))
ax2.xaxis.set_major_locator(MultipleLocator(90))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_minor_locator(MultipleLocator(45))
ax2.set_yticks(np.linspace(0,0.6,7))
ax2.set_title("Surround modulation strength",fontsize=30)
ax1.legend(loc="best",prop={'size':20})
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.94, wspace=0.11, hspace=0.28)
plt.savefig(path+"\\dep_kap_plots.pdf")
#USE TOTAL AREA NORMALIZED VECSUM

"""
Depmod distribution plots for center-only uniform models
"""
mlbestcuni=mlParamscuni[mlindcuni[0]]
pvbestcuni=popVecParams[popvecindcuni[0]]

mldepcuni=dep_vals([mlbestcuni["depb"],mlbestcuni["depu"]],mlbestcuni["phase"])
pvdepcuni=dep_vals([pvbestcuni["depb"],pvbestcuni["depu"]],pvbestcuni["phase"])

fig=plt.figure()
plt.xticks([])
plt.yticks([])
ax1=fig.add_subplot(1,1,1)
ax1.plot(np.linspace(0,359,360),mldepcuni,color="magenta",label="Maximum likelihood",linewidth=3)
ax1.plot(np.linspace(0,359,360),pvdepcuni,color="green",label="Population vector decoder",linewidth=3)
ax1.set_xticks(np.linspace(0,360,9))
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_yticks(np.linspace(0,0.8,9))
ax1.set_title("Surround modulation strength",fontsize=30)
ax1.legend(loc="best",prop={'size':20})
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.pause(0.1)
plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.93, wspace=0.11, hspace=0.28)




"""
THE FIT INDICES APART FROM RMSEA
#TODO: Rearrange all the code (null model stuff etc.), mostly done
"""
"""
*REDIDUAL variance:
*Chi square of the model error: TODO: Revise, DONE, now it is correct formula that it uses standart deviation not standard error
*Chi square between models (nested): TODO: implement p.58 method from Chapter 4 Estimation and Testing of Computational
                                    Psychological Models, Jerome R. Busemeyer and Adele Diederich
                                    chi=chi(simple model)-chi(complex model), df is k(complex)-k(simple)
                                    Done but ask TW as p values are irrealistically low (10^-20 ish)
!QUESTION ON NULL MODEL: use the average hue shift per surround or the total average hue shift over all surrounds?
                         if average hue shift per surround is used, then is the number of parameters for the null model
                         8 (i.e. number of surround conditions)?
*AIC between unnested models (Null and Rest): AIC_model=chi_sq(model error)+2k (k:# of parameters) TODO: implement done,
                                                TODO: maybe convert to proportions relative to null model (ask TW)
*BIC             -''-                       : BIC_model=sqrt(chi_sq(model error))+ln(T)*k (T:# of observations) TODO: implement done with a plot twist (ask TW)
*BIC for nested models: (chi_sq(simple model error)-chi_sq(complex model error))-ln(T)*(k(complex)-k(simple)) TODO: implement done
*R square: R^2= 1-(SSE/TSS) , SSE is unweighted sum of square errors for all conditions ((dec-dat)**2)
and TSS is the sum of squared deviations around the mean of the observed data (averaged across conditions), corresponding to null model
"""

"""
Firstly redefine all models so it is simpler to run the code
"""
"""
popvec 
"""
popVecParams,popVecRms,popvecflt,popvecind,popvecout=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr",4.5)
popVecParamsmf,popVecRmsmf,popvecfltmf,popvecindmf,popvecoutmf=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr_maxnorm",4.5)
popVecParamsuni,popVecRmsuni,popvecfltuni,popvecinduni,popvecoutuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-02-14_nocorr_uni",4.5)
popVecParamscuni,popVecRmscuni,popvecfltcuni,popvecindcuni,popvecoutcuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-02-14_nocorr_unicent",4.5)

"""
ml
"""
mlParams,mlRms,mlflt,mlind,mlout=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-08-23",4)
mlParamsuni,mlRmsuni,mlfltuni,mlinduni,mloutuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-14_uni",4.5)
mlParamscuni,mlRmscuni,mlfltcuni,mlindcuni,mloutcuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-15_unicent",4.5)

"""
ml surkap modulated
"""
mlParamsskm,mlRmsskm,mlfltskm,mlindskm,mloutskm=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-22_surkap_modulated_ksurphase_112.5",4)#surround kappa modulated scan
mlParamsskm2,mlRmsskm2,mlfltskm2,mlindskm2,mloutskm2=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-20_surkap_modulated",4)#surround kappa modulated scan
mlParamsskm3,mlRmsskm3,mlfltskm3,mlindskm3,mloutskm3=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=22.5_kcentphase=112.5_2020-03-17",5)#not worked out well
mlParamsskm4,mlRmsskm4,mlfltskm4,mlindskm4,mloutskm4=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well
mlParamsskm5,mlRmsskm5,mlfltskm5,mlindskm5,mloutskm5=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well



"""
The null model, which predicts the average for each surround condition: Find the RMS per surround condition to estimate how well the models explain the variability.
Here, chi square will be done, where each data value is weighted by the se.
"""
avgvals=[]#average hue shift value per surround, spanning from 0 to 315 degrees. These values are null model estimation
for i in list(dictTot.keys()):
    meanval=np.mean(np.array(list(dictTot[i]["angshi"].values())))
    avgvals.append(meanval)

"""
!!!RMS is based on standard error! TO get back to variance, first compute standard deviation (SE*sqrt(n))
Here n=5 (5 participants, SE was calculated for each surround as sd(list(mean(observer)))/sqrt(# of each observer))
"""

#RMS calculation:
nullRMS=[]#RMS per each surround
for i in list(dictTot.keys()):
    angshi=np.array(list(dictTot[i]["angshi"].values()))
    sterr=np.array(list(dictTot[i]["se"].values()))
    RMS=np.sqrt(np.mean((((avgvals[list(dictTot.keys()).index(i)]-angshi)/sterr)**2))) ##Formula: sqrt(mean(((model-data)/SE_data)^2))
    nullRMS.append(RMS)

#!!!CAUTION: The average observer data is normalized by its standard error, but this new normalized data also requires its variance for the further computation.

def chi_sq_calc(dec,data):#dec is the best model angshift predictions, data is the data dictionary for a given surround
    return np.sum((dec-np.array(list(data["angshi"].values())))**2/(np.array(list(data["se"].values())))**2)
    """
    Nice function but unnecessary, do instead np.array(list(mlParams[mlind[0]]["dif"].values()))**2*16 for each model dictionary
    At least you know it now :)
    If function not working properly, inspect the codes below:
    mlbestmod=col.colmod(1,2.3,1,stdInt=[1.2,0.9],bwType="gradient/sum",phase=22.5,avgSur=0,depInt=[0.2,0.3999999999999999],depmod=True,stdtransform=False)
    mlbestdec=col.decoder.ml(mlbestmod.x,mlbestmod.centery,mlbestmod.resulty,mlbestmod.unitTracker,avgSur=0,dataFit=True)
    dat0=dictTot[0]
    chi=np.sum((mlbestdec.angShift-np.array(list(dat0["angshi"].values())))**2/(np.array(list(dat0["se"].values())))**2)
    a=np.mean((mlbestdec.angShift-np.array(list(dat0["angshi"].values())))**2/(np.array(list(dat0["se"].values())))**2)
    #chi and a the same up to 8 decimals!    
    """

"""
#The loop for calculating chi square for each surround:
for i in dictTot.keys():
    print(i)
    nullchi.append(chi_sq_calc(avgvals[int(i/45)],dictTot[i]))

    mod=col.colmod(1,2.3,1,stdInt=[1.2,0.9],bwType="gradient/sum",phase=22.5,avgSur=i,depInt=[0.2,0.4],depmod=True,stdtransform=False)
    dec=col.decoder.ml(mod.x,mod.centery,mod.resulty,mod.unitTracker,avgSur=i,dataFit=True)
    mlchi.append(chi_sq_calc(dec.angShift,dictTot[i]))
"""
nullchi=np.array(nullRMS)**2*16/5#chi square per each surround: sum(((dec-dat)/std(dat))**2) where dec and dat are hue shifts per surround (n=16)

"""
RMS**2/16 gives sum(((dec-dat)/SE)**2) and to come to std from SE (as in chi square=sum(((dec-dat)/std)**2)) SE has to be 
multiplied by sqrt(N) (number of participants, N=5), whereas all is squared and n is constant so it can be taken all the 
way out of the sum, i.e. RMS=sqrt(1/16*sum(((dec-dat)/s)**2*5)) 
(where 16 is number of avg. observer datapoints per surround and 5 is number of observers)
"""                                

"""
Model error chi square with descending complexity (df inbetween for each is 1)
"""
mlchiun=np.array(list(mlParamsuni[mlinduni[0]]["dif"].values()))**2*16/5
mlchicun=np.array(list(mlParamscuni[mlindcuni[0]]["dif"].values()))**2*16/5
mlchinun=np.array(list(mlParams[mlind[0]]["dif"].values()))**2*16/5

"""
# of variables for each model
"""
mlparamUn=len(mlParamsuni[0])-1
mlparamCun=len(mlParamscuni[0])-2#as kb=ku
mlparamNun=len(mlParams[0])-1

"""
chi square test between models
"""
chiuncun=np.sum(mlchiun-mlchicun)#chi test between uniform and center only uniform
dfuncun=mlparamCun-mlparamUn
puncun=1-chisq.cdf(chiuncun,dfuncun)#says zero, so significant

chicunnun=np.sum(mlchicun-mlchinun)#chi test between center only uniform and non-uniform  
dfcunnun=mlparamNun-mlparamCun
punnun=1-chisq.cdf(chicunnun,dfcunnun)#says zero, so significant
#chi=chi(simple model)-chi(complex model), df is k(complex)-k(simple)

"""
AIC of the models (to compare with null model as they are not nested)
"""
nullaic=np.sum(nullchi)+2
mlunaic=np.sum(mlchiun)+2*mlparamUn
mlcunaic=np.sum(mlchicun)+2*mlparamCun
mlnunaic=np.sum(mlchinun)+2*mlparamNun
#AIC_model=chi_sq(model error)+2k (k:# of parameters)


"""
BIC of the models (to compare with null model as they are not nested)
"""
obsnum=len(dictTot.keys())*len(dictTot[0]["angshi"].keys())#number of observations over all surrounds
nullbic=np.sqrt(np.sum(nullchi))+np.log(obsnum)*1
mlunbic=np.sqrt(np.sum(mlchiun))+np.log(obsnum)*mlparamUn
mlcunbic=np.sqrt(np.sum(mlchicun))+np.log(obsnum)*mlparamCun
mlnunbic=np.sqrt(np.sum(mlchinun))+np.log(obsnum)*mlparamNun
#Values not as expected and not fitting to AIC outcome
#BIC_model=sqrt(chi_sq(model error))+ln(T)*k (T:# of observations)

"""
BIC for nested models
"""
bicuncun=np.sum(mlchiun)-np.sum(mlchicun)-np.log(obsnum)*(mlparamCun-mlparamUn)
biccunnun=np.sum(mlchicun)-np.sum(mlchinun)-np.log(obsnum)*(mlparamNun-mlparamCun)
bicunnun=np.sum(mlchiun)-np.sum(mlchinun)-np.log(obsnum)*(mlparamNun-mlparamUn)
#values as expected
#nestedbic=(chi_sq(simple model error)-chi_sq(complex model error))-ln(T)*(k(complex)-k(simple))

"""
R square
"""
def sse_calc(dec,data):#dec is the best model angshift predictions, data is the data dictionary for a given surround
    return np.sum((dec-np.array(list(data["angshi"].values())))**2)

#*R^2= 1-(SSE/TSS) , SSE is unweighted sum of square errors for all conditions ((dec-dat)**2)
#and TSS is the sum of squared deviations around the mean of the observed data (averaged across conditions), corresponding to null model

"""
AIK values (david a kenny)
"""
df=len(dictTot)*len(dictTot[0]["se"])
nullaik=np.sum(nullchi)+1*4-2*df
mlnunAik=np.sum(mlchinun)+mlparamNun*(mlparamNun+1)-2*df
mlunAik=np.sum(mlchiun)+mlparamUn*(mlparamUn+1)-2*df
mlcunAik=np.sum(mlchicun)+mlparamUn*(mlparamCun+1)-2*df

"""
BIC (david a kenny)
chi+ln(N)*((k*(k+1))/2-df) k(k + 3)/2 if mean in model.
"""
nullbik=np.sum(nullchi)+np.log(obsnum)*((1*4)/2-df)
mlnunbik=np.sum(mlchinun)+np.log(obsnum)*((mlparamNun*(mlparamNun+1))/2-(df-mlparamNun))
mlunbik=np.sum(mlchinun)+np.log(obsnum)*((mlparamUn*(mlparamUn+1))/2-(df-mlparamUn))
mlcunbik=np.sum(mlchinun)+np.log(obsnum)*((mlparamCun*(mlparamCun+1))/2-(df-mlparamCun))



    #16 measurements per surround condition, so multiply mean with 16
    #RMS Formula: sqrt(mean(((model-data)/SE_data)^2)), change it to sum((expected-observed)^2/variance)
        
scanrms={"ml":mlbestRMS,"mluni":mlbestRMSuni,"mlcuni":mlbestRMScuni,"mlskm":mlbestRMSskm,"pv":pvbestRMS,"pvuni":pvbestRMSuni,"pvcuni":pvbestRMScuni}
rmsind={}
meanmodrms={}
for i in list(scanrms.keys()):
    rmsind.update({i:(nullRMS-np.array(list(scanrms[i].values())))/nullRMS})
    meanmodrms.update({i:np.mean((nullRMS-np.array(list(scanrms[i].values())))/nullRMS)})

"""
ML best params, for the surkappa modulated scan. This approach did not yield any good result, so total scan was conducted.
"""
depb10=[]
depu10=[]
kb10=[]
ku10=[]
ksi10=[]

for i in range(0,10):
    dicttt=mlParams[mlind[i]]
    depb10.append(dicttt["depb"])
    depu10.append(dicttt["depu"])
    kb10.append(dicttt["kb"])
    ku10.append(dicttt["ku"])
    ksi10.append(dicttt["ksi"])
print(min(depb10),max(depu10),min(kb10),max(ku10),min(ksi10),max(ksi10))

"""
INVESTIGATE THE EFFECT OF PHASE ON COLOR TILT CURVES
"""

"""
IT SEEMS LIKE BETWEEN SURROUNDS (+-45°) THE ANGULAR SHIFT NEGATIVE AND POSITIVE VALUES FLIP.
"""
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

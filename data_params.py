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
import plotly.express as px

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

def combine_scans(nuni,cuni,uni,nunirms,cunirms,unirms):
#Combine different model scans together, nuni is non uniform, cuni center only uniform and uni uniform model
#First three variables are scan dictionary lists, last 3 are mean rms values
    for i in range(0,len(uni)):
        try:
            dep=uni[i]["depval"]
            k=np.round(uni[i]["kci"],1)
            uni[i].update({"depb":dep,"depu":dep,"kb":k,"ku":k})
        except KeyError:
            pass
        try:
            del(uni[i]["depval"])    
            del(uni[i]["kci"])
        except KeyError:
            pass
    rmslist=nunirms+cunirms+unirms#all rms values coaggulated together in the order of nuni,cuni,uni
    dictilist=nuni+cuni+uni#all scan dictionaries concatanated in nuni cuni uni order
    indexlist=np.arange(0,len(dictilist))#index list of the dictionary list, used to order rms from small to big values
                                         #without losing the track of the scan parameter set in dictionary list
                                         #NOTE that here index list encompasses everything
    rmslist1,indexlist=zip(*sorted(zip(rmslist,indexlist)))
    return dictilist,rmslist,indexlist


"""
Histogram of the parameter distribution. FIGURE 6
"""
"""
popvec 
"""
popVecParams,popVecRms,popvecflt,popvecind,popvecout=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr",4.5)
popVecParamsmf,popVecRmsmf,popvecfltmf,popvecindmf,popvecoutmf=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr_maxnorm",4.5)
popVecParamsuni,popVecRmsuni,popvecfltuni,popvecinduni,popvecoutuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-04-17_nocorr_uni",4.5)
popVecParamscuni,popVecRmscuni,popvecfltcuni,popvecindcuni,popvecoutcuni=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-04-17_nocorr_unicent",4.5)
"""
ml
"""
mlParams,mlRms,mlflt,mlind,mlout=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-08-23",4)
mlParamsuni,mlRmsuni,mlfltuni,mlinduni,mloutuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-04-17_uni",4.5)
mlParamscuni,mlRmscuni,mlfltcuni,mlindcuni,mloutcuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-04-17_unicent",4.5)
mlParamsmf,mlRmsmf,mlfltmf,mlindmf,mloutmf=file_opener("paraml_fit_10_decoder_ml_errType_rms_phase=22.5_maxfr_2020-05-26",4)

"""
Totally combined lists
"""
popVecParamstot,popVecRmstot,popvecindtot=combine_scans(popVecParams,popVecParamscuni,popVecParamsuni,popVecRms,popVecRmscuni,popVecRmsuni)
mlParamstot,mlRmstot,mlindtot=combine_scans(mlParams,mlParamscuni,mlParamsuni,mlRms,mlRmscuni,mlRmsuni)

"""
Same combination by using the maxact normalized scans (note that normalization is only matter of nonunif model as other models have same center bw!)
"""
popVecParamsmftot,popVecRmsmftot,popvecindmftot=combine_scans(popVecParamsmf,popVecParamscuni,popVecParamsuni,popVecRmsmf,popVecRmscuni,popVecRmsuni)
mlParamsmftot,mlRmsmftot,mlindmftot=combine_scans(mlParamsmf,mlParamscuni,mlParamsuni,mlRmsmf,mlRmscuni,mlRmsuni)


#mlParamsskm,mlRmsskm,mlfltskm,mlindskm,mloutskm=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-22_surkap_modulated_ksurphase_112.5",4)#surround kappa modulated scan
#mlParamsskm2,mlRmsskm2,mlfltskm2,mlindskm2,mloutskm2=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-20_surkap_modulated",4)#surround kappa modulated scan
#mlParamsskm3,mlRmsskm3,mlfltskm3,mlindskm3,mloutskm3=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=22.5_kcentphase=112.5_2020-03-17",5)#not worked out well
#mlParamsskm4,mlRmsskm4,mlfltskm4,mlindskm4,mloutskm4=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well
#mlParamsskm5,mlRmsskm5,mlfltskm5,mlindskm5,mloutskm5=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well

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

def param_calculator(paraml,fltind,outind,rmslist,rmsThres,dataPlot=False,deco="ml",paraml2=None,fltind2=None,datDict=None,speplot=False,unimod=False,bwType="gradient/sum",bwType2="gradient/sum",label=None,RMSnorm=False):
    #paraml2 for deco=="both", that 2nd colmod model is created for the other decoder (1st ml, 2nd vecsum)
    #speplot: use to specify the model you want to plot (based on the index entry)
    #unimod: use to specify if plotted model is uniform or non-uniform (for uniform model the dictionary has to be slightly different)
    #bwType is for 1st model, bwType2 only when bothdec 
    #label is for putting labels on the color coded parameter space plot
    #for the parameter plot the number of subplots can be reduced to a considerable amount.
    #to see which one is distributed to a wider area.
    #RMSnorm: bool, optional. If true, the RMS values are normalized per decoder so that the minimum value is 1. This is used to
    #compare relative distribution of the parameters for the decoders. You can also use it to only get the parameter distribution 
    #plots, since this variable has nothing to do with the RMS variable given to the function
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
        LAST PLOT HAS A MISTAKE ON COLOR CODE!!! SOMEHOW ONLY THE LAST ONE, something wrong in loop? nope all good now!
        scheme:
        fig=plt.figure()
        ax=plt.subplot(4,1,1)
        ax.scatter(mlflt,mlind,c=np.linspace(0,1,35),cmap="inferno")
        plot = ax.pcolor([mlflt,mlind]); fig.colorbar(plot)
        ax.set_xlim([3,5,4])
        """    
        i=0
        fig2 = plt.figure()
        if label!=None:
            fig2.text(0.05,0.95,label,fontsize=20)
        else:
            pass
        
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
                a=ax.scatter(x,y,c=np.array(rmslist),cmap='coolwarm',edgecolors="none",s=10)
                if RMSnorm==False:
                    a.set_clim(4.5-0.8,4.5+0.8)                    
                else:
                    a.set_clim(min(rmslist),min(rmslist)+1.6)#color coding to each of the minimum RMS value.
                #ax.set_facecolor("whitesmoke")
                ax.set_title('x=%s , y=%s'%(params[j],params[k]),fontdict={'fontsize': 15, 'fontweight': 'medium'})
                ax.tick_params(axis='both', which='major', labelsize=15)
        colbar=plt.colorbar(a,extend="max")
        colbar.ax.tick_params(labelsize=15)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.89, wspace=0.2, hspace=0.39)
        if RMSnorm == True:
            return
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
                dec=col.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=surrInt[j],centery=colMod.centery)#the decoder
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

                dec1=col.decoder.vecsum(colMod2.x,colMod2.resulty,colMod2.unitTracker,avgSur=surrInt[j],centery=colMod2.centery)#the decoder vecsum
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

"""
THESE ARE KIND OF OBSOLETE NOW; USE THE TOTAL COMBNED LIST
bothdec=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParams,fltind2=popvecind)
bothdec2=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParamsmf,fltind2=popvecindmf,bwType2="gradient/max")

bothdecbest=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParamscuni,fltind2=popvecindcuni)

#Best models
mldec=param_calculator(mlParams,mlind,mlout,mlRms,4,deco="ml",dataPlot=True,label="A")
pvcuni=param_calculator(popVecParamscuni,popvecindcuni,popvecoutcuni,popVecRmscuni,4.5,deco="vecsum",dataPlot=True,bwType="gradient/sum",label="B",nonunif=False)
"""
mltot=param_calculator(mlParamstot,mlindtot,mlout,mlRmstot,4,deco="ml",dataPlot=True,label="A")#mlout now unnecessary parameter which has no effect so leave it be
pvtot=param_calculator(popVecParamstot,popvecindtot,popvecoutcuni,popVecRmstot,4.5,deco="vecsum",dataPlot=True,bwType="gradient/sum",label="B")
#!!!PLOTS LOOK UGLY; REDO THE UNI AND CUNI SCANS WITH EXACTLY THE SAME PARAMETER SCAN SPACE done

#Same with maxfr normalization in nonuniform model
mltotmf=param_calculator(mlParamsmftot,mlindmftot,mlout,mlRmsmftot,4,deco="ml",dataPlot=True,label="A")#mlout now unnecessary parameter which has no effect so leave it be
pvtotmf=param_calculator(popVecParamsmftot,popvecindmftot,popvecoutmf,popVecRmsmftot,4.5,deco="vecsum",dataPlot=True,bwType="gradient/max",label="B")

#best model fits considering also maxfr vs totarea normalizations:
bothdecbest=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParamsmf,fltind2=popvecindmf,bwType2="gradient/max")
bothdecbestttar=param_calculator(mlParams,mlind,mlout,mlRms,4.5,deco="both",dataPlot=True,paraml2=popVecParamstot,fltind2=popvecindtot)


#Other considerations
pvsum=param_calculator(popVecParams,popvecind,popvecout,popVecRms,4.5,deco="vecsum",dataPlot=True)
pvmf=param_calculator(popVecParamsmf,popvecindmf,popvecoutmf,popVecRmsmf,4.5,deco="vecsum",dataPlot=True,bwType="gradient/max")

#Generate the parameter scan with all color coding corresponding to the minimal RMS value of each decoder.
mltotn=param_calculator(mlParamstot,mlindtot,mlout,mlRmstot,4,deco="ml",dataPlot=True,label="A",RMSnorm=True)#mlout now unnecessary parameter which has no effect so leave it be
pvtotmfn=param_calculator(popVecParamsmftot,popvecindmftot,popvecoutmf,popVecRmsmftot,4.5,deco="vecsum",dataPlot=True,bwType="gradient/max",label="B",RMSnorm=True)


#Same trick as before, try to open the pckl file, if not create one
#TODO: redo this with popvec cuni
try: 
    with open('dictionaries_pvtotarea_2020-06-30.json',"rb") as file:
        datDicts=json.load(file)
except FileNotFoundError:    
    print("forming the dictionaries")
    mlSymDict=param_calculator(mlParamstot,mlindtot,mlout,mlRmstot,4,deco="ml",dataPlot=True,label="A")
    popvecSymDict=param_calculator(popVecParamsmf,popvecindmf,popvecoutmf,popVecRmsmf,4.5,deco="vecsum",dataPlot=True,bwType="gradient/max",label="B")
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
mlbestRMS=mlParamstot[mlindtot[0]]["dif"]
pvbestRMS=popVecParamsmftot[popvecindmftot[0]]["dif"]
plt.figure()
ax=plt.subplot(111)
ax.bar(np.array(list(mlbestRMS.keys()))-3,list(mlbestRMS.values()), width=5, edgecolor="magenta",color="None", align="center",label="Maximum likelihood",linewidth=3)
ax.bar(np.array(list(mlbestRMS.keys()))+3,list(pvbestRMS.values()), width=5, edgecolor="green",color="None", align="center",label="Population vector",linewidth=3)
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

mlbest=mlParamstot[mlindtot[0]]
pvbest=popVecParamstot[popvecindtot[0]]
pvbestmf=popVecParamsmf[popvecindmf[0]]

mlkap=kappa_vals(mlbest["ku"],mlbest["kb"],mlbest["phase"])
pvkap=kappa_vals(pvbest["ku"],pvbest["kb"],pvbest["phase"])
pvkapmf=kappa_vals(pvbestmf["ku"],pvbestmf["kb"],pvbestmf["phase"])

mldep=dep_vals([mlbest["depb"],mlbest["depu"]],mlbest["phase"])
pvdep=dep_vals([pvbest["depb"],pvbest["depu"]],pvbest["phase"])
pvdepmf=dep_vals([pvbestmf["depb"],pvbestmf["depu"]],pvbestmf["phase"])

mlbestRMS=mlbest["dif"]
pvbestRMS=pvbest["dif"]
pvmfbestRMS=pvbestmf["dif"]

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
ax1.plot(np.linspace(0,359,360),pvkapmf,color="green",label="Population vector decoder",linewidth=3)
ax1.set_xticks(np.linspace(0,360,9))
ax1.xaxis.set_major_locator(MultipleLocator(90))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(45))
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_ylim([0,2.1])
ax1.set_title("Kappa distribution",fontsize=30)

ax2=fig.add_subplot(1,2,2)
ax2.plot(np.linspace(0,359,360),mldep,color="magenta",label="Maximum likelihood",linewidth=3)
ax2.plot(np.linspace(0,359,360),pvdepmf,color="green",label="Population vector decoder",linewidth=3)
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


#TODO: MAKE A SINGLE PLOT OUTTA THOSE 5 PLOTS ABOVE
#Take 2: put the polar plots inside the linear ones (https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib)
#Conclusion: Ignore polar plot, linear is enough
fig=plt.figure()
ax1=plt.subplot(2,1,1)#RMS PLOT
ax2=plt.subplot(2,2,3)#lin kappa
ax3=plt.subplot(2,2,4)#lin surr

#Plot fine tunings
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.98, wspace=0.17, hspace=0.38)
fig.text(0.08,0.92,"A",fontsize=20)
fig.text(0.055,0.43,"B",fontsize=20)
fig.text(0.57,0.43,"C",fontsize=20)

#RMS PLOT
ax1.bar(np.array(list(mlbestRMS.keys()))-3,list(mlbestRMS.values()), width=5, edgecolor="magenta",color="None", align="center",label="Maximum likelihood",linewidth=3)
ax1.bar(np.array(list(mlbestRMS.keys()))+3,list(pvmfbestRMS.values()), width=5, edgecolor="green",color="None", align="center",label="Population vector",linewidth=3)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_xticks(np.linspace(0,315,8))
ax1.set_xlabel("Surround hue angle [°]",fontsize=20)
ax1.set_ylabel("RMS",fontsize=20)
ax1.legend(bbox_to_anchor=(0.07,0.6),fontsize=15)
ax1.set_ylim([0,7.5])
ax1.set_yticks(np.arange(0,8))


#LINEAR PLOTS
#------------
#KAPPA
ax2.plot(np.linspace(0,359,360),mlkap,color="magenta",linewidth=3)
ax2.plot(np.linspace(0,359,360),pvkapmf,color="green",linewidth=3)
ax2.set_xticks(np.linspace(0,360,9))
ax2.xaxis.set_major_locator(MultipleLocator(90))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_minor_locator(MultipleLocator(45))
ax2.set_ylim([0.75,2.1])
ax2.set_yticks(np.linspace(0.8,2,7))
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_title("Tuning Bandwidth $\kappa$",fontsize=20)
ax2.set_xlabel("Preferred hue angle [°]",fontsize=20)
ax2.set_ylabel("$\kappa$",fontsize=20)
#SURR
ax3.plot(np.linspace(0,359,360),mldep,color="magenta",linewidth=3)
ax3.plot(np.linspace(0,359,360),pvdepmf,color="green",linewidth=3)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_xticks(np.linspace(0,360,9))
ax3.xaxis.set_major_locator(MultipleLocator(90))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_minor_locator(MultipleLocator(45))
ax3.set_yticks(np.linspace(0,0.6,7))
ax3.set_ylim([0.1,0.67])
ax3.set_title("Surround Suppression Strength",fontsize=20)
ax3.set_xlabel("Surround hue angle [°]",fontsize=20)
ax3.set_ylabel("Suppression strength",fontsize=20)
plt.savefig(path+"\\new\\rms_dep_kap_plots_best_models.pdf")
"""
#POLAR KAPPA
#-----------
#ML
#ax2.set_title("Kappa distribution",fontsize=30,y=1.08)#Do with text
ax2.plot(np.deg2rad(np.linspace(0,359,360)),mlkap,".",color="magenta")
ax2.set_ylim(0.8,1.3)
ax2.set_yticks(np.arange(0.8,1.3,0.1))
ax2.tick_params(axis='both', which='major', labelsize=15)

#PV
ax3.plot(np.deg2rad(np.linspace(0,359,360)),pvkap,".",color="green")
ax3.set_ylim(1.7,1.9)
ax3.set_yticks(np.arange(1.7,1.9,0.1))
ax3.tick_params(axis='both', which='major', labelsize=15)

#POLAR SURR
#----------
#ML
#plt.title("Surround modulation strength",fontsize=30,y=1.08)#Text
ax4.plot(np.deg2rad(np.linspace(0,359,360)),mldep,".",color="magenta")
ax4.set_ylim(0.1,0.5)
ax4.set_yticks(np.arange(0.1,0.5,0.1))
ax4.tick_params(axis='both', which='major', labelsize=15)

#PV
ax5.plot(np.deg2rad(np.linspace(0,359,360)),pvdep,".",color="green")
ax5.set_ylim(0.3,0.7)
ax5.set_yticks(np.arange(0.4,0.7,0.1))
ax5.tick_params(axis='both', which='major', labelsize=15)

#Plot fine tuning
plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.98, wspace=0.12, hspace=0.41)
"""


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
        symVal={"rms":{},"std":{},"angdif":{}}#angdif is the absolute mean angle difference per surround
        
        fig=plotter.plot_template(auto=True)
        plt.title("Symmetry analysis of %s"%(name),y=1.08,fontsize=30)
        plt.xlabel("Absolute center surround angle difference [°]",fontsize=30)
        plt.ylabel("Absolute hue shift [°]",fontsize=30)
        for i in range(0,len(dicti)):
            csdneg=np.flip(abs(np.array(list(dicti[angles[i]]["angshi"].keys())))[0:7],0)
            angsneg=np.flip(abs(np.array(list(dicti[angles[i]]["angshi"].values())))[0:7],0)
            negse=np.flip(abs(np.array(list(dicti[angles[i]]["se"].values())))[0:7],0)#standard error values at negative side
            
            csdpos=np.array(list(dicti[angles[i]]["angshi"].keys()))[8:-1]
            angspos=np.array(list(dicti[angles[i]]["angshi"].values()))[8:-1]
            posse=np.array(list(dicti[angles[i]]["se"].values()))[8:-1]
            
            angdif = np.mean(abs(angsneg-angspos))
            
            rms=np.sqrt(((angsneg-angspos)**2).mean())
            sd=np.std(abs(angsneg-angspos))#the standart deviation of the angle difference between halves (all positive)
            symVal["rms"].update({angles[i]:rms})
            symVal["std"].update({angles[i]:sd})
            symVal["angdif"].update({angles[i]:angdif})
            
            
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
        symVal["ml"].update({"angdif":{},"rms":{}})
        symVal["vecsum"].update({"angdif":{},"rms":{}})
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
                angdif = np.mean(abs(angsneg-angspos))
                rms=np.sqrt(((angsneg-angspos)**2).mean())
                symVal[i]["rms"].update({j:rms})
                symVal[i]["angdif"].update({j:angdif})
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
decsym=symmetry_analyzer(datDicts)#decoder symmetry (added the absolute mean difference per surround per 30.06)

mlsym = decsym["ml"]
popvecsym = decsym["vecsum"]


mlmeansymdif = np.mean(np.array(list(mlsym["angdif"].values())))
mlsymdifstd = np.std(np.array(list(mlsym["angdif"].values())))

pvmeansymdif = np.mean(np.array(list(popvecsym["angdif"].values())))
pvsymdifstd = np.std(np.array(list(popvecsym["angdif"].values())))

datmeansymdif = np.mean(np.array(list(datsym["angdif"].values())))
datsymdifstd = np.std(np.array(list(datsym["angdif"].values())))

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
#mlbestRMSskm=mlParamsskm2[mlindskm2[0]]["dif"]

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
COMPARE THE VECSUM ERRORS BETWEEN MAXFR VS TOTAREA NORMALIZATION:
"""
def vecsum_error_comparison(ksi,phInt,kus,kbs,kstep):#WRONG, take the absolute error all the time!!!
    #ADDITIONALLY; THE SURROUND IS UTTERLY IRRELEVANT YOU HAVE NO SUPPRESSION IBRAHIM WTF!
    #MODDEPBEL AND MODDEPUP AS WELLL UGHHHHH as no surround modulation
    #compare vecsum decoding errors for maxfr vs totarea, firstly modelfit values, not all and take average over all surrounds for each decoder and do a scatterplot
    meanlisttotar=[]#mean decoding bias in degrees for the models
    meanlistmaxact=[]
    scannum=0
    for i in range(0,len(ksi)):#From here on, each parameter is scanned as a nested loop, so each parameter combination can be considered
        for m in range(0,len(phInt)):
            phase=phInt[m]
            for j in range(0,100):
                ku=-kstep*j+kus
                for k in range(0,100):
                    kb=kstep*k+kbs
                    if kb>=ku:
                        break
                    print("kbel=%s,kup=%s,ksur=%s,phase=%s"%(kb,ku,ksi[i],phase))#The model parameters
                    #the models, surround 180 as any surround goes (irrelevant for us what surround)
                    modtotarea = col.colmod(None,ksi[i],None,stdInt=[ku,kb],bwType="gradient/sum",phase=phInt[m],depmod=True,stdtransform=False)
                    modmaxfr = col.colmod(None,ksi[i],None,stdInt=[ku,kb],bwType="gradient/max",phase=phInt[m],depmod=True,stdtransform=False)
                    vstotarea = col.decoder.vecsum(modtotarea.x,modtotarea.centery,modtotarea.unitTracker,dataFit=True)
                    vsmaxfr = col.decoder.vecsum(modmaxfr.x,modmaxfr.centery,modmaxfr.unitTracker,dataFit=True)
                    biastotarea = np.mean(abs(np.array(list(vstotarea.angShift))))
                    biasmfr = np.mean(abs(np.array(list(vsmaxfr.angShift))))
                    print(biastotarea,biasmfr)
                    meanlisttotar.append(biastotarea)
                    meanlistmaxact.append(biasmfr)
                    scannum=scannum+1
                    print(scannum)
    return meanlisttotar,meanlistmaxact

ksur = np.linspace(0.1,2.3,10)
kbs,kus = 0.5,2
delta = 0.2
phInt=np.linspace(0,157.5,8)
                               
biastotar,biasmaxact=vecsum_error_comparison(ksur,phInt,kus,kbs,delta)
sp=col.pathes.scanpath
g = open(sp+'\\popvec_decobias_maxfr_vs_totact_phase_%s_%s.pckl'%(phInt,date),'wb')#no decoder correction
pickle.dump([biastotar,biasmaxact], g)

#Open already scanned data
sp=col.pathes.scanpath
biastotar,biasmaxact = pickle.load(open(sp+"\\popvec_decobias_maxfr_vs_totact_phase_[  0.   22.5  45.   67.5  90.  112.5 135.  157.5]_2020-05-26.pckl","rb"))
plt.figure()
plt.plot(biastotar,biasmaxact,".",color="black")                               
plt.plot([0,6],[0,6],'--',color="gray")
plt.xlabel("Total area normalized",size=20)
plt.ylabel("Maximum activity normalized",size=20)
plt.title("Population vector decoding error comparison",size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim([0,6.1])
plt.ylim([0,6.1])
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.savefig(path+"\\new\\popvec_decoding_bias_comparison.pdf")

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
                                              TODO: TRY BIC WITH CHI SQUARE done
*BIC for nested models: (chi_sq(simple model error)-chi_sq(complex model error))-ln(T)*(k(complex)-k(simple)) TODO: implement done
*R square: R^2= 1-(SSE/TSS) , SSE is unweighted sum of square errors for all conditions ((dec-dat)**2)
and TSS is the sum of squared deviations around the mean of the observed data (averaged across conditions), corresponding to null model

READ TW CHAPTERS!
MAKE TABLE OF ALL VALUES as excel folder
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
popVecParamsmf,popVecRmsmf,popvecfltmf,popvecindmf,popvecoutmf=file_opener("paraml_fit_10_decoder_vecsum_errType_rms_2020-01-25_nocorr_maxnorm",4.5)

"""
ml
"""
mlParams,mlRms,mlflt,mlind,mlout=file_opener("paraml_fit_10_decoder_ml_errType_rms_2019-08-23",4)
mlParamsuni,mlRmsuni,mlfltuni,mlinduni,mloutuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-14_uni",4.5)
mlParamscuni,mlRmscuni,mlfltcuni,mlindcuni,mloutcuni=file_opener("paraml_fit_10_decoder_ml_errType_rms_2020-02-15_unicent",4.5)
mlParamsmf,mlRmsmf,mlfltmf,mlindmf,mloutmf=file_opener("paraml_fit_10_decoder_ml_errType_rms_phase=22.5_maxfr_2020-05-26",4)

"""
ml surkap modulated
"""
mlParamsskm,mlRmsskm,mlfltskm,mlindskm,mloutskm=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-22_surkap_modulated_ksurphase_112.5",4)#surround kappa modulated scan
mlParamsskm2,mlRmsskm2,mlfltskm2,mlindskm2,mloutskm2=file_opener("paraml_fit_7_decoder_ml_errType_rms_2020-02-20_surkap_modulated",4)#surround kappa modulated scan
mlParamsskm3,mlRmsskm3,mlfltskm3,mlindskm3,mloutskm3=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=22.5_kcentphase=112.5_2020-03-17",5)#not worked out well
mlParamsskm4,mlRmsskm4,mlfltskm4,mlindskm4,mloutskm4=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well
mlParamsskm5,mlRmsskm5,mlfltskm5,mlindskm5,mloutskm5=file_opener("paraml_fit_10_decoder_ml_errType_rms_ksurphase=112.5_kcentphase=22.5_2020-03-18",5)#not worked out well

"""
Best model parameter extractor function
"""
def param_extractor(moddict,indlist,**kwargs):#moddict is the dictionary list of the fit model having all models, indlist is the index list of the dictionary list,**kwargs other model parameters (see colmod)
    """
    Create a dictionary paramdict of wished model parameters
    """
    dicti=moddict[indlist[0]]
    paramdict={"depb":None,"depu":None,"kb":None,"ku":None,"ksi":None,"phase":None,"kci":None,"depval":None,"ksb":None,"ksu":None,"ksurphase":None,"kcentphase":None}#Preallocate
    for i in dicti.keys():
        if i in paramdict.keys():
            paramdict[i]=dicti[i]#works
    """
    Feed the parameters to the colmod
    """
    model=col.colmod(Kcent=paramdict["kci"],Ksur=paramdict["ksi"],maxInhRate=paramdict["depval"],stdInt=[paramdict["ku"],paramdict["kb"]],\
                     phase=paramdict["phase"],depInt=[paramdict["depb"],paramdict["depu"]],KsurInt=[paramdict["ksu"],paramdict["ksb"]],\
                     ksurphase=paramdict["ksurphase"],kcentphase=paramdict["kcentphase"],**kwargs)
    return model #YOU NEED TO THINK WHAT THE MODEL IS, SUCH THAT YOU NEED TO SPECIFY THE KWARGS ACCORDINGLY (BWTYPE=, STDTRANSFORM= etc!)
            
#bwType gradient/sum, avgSur=i, depmod=True, stdtransform=False
"""
The null model, which predicts the average for each surround condition: Find the RMS per surround condition to estimate how well the models explain the variability.
Here, chi square will be done, where each data value is weighted by the se.
"""
avgvals=[]#average hue shift value per surround, spanning from 0 to 315 degrees. These values are null model estimation
for i in list(dictTot.keys()):
    meanval=np.mean(np.array(list(dictTot[i]["angshi"].values())))
    avgvals.append(meanval)

"""
Global mean of the data (used for R^2, can also be used for null model)
Everything denoted with g in the beginning are now related to the global mean null model.
"""
gmean=np.mean(avgvals)

"""
!!!RMS is based on standard error! TO get back to variance, first compute standard deviation (SE*sqrt(n))
Here n=5 (5 participants, SE was calculated for each surround as sd(list(mean(observer)))/sqrt(# of each observer))
"""

#RMS calculation:
nullRMS=[]#RMS per each surround
gnullRMS=[]
for i in list(dictTot.keys()):
    angshi=np.array(list(dictTot[i]["angshi"].values()))
    sterr=np.array(list(dictTot[i]["se"].values()))
    RMS=np.sqrt(np.mean((((avgvals[list(dictTot.keys()).index(i)]-angshi)/sterr)**2))) ##Formula: sqrt(mean(((model-data)/SE_data)^2))
    gRMS=np.sqrt(np.mean((((gmean-angshi)/sterr)**2))) ##Formula: sqrt(mean(((model-data)/SE_data)^2))
    nullRMS.append(RMS)
    gnullRMS.append(gRMS)

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
gnullchi=np.array(gnullRMS)**2*16/5#chi square per each surround: sum(((dec-dat)/std(dat))**2) where dec and dat are hue shifts per surround (n=16)

"""
RMS**2/16 gives sum(((dec-dat)/SE)**2) and to come to std from SE (as in chi square=sum(((dec-dat)/std)**2)) SE has to be 
multiplied by sqrt(N) (number of participants, N=5), whereas all is squared and n is constant so it can be taken all the 
way out of the sum, i.e. RMS=sqrt(1/16*sum(((dec-dat)/s)**2*5)) 
(where 16 is number of avg. observer datapoints per surround and 5 is number of observers)
"""                                

"""
Model error chi square with descending complexity (df inbetween for each is 1)
"""
#ML
mlchiun=np.array(list(mlParamsuni[mlinduni[0]]["dif"].values()))**2*16/5
mlchicun=np.array(list(mlParamscuni[mlindcuni[0]]["dif"].values()))**2*16/5
mlchinun=np.array(list(mlParams[mlind[0]]["dif"].values()))**2*16/5
mlchinunmf=np.array(list(mlParamsmf[mlindmf[0]]["dif"].values()))**2*16/5

#Popvec
pvchiun=np.array(list(popVecParamsuni[popvecinduni[0]]["dif"].values()))**2*16/5
pvchicun=np.array(list(popVecParamscuni[popvecindcuni[0]]["dif"].values()))**2*16/5
pvchinun=np.array(list(popVecParams[popvecind[0]]["dif"].values()))**2*16/5
pvchinunmf=np.array(list(popVecParamsmf[popvecindmf[0]]["dif"].values()))**2*16/5

"""
# of variables for each model
"""
#ML
nullparams=len(dictTot)
gnullparams=1
mlparamUn=len(mlParamsuni[0])-1
mlparamCun=len(mlParamscuni[0])-2#as kb=ku
mlparamNun=len(mlParams[0])-1

#Popvec
pvparamUn=len(popVecParamsuni[0])-1
pvparamCun=len(popVecParamscuni[0])-2#as kb=ku
pvparamNun=len(popVecParams[0])-1
"""
chi square test between models
"""
#ML
chiuncunml=np.sum(mlchiun-mlchicun)#chi test between uniform and center only uniform
dfuncunml=mlparamCun-mlparamUn
puncunml=1-chisq.cdf(chiuncunml,dfuncunml)#says zero, so significant

chicunnunml=np.sum(mlchicun-mlchinun)#chi test between center only uniform and non-uniform  
dfcunnunml=mlparamNun-mlparamCun
pcunnunml=1-chisq.cdf(chicunnunml,dfcunnunml)#says zero, so significant

chiunnunml=np.sum(mlchiun-mlchinun)#chi test between center only uniform and non-uniform  
dfunnunml=mlparamNun-mlparamUn
punnunml=1-chisq.cdf(chiunnunml,dfunnunml)#says zero, so significant


#compare with null model (gmeans)
chiungnullml=np.sum(gnullchi-mlchiun)#chi test between center only uniform and non-uniform  
dfungnullml=mlparamUn-gnullparams
pungnullml=1-chisq.cdf(chiungnullml,dfungnullml)#says zero, so significant

chicungnullml=np.sum(gnullchi-mlchicun)#chi test between center only uniform and non-uniform  
dfcungnullml=mlparamCun-gnullparams
pcungnullml=1-chisq.cdf(chicungnullml,dfcungnullml)#says zero, so significant

chinungnullml=np.sum(gnullchi-mlchinun)#chi test between center only uniform and non-uniform  
dfnungnullml=mlparamNun-gnullparams
pnungnullml=1-chisq.cdf(chinungnullml,dfnungnullml)#says zero, so significant

chinungnullmlmf=np.sum(gnullchi-mlchinunmf)#chi test between center only uniform and non-uniform  
dfnungnullmlmf=mlparamNun-gnullparams
pnungnullmlmf=1-chisq.cdf(chinungnullmlmf,dfnungnullmlmf)#says zero, so significant

#Popvec
chiuncunpv=np.sum(pvchiun-pvchicun)#chi test between uniform and center only uniform
dfuncunpv=pvparamCun-pvparamUn
puncunpv=1-chisq.cdf(chiuncunpv,dfuncunpv)#says zero, so significant

chicunnunpv=np.sum(pvchicun-pvchinun)#chi test between center only uniform and non-uniform  
dfcunnunpv=pvparamNun-pvparamCun
pcunnunpv=1-chisq.cdf(chicunnunpv,dfcunnunpv)

chiunnunpv=np.sum(pvchiun-pvchinun)#chi test between center only uniform and non-uniform  
dfunnunpv=pvparamNun-pvparamUn
punnunpv=1-chisq.cdf(chiunnunpv,dfunnunpv)#says zero, so significant
#chi=chi(simple model)-chi(complex model), df is k(complex)-k(simple)

#compare with null model
chiungnullpv=np.sum(gnullchi-pvchiun)#chi test between center only uniform and non-uniform  
dfungnullpv=pvparamUn-gnullparams
pungnullpv=1-chisq.cdf(chiungnullpv,dfungnullpv)#says zero, so significant

chicungnullpv=np.sum(gnullchi-pvchicun)#chi test between center only uniform and non-uniform  
dfcungnullpv=pvparamCun-gnullparams
pcungnullpv=1-chisq.cdf(chicungnullpv,dfcungnullpv)#says zero, so significant

chinungnullpv=np.sum(gnullchi-pvchinun)#chi test between center only uniform and non-uniform  
dfnungnullpv=pvparamNun-gnullparams
pnungnullpv=1-chisq.cdf(chinungnullpv,dfnungnullpv)#says zero, so significant

chinungnullpvmf=np.sum(gnullchi-pvchinunmf)#chi test between center only uniform and non-uniform  
dfnungnullpvmf=pvparamNun-gnullparams
pnungnullpvmf=1-chisq.cdf(chinungnullpv,dfnungnullpvmf)#says zero, so significant

"""
AIC of the models (to compare with null model as they are not nested)
"""
#Null
gnullaic=np.sum(gnullchi)+2
nullaic=np.sum(nullchi)+2*nullparams

#ML
mlunaic=np.sum(mlchiun)+2*mlparamUn
mlcunaic=np.sum(mlchicun)+2*mlparamCun
mlnunaic=np.sum(mlchinun)+2*mlparamNun
mlnunaicmf=np.sum(mlchinunmf)+2*mlparamNun

#Popvec
pvunaic=np.sum(pvchiun)+2*pvparamUn
pvcunaic=np.sum(pvchicun)+2*pvparamCun
pvnunaic=np.sum(pvchinun)+2*pvparamNun
pvnunaicmf=np.sum(pvchinunmf)+2*pvparamNun

#AIC_model=chi_sq(model error)+2k (k:# of parameters)


"""
BIC of the models (to compare with null model as they are not nested)
"""
obsnum=len(dictTot.keys())*len(dictTot[0]["angshi"].keys())#number of observations over all surrounds

#Null
gnullbic=np.sqrt(np.sum(gnullchi))+np.log(obsnum)*1
nullbic=np.sqrt(np.sum(nullchi))+np.log(obsnum)*nullparams

gnullbic2=np.sum(gnullchi)+np.log(obsnum)*1
nullbic2=np.sum(nullchi)+np.log(obsnum)*nullparams


#ML
mlunbic=np.sqrt(np.sum(mlchiun))+np.log(obsnum)*mlparamUn
mlcunbic=np.sqrt(np.sum(mlchicun))+np.log(obsnum)*mlparamCun
mlnunbic=np.sqrt(np.sum(mlchinun))+np.log(obsnum)*mlparamNun

mlunbic2=np.sum(mlchiun)+np.log(obsnum)*mlparamUn
mlcunbic2=np.sum(mlchicun)+np.log(obsnum)*mlparamCun
mlnunbic2=np.sum(mlchinun)+np.log(obsnum)*mlparamNun
mlnunbic2mf=np.sum(mlchinunmf)+np.log(obsnum)*mlparamNun

#Values not as expected and not fitting to AIC outcome

#Popvec
pvunbic=np.sqrt(np.sum(pvchiun))+np.log(obsnum)*pvparamUn
pvcunbic=np.sqrt(np.sum(pvchicun))+np.log(obsnum)*pvparamCun
pvnunbic=np.sqrt(np.sum(pvchinun))+np.log(obsnum)*pvparamNun

pvunbic2=np.sum(pvchiun)+np.log(obsnum)*pvparamUn
pvcunbic2=np.sum(pvchicun)+np.log(obsnum)*pvparamCun
pvnunbic2=np.sum(pvchinun)+np.log(obsnum)*pvparamNun
pvnunbic2mf=np.sum(pvchinunmf)+np.log(obsnum)*pvparamNun

#BIC_model=sqrt(chi_sq(model error))+ln(T)*k (T:# of observations) TRY WITH CHI SQUARE

"""
BIC for nested models NOT USEFUL FOR MY CASE AS THIS ALSO NOT APPLICABLE FOR NESTED MODELS!!!
USE TO COMPARE WITH NULL MODEL (GMEANS)
"""
#ML
bicuncun=np.sum(mlchiun)-np.sum(mlchicun)-np.log(obsnum)*(mlparamCun-mlparamUn)
biccunnun=np.sum(mlchicun)-np.sum(mlchinun)-np.log(obsnum)*(mlparamNun-mlparamCun)
bicunnun=np.sum(mlchiun)-np.sum(mlchinun)-np.log(obsnum)*(mlparamNun-mlparamUn)
#values as expected

#nullmodel comparison
bicungnull=np.sum(gnullchi)-np.sum(mlchiun)-np.log(obsnum)*(mlparamUn-1)
biccungnull=np.sum(gnullchi)-np.sum(mlchicun)-np.log(obsnum)*(mlparamCun-1)
bicnungnull=np.sum(gnullchi)-np.sum(mlchinun)-np.log(obsnum)*(mlparamNun-1)
bicnungnullmf=np.sum(gnullchi)-np.sum(mlchinunmf)-np.log(obsnum)*(mlparamNun-1)


#Popvec
bicuncunpv=np.sum(pvchiun)-np.sum(pvchicun)-np.log(obsnum)*(pvparamCun-pvparamUn)
biccunnunpv=np.sum(pvchicun)-np.sum(pvchinun)-np.log(obsnum)*(pvparamNun-pvparamCun)
bicunnunpv=np.sum(pvchiun)-np.sum(pvchinun)-np.log(obsnum)*(pvparamNun-pvparamUn)

#nullmodel comparison
bicungnullpv=np.sum(gnullchi)-np.sum(pvchiun)-np.log(obsnum)*(pvparamUn-1)
biccungnullpv=np.sum(gnullchi)-np.sum(pvchicun)-np.log(obsnum)*(pvparamCun-1)
bicnungnullpv=np.sum(gnullchi)-np.sum(pvchinun)-np.log(obsnum)*(pvparamNun-1)
bicnungnullpvmf=np.sum(gnullchi)-np.sum(pvchinunmf)-np.log(obsnum)*(pvparamNun-1)


#nestedbic=(chi_sq(simple model error)-chi_sq(complex model error))-ln(T)*(k(complex)-k(simple))

"""
R square (TSS uses GLOBAL MEAN of the datapoint!)
"""
def sse_calc(dec,data):#dec is the best model angshift predictions, data is the data dictionary for a given surround
    return np.sum((dec-np.array(list(data["angshi"].values())))**2)

tss=[]
ssenun=[]#for mlind
ssenunmf=[]#for mlind
ssecun=[]#for mlindcuni
sseun=[]#for mlinduni

pvssenun=[]
pvssenunmf=[]
pvssecun=[]
pvsseun=[]


for i in dictTot.keys():
    print(i)
    tss.append(sse_calc(gmean,dictTot[i]))
  
    #ML
    modnun=param_extractor(mlParams,mlind,bwType="gradient/sum",avgSur=i,depmod=True,stdtransform=False)
    decnun=col.decoder.ml(modnun.x,modnun.centery,modnun.resulty,modnun.unitTracker,avgSur=i,dataFit=True)
    ssenun.append(sse_calc(decnun.angShift,dictTot[i]))
    
    modnunmf=param_extractor(mlParamsmf,mlindmf,bwType="gradient/max",avgSur=i,depmod=True,stdtransform=False)
    decnunmf=col.decoder.ml(modnunmf.x,modnunmf.centery,modnunmf.resulty,modnunmf.unitTracker,avgSur=i,dataFit=True)
    ssenunmf.append(sse_calc(decnunmf.angShift,dictTot[i]))
    
    modcun=param_extractor(mlParamscuni,mlindcuni,bwType="gradient/sum",avgSur=i,depmod=True,stdtransform=False)
    deccun=col.decoder.ml(modcun.x,modcun.centery,modcun.resulty,modcun.unitTracker,avgSur=i,dataFit=True)
    ssecun.append(sse_calc(deccun.angShift,dictTot[i]))
    
    modun=param_extractor(mlParamsuni,mlinduni,bwType="regular",avgSur=i,depmod=False,stdtransform=False)
    decun=col.decoder.ml(modun.x,modun.centery,modun.resulty,modun.unitTracker,avgSur=i,dataFit=True)
    sseun.append(sse_calc(decun.angShift,dictTot[i]))
    
    #Popvec
    modnunpv=param_extractor(popVecParams,popvecind,bwType="gradient/sum",avgSur=i,depmod=True,stdtransform=False)
    decnunpv=col.decoder.vecsum(modnunpv.x,modnunpv.resulty,modnunpv.unitTracker,avgSur=i,dataFit=True)
    pvssenun.append(sse_calc(decnunpv.angShift,dictTot[i]))
    
    modnunpvmf=param_extractor(popVecParamsmf,popvecindmf,bwType="gradient/max",avgSur=i,depmod=True,stdtransform=False)
    decnunpvmf=col.decoder.vecsum(modnunpvmf.x,modnunpvmf.resulty,modnunpvmf.unitTracker,avgSur=i,dataFit=True)
    pvssenunmf.append(sse_calc(decnunpvmf.angShift,dictTot[i]))

    
    modcunpv=param_extractor(popVecParamscuni,popvecindcuni,bwType="gradient/sum",avgSur=i,depmod=True,stdtransform=False)
    deccunpv=col.decoder.vecsum(modcunpv.x,modcunpv.resulty,modcunpv.unitTracker,avgSur=i,dataFit=True)
    pvssecun.append(sse_calc(deccunpv.angShift,dictTot[i]))
    
    modunpv=param_extractor(popVecParamsuni,popvecinduni,bwType="regular",avgSur=i,depmod=False,stdtransform=False)
    decunpv=col.decoder.vecsum(modunpv.x,modunpv.resulty,modunpv.unitTracker,avgSur=i,dataFit=True)
    pvsseun.append(sse_calc(decunpv.angShift,dictTot[i]))


#ML
r2nun=1-(np.sum(ssenun)/np.sum(tss))
r2nunmf=1-(np.sum(ssenunmf)/np.sum(tss))
r2cun=1-(np.sum(ssecun)/np.sum(tss))
r2un=1-(np.sum(sseun)/np.sum(tss))

#Popvec
r2nunpv=1-(np.sum(pvssenun)/np.sum(tss))
r2nunpvmf=1-(np.sum(pvssenunmf)/np.sum(tss))
r2cunpv=1-(np.sum(pvssecun)/np.sum(tss))
r2unpv=1-(np.sum(pvsseun)/np.sum(tss))

#values like expected, only addition of parameters small percentage increase in explained variability.
#Popvec wise interesting outcome, that best overall is pv center only uniform.


#*R^2= 1-(SSE/TSS) , SSE is UNWEIGHTED sum of square errors for all conditions ((dec-dat)**2)
#and TSS is the sum of squared deviations around the mean of the observed data (averaged across conditions)
#corresponding to null model encorporating the GLOBAL MEAN as the single parameter


#END



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


"""
SAD MADNESS
"""
maxtilt = np.zeros(len(dictTot))
angdifmax = np.zeros(len(dictTot))
for i in range(len(dictTot)):
    maxtilt[i] = np.max(np.abs(np.array(list(dictTot[i*45]['angshi'].values()))))
    
maxsurround = np.where(maxtilt == max(maxtilt))[0][0]*45
maxsurangs = np.array(list(dictTot[maxsurround]["angshi"].values()))
stds = np.array(list(dictTot[maxsurround]["se"].values()))*np.sqrt(5)
std = stds[maxsurangs==max(maxsurangs)] #standard deviation of the maximum hue shift value.

#do the simulations:
model = col.colmod(None, 2.3, None, [1.2,0.9], bwType="gradient/sum", phase=22.5, avgSur=maxsurround, depInt=[0.2,0.4],\
                   depmod=True, stdtransform=False)

dec = col.decoder.ml(model.x, model.centery, model.resulty, model.unitTracker, avgSur=maxsurround, dataFit=True)

#schizo for different rates of sur suppression
cohensd = np.zeros(4)
idx = 0
for i in [0.01, 0.03, 0.05, 0.1]:    
    modelschiz = col.colmod(None, 2.3, None, [1.2,0.9], bwType="gradient/sum", phase=22.5, avgSur=maxsurround, depInt=[0.2-i,0.4-i],\
                        depmod=True, stdtransform=False)
    decschiz = col.decoder.ml(modelschiz.x, modelschiz.centery, modelschiz.resulty, modelschiz.unitTracker, avgSur=maxsurround, dataFit=True)
    cohensd[idx] = (max(dec.angShift)-max(decschiz.angShift))/std
    idx += 1

ntots = [145, 44, 24, 12] #balanced design etc etc....
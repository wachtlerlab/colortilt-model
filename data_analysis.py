# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:02:37 2019

@author: Ibrahim Alperen Tunc
"""
"""
Paramter scan script to fit the model to the psychophysics data (Klauke & Wachtler)
TO DO: Do the parameter scan for the population vector error corrected. DONE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import colclass as col
import sys
sys.path.insert(0,col.pathes.runpath)#!Change the directory accordingly
from scipy import stats as st
from supplementary_functions import std2kappa, depth_modulator, plotter, param_dict
import pickle
from datetime import date

sp=col.pathes.scanpath

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

dictTot=param_dict(np.linspace(0,315,8),params)#upper dictionaries are special subject values, dictTot is the mean of all 5 subjects.

"""
Transfer the data into dictionaries
"""
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
Get a mean value from all individuals for given surround and find the SE (mean of means/sqrt(number of means))
"""
for i in range(0,360,45):#loop for each surround stimulus angle condition.
    for k in range(0,16):#loop for each center hue angle measured in each surround.
        a=[list(dictTW[i][params[0]].values())[k],list(dictSU[i][params[0]].values())[k],list(dictMH[i][params[0]].values())[k],\
           list(dictLH[i][params[0]].values())[k],list(dictHN[i][params[0]].values())[k]]#list of each subject angshi values for the given surround
        dictTot[i][params[0]].update({-157.5+22.5*k:np.mean(a)})#update the dictTot angshi value with average, where key is the corresponding csd value.
        dictTot[i][params[1]].update({-157.5+22.5*k:st.sem(a)})##update the dictTot se value, where key is the corresponding csd value.

#plt.plot(dictTot[45]["angshi"].keys(),dictTot[45]["angshi"].values(),color="black")
plt.errorbar(dictTot[0]["angshi"].keys(),dictTot[0]["angshi"].values(),dictTot[0]["se"].values(),ecolor="red",color="black",capsize=3)#exemplary plot of the data.


"""
Now data fit analysis, ml decoder is optimized to return only the angle values evident in data.
model type: ml gs depmod
parameters:Kcent, Ksur, maxInh, stdInt, depInt (phase of both BW and depInt firstly set as 22.5), the influence of phase can also be further studied, maybe phase leads to a change?
Possible data fit quantifying methods: Root mean square error, mean absolute error, Tucker-Lewis index, Comparative fit index. The last 2 indices should be improved regarding the choice of df.
"""

"""
The comparative fit index (CFI) should not be computed if the RMSEA of the null model is less than 0.158 or otherwise one will obtain 
too small a value of the CFI. Following loop checks this case.
"""
for i in range(0,360,45):
    print(i)
    a=((np.array(list(dictTot[i]["angshi"].values())).mean()-np.array(list(dictTot[i]["angshi"].values())))\
                /np.array(list(dictTot[i]["se"].values())))**2
    if np.sqrt(a.mean())<0.158:
        print("cfi cannot be used")
        break
    if i==315:
        print("all ok, ready to roll with cfi")

def data_dist(Kcent,Ksur,maxInh,stdInt,depInt,fitThres,phase=22.5,errType="rms",deco="ml",dicti=dictTot,se=True,errN=False,bwType="gradient/sum",KsurInt=None,ksurphase=None,kcentphase=None):
    """
    Data fit function:
        Checks the fit quality of the model with given parameters to the psychophysics data by using the given error estimation method.
        If the model fit is not as good as the chosen threshold, function stops running. Maximum likelihood decoder is used with the
        non-uniform model normalized with total unit activity and non-uniform surround suppression.
        
        Pararmeters
        -----------
        Kcent: float. The Kappa (concentration parameter) for the center unit tuning curves. Here can be given any value as this parameter
        has no effect in case of the non-uniform model.
        Ksur: float. The Kappa (concentration parameter) for the surround modulation curve. This parameter is the same for all surround conditions.
        maxInh: float. Maximum inhibiton rate for the surround suppression curve. Here can be given any value as this parameter has no effect in 
        case of the non-uniform model.
        stdInt: list. The upper and lower Kappa values of the center tuning curves in the model. Note that the first value here has to be the bigger
        Kappa, as the bigger kappa corresponds to the smaller standard deviation value. In other words stdInt should be given as [Kappa_up,Kappa_below].
        depInt: list. The lower and upper values of the maximum surround suppression for different surround conditions. depInt should be given as [dep_below,dep_up]
        fitThres: float. The threshold for the fit to be considered as good. For root mean square error (rms) and mean absolute error (mae), the fit is considered
        to be poor when the fit value is bigger than fit threshold. For comparative fit index (CFI) and Tucker-Lewis index (TLI) the fit quality is considered to be poor
        when the fit value is smaller than fit threshold. For the last 2 fit quantifying methods, the fit value is between 0-1
        phase: float, optional. The phase of the non-uniformity in the model. Chosen as 22.5° (blue-yellow axis) for standard. 
        errType: string, optional. The method to be used for quantifying the fit quality. "rms"=Root mean square error, "mae"= mean absolute error, "cfi"= comparative fit index,
        "tli"= Tucker-Lewis index. For a detailed information about each of the model fit measurement: http://www.davidakenny.net/cm/fit.htm
        dec: string, optional. The decoder type to be used. The variable is "ml" (maximum likelihood) for standard, can also be "vecsum" (population vector).
        dicti: dictionary, optional. The dictionary object which contains the data to be analyzed. The default is the average observer dictionary. 
        se: boolean, optional. If false, then the scan is done without se normalization and RMS is based on absolute angular values.
        errN: boolean, optional. Only useful vor population vector decoder. If true, the decoder is error corrected (for details see colclass.)
        KsurInt: list, optional. If a list value is given, then surround kappa is also phase modulated. Default is None. Give as [ku,kb]
        ksurphase: integer, optional. Default is 0. If a number is specified, the phase of kappa surround is shifted by the given value in degrees.
        kcentphase: integer, optional. Default is None. If a number is specified, the phase of center unit kappa is changed by the given value in degrees. If the value
        is None (as in default), then the center unit phase modulation is via the variable "phase" (see colclass.py).
    

        Note that the rms and mae values are given in terms of data standard error values. For example, rms=1 means the RMSEA between data and model is in average 1 standard error
        for each of the datapoint measured.
        
        From 15.03.2020 on, the decoder variable is no more returned by the function as this method slows down the scan significantly and the wished decoder object can always be 
        recreated by using the returned parameters by the scan_params() function.
        
        Returns
        -------
        rs: dictionary. The dictionary of fit value for each of the surround condition.
        modelfit: boolean. If True, the model fit works for all surround conditions, so the parameters are appended to the dictionary in function scan_params
    """
    surrAvg=np.linspace(0,315,8)#measured surround conditions in Klauke & Wachtler paper
    rs={}#Dictionary of the fit value
    modelfit=False
    
    """
    Create the model and decoder objects for each surround condition, check if the fit is good enough in each surround step and stop if not
    """
    for i in range(0,len(surrAvg)):
        print("unit activity normalization %s" %(bwType))
        if bwType=="regular":
            print("uniform")
            colMod=col.colmod(Kcent,Ksur,maxInh,avgSur=surrAvg[i],bwType=bwType)
        elif bwType!="regular":
            print("non-uniform")
            colMod=col.colmod(Kcent,Ksur,maxInh,stdInt,bwType=bwType,phase=phase,avgSur=surrAvg[i],depInt=depInt,depmod=True,stdtransform=False,KsurInt=KsurInt,ksurphase=ksurphase,kcentphase=kcentphase)#see colclass.py for details.

        if deco=="vecsum":
            print("vecsum decoder")
            dec=col.decoder.vecsum(colMod.x,colMod.resulty,colMod.unitTracker,avgSur=surrAvg[i],errNorm=errN,centery=colMod.centery,dataFit=True)
            if errN==False:
                print("Warning, the population vector decoder has no error correction.")
            else:
                print("Popvec decoder with error correction.")
        elif deco=="ml":
            print("ml decoder")
            dec=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=surrAvg[i],dataFit=True)#maximum likelihood decoder, see colclass.py for details.
        else:
            raise Exception("Wrong decoder name given, please look at documentation")
        if errType=="rms":#Root mean square error, quantified in standard error values.
            if se==True:
                sumval=np.sqrt((((np.array(dec.angShift)-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                                /np.array(list(dicti[surrAvg[i]]["se"].values())))**2).mean())#Formula: sqrt(mean(((model-data)/SE_data)^2))
                if sumval>fitThres:#stop if the fit is not good enough
                    print("Fit is not good enough for surround=%s, rs=%s"%(surrAvg[i],sumval))
                    break
            else:
                print("no standart error normalization")
                sumval=np.sqrt(((np.array(dec.angShift)-np.array(list(dicti[surrAvg[i]]["angshi"].values())))**2).mean())#Formula: sqrt(mean(((model-data)/SE_data)^2))
                if sumval>fitThres:#stop if the fit is not good enough
                    print("Fit is not good enough for surround=%s, rs=%s"%(surrAvg[i],sumval))
                    break
            
        if errType=="mae":#mean absolute error, given in terms of standard error
            sumval=(abs((np.array(dec.angShift)-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                            /np.array(list(dictTot[surrAvg[i]]["se"].values())))).mean()#Formula: mean(abs((model-data)/SE_data))
            if sumval>fitThres:#stop if the fit is not good enough
                    print("Fit is not good enough for surround=%s, rs=%s"%(surrAvg[i],sumval))
                    break    
        
        if errType=="cfi":#comparative fit index, d=chi^2(model,data)-df, cfi=(d(model)-d(nullmodel))/d(nullmodel). nullmodel always average 
                          #but with same df as my model, here the choice of df is controversial!
                          #chi^2(model,data)=((model-data)/SE(data))^2, df=number of measurements-number of model params (6).
            #The nullmodel predicts always the average of the dataa value. This prediction can also be changed to no prediction of color tilt.
            nulldist=sum(((np.array(list(dicti[surrAvg[i]]["angshi"].values())).mean()-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                            /np.array(list(dicti[surrAvg[i]]["se"].values())))**2)-(len(np.array(list(dicti[surrAvg[i]]["angshi"].values())))-6)
            #The model prediction of interest.
            moddist=sum(((np.array(dec.angShift)-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                            /np.array(list(dicti[surrAvg[i]]["se"].values())))**2)-(len(np.array(list(dicti[surrAvg[i]]["angshi"].values())))-6)
            sumval=(nulldist-moddist)/nulldist#d value of the cfi.
            """
            If sumval is not between 0 and 1, set the lower limit as 0 and upper limit as 1.
            """
            if sumval<0:
                sumval=0
            if sumval>1:
                sumval=1
            if sumval<fitThres:#Stop if fit is not good enough.
                print("Fit is not good enough for surround=%s, cfi=%s"%(surrAvg[i],sumval))
                break
        
        if errType=="tli":#Tucker Lewis index, similar to cfi, but now each chi^2 is divided to df instead of subtraction.
            #Nullmodel predicts again the data average, can be changed to prediction of no color tilt.
            nullchi=sum(((np.array(list(dicti[surrAvg[i]]["angshi"].values())).mean()-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                            /np.array(list(dicti[surrAvg[i]]["se"].values())))**2)/(len(np.array(list(dicti[surrAvg[i]]["angshi"].values())))-6)
            #Prediction of the model of interest
            modchi=sum(((np.array(dec.angShift)-np.array(list(dicti[surrAvg[i]]["angshi"].values())))\
                            /np.array(list(dicti[surrAvg[i]]["se"].values())))**2)/(len(np.array(list(dicti[surrAvg[i]]["angshi"].values())))-6)
            sumval=(nullchi-modchi)/(nullchi-1)#d value
            if sumval<0:
                sumval=0
            if sumval>1:
                sumval=1
            if sumval<fitThres:
                print("Fit is not good enough for surround=%s, tli=%s"%(surrAvg[i],sumval))
                break
        #how would be if null model would imply no hue shift? or should null model comprise hue shift without surround? then also same parameters to null model???? 
        """
        Create the dictionaries for function output
        """
        rs.update({surrAvg[i]:sumval})
        print("Next surround")
        if i==len(surrAvg)-1:
            modelfit=True
    return rs,modelfit

def scan_params(fit,ksi,kbs,kus,depbs,depus,kstep,depstep,phInt,errType="rms",deco="ml",dicti="dictTot",se=True,errN=False,bwType="gradient/sum",kci=None,depval=None,ckapun=False,KsurInt=None,KsurStep=None,ksurphase=None,kcentphase=None):
    """Parameter scan function:
        This function uses the data_dist() function to scan through all parameter combinations given in the function. Warning: The scanning
        process takes long time, in some cases even days. 
        
        Parameters
        -----------
        fit: float. The fit threshold (see also data_dist())
        ksi: tuple/list. The surround kappa interval
        kbs: float. Lower limit of the center tuning curve Kappa.
        kus: float. Upper limit of the center tuning curve Kappa.
        depbs: float. Lower limit of the maximum surround suppression.
        depus. float. Upper limit of the maximum surround suppression.
        kstep: float. Increments to increase kbs or decrease kus.
        depstep: Increments to increase depbs or decrease depus.
        phInt: list. The non-uniformity phase values to be scanned.
        errType: string. Possible fit error measurements. To see the possible strings check data_dist().
        dec: string, optional. The decoder type to be used. The variable is "ml" (maximum likelihood) for standard, can also be "vecsum" (population vector).
        dicti: string, optional. The name of the dictionary object which contains the data to be analyzed. The default name is the average observer dictionary. 
        se: boolean, optional. If false, then the scan is done without se normalization and RMS is based on absolute angular values.
        errN: boolean, optional. Only useful vor population vector decoder. If true, the decoder is error corrected (for details see colclass.)
        bwType: string, optional. Denotes the model type (see also colclass). Uniform if "regular", non-uniform for other possibilities
        kci: list, optional. The uniform scan center kappa value interval to be scanned
        depval: list, optional. The uniform scan modulation depth value interval to be scanned
        ckapun: boolean, optional. The uniformity of the center unit tuning curves in the case of non-uniform model.
        KsurInt: list, optional. If a list value is given, then surround kappa is also phase modulated. Default is None. Give as [ku,kb]
        KsurStep: float. The binning of KsurInt during scan.
        ksurphase: integer, optional. Default is 0. If a number is specified, the phase of kappa surround is shifted by the given value in degrees.
        kcentphase: integer, optional. Default is None. If a number is specified, the phase of center unit kappa is changed by the given value in degrees. If the value
        is None (as in default), then the center unit phase modulation is via the variable "phase" (see colclass.py).
    
        
        For the uniform analysis, following three parameters are necessary: center kappa, surround kappa, modulation strength, 3 free parameters as 
        opposed to non-uniform case (which has additional to surround kappa the center kapa max/min values and surround suppression max/min values as a function 
        of surround hue as well as modulation phase)
        
        Returns
        -------
        params: list. The list of dictionaries including parameters of the models giving good model fits.
    """
    scannum=0#number of scans done in the end
    print("decoder=%s for the dictionary %s"%(deco,dicti))
    dicti=eval(dicti)
    if bwType!="regular":
        kc=1#Kcent is arbitrary as the model is non-uniform!
        maxInh=1#these 2 parameters irrelevant, they dont do any job here!
        params=[]
        if KsurInt!=None:
            ksi=[1]#if surround kappa is modulated, this loop is done once.
        else:
            pass
        for i in range(0,len(ksi)):#From here on, each parameter is scanned as a nested loop, so each parameter combination can be considered
            for m in range(0,len(phInt)):
                phase=phInt[m]
                for j in range(0,100):
                    ku=-kstep*j+kus
                    #print("upper kappa=%s"%(ku))
                    if ckapun==True:#this ckapun thing might be problematic, so possible bug in scan due to this variable, seems like nothings up in regular scans tho
                        kb=ku
                        print("No center non-uniformity")
                        if ku<0:
                            break
                    b=1#b is to bypass k loop when only center tuning is uniform and mod depth iteration is over
                    for k in range(0,100):
                        if b==0 and ckapun==True:
                            break
                        a=True#a is to iterate over kbel when ckapun is false
                        if ckapun==True:
                            a=False
                        while a==True:
                            kb=kstep*k+kbs
                            a=False
                        if kb>=ku and ckapun==False:#To ensure the lower limit does not exceed the upper limit
                            break
                        #print("below kappa=%s"%(kb))
                        for l in range(0,100):
                            depu=-depstep*l+depus
                            for n in range(0,100):
                                depb=depstep*n+depbs
                                if depb>=depu:#To ensure the lower limit does not exceed the upper limit
                                    b=0
                                    break 
                            
                                if KsurInt!=None:
                                        print("surround kappa modulated")
                                        for q in range(0,100):
                                            sku=-KsurStep*q+KsurInt[0]#surround kappa upper value
                                            for c in range(0,100):
                                                skb=KsurStep*c+KsurInt[1]#surround kappa upper value
                                                if skb>=sku:
                                                    break
                                                print("moddepbel=%s,moddepup=%s,kbel=%s,kup=%s,ksur=%s,phase=%s,ksb=%s,ksu=%s,ksurphase=%s,kcentphase=%s"%(depb,depu,kb,ku,ksi[i],phase,skb,sku,ksurphase,kcentphase))#The model parameters
                                                dif,modfit=data_dist(kc,ksi[i],maxInh,stdInt=[ku,kb],depInt=[depb,depu],fitThres=fit,errType=errType,phase=phase,deco=deco,dicti=dicti,se=se,errN=errN,bwType=bwType,KsurInt=KsurInt,ksurphase=ksurphase,kcentphase=kcentphase)#fit value
                                                if modfit==True:
                                                    moddict={}
                                                    moddict.update({"depb":depb,"depu":depu,"kb":kb,"ku":ku,"ksb":skb,"ksu":sku,"phase":phase,"ksurphase":ksurphase,"dif":dif})
                                                    if kcentphase!=None:
                                                        print("center kappa modulated independent of suppression phase")
                                                        moddict.update({"kcentphase":kcentphase})
                                                    print("fit params work for each of the surround for given rms threshold")
                                                    params.append(moddict)
                                                scannum=scannum+1
                                else:
                                    print("moddepbel=%s,moddepup=%s,kbel=%s,kup=%s,ksur=%s,phase=%s,ksurphase=%s,kcentphase=%s"%(depb,depu,kb,ku,ksi[i],phase,ksurphase,kcentphase))#The model parameters
                                    dif,modfit=data_dist(kc,ksi[i],maxInh,stdInt=[ku,kb],depInt=[depb,depu],fitThres=fit,errType=errType,phase=phase,deco=deco,dicti=dicti,se=se,errN=errN,bwType=bwType,KsurInt=KsurInt,ksurphase=ksurphase,kcentphase=kcentphase)#fit value
                                    if modfit==True:
                                        moddict={}
                                        moddict.update({"depb":depb,"depu":depu,"kb":kb,"ku":ku,"ksi":ksi[i],"phase":phase,"dif":dif})
                                    scannum=scannum+1
        print(scannum)
        return params
    
    else:
        stdInt=None;depInt=None;phase=None
        params=[]
        for i in range(0,len(ksi)):#From here on, each parameter is scanned as a nested loop, so each parameter combination can be considered
            for m in range(0,len(kci)):
                for j in range(0,len(depval)):
                    print("ckappa=%s,skappa=%s,maxinh=%s"%(kci[m],ksi[i],depval[j]))#The model parameters
                    dif,dec=data_dist(kci[m],ksi[i],depval[j],stdInt=stdInt,depInt=depInt,fitThres=fit,errType=errType,phase=phase,deco=deco,dicti=dicti,se=se,errN=errN,bwType=bwType)#fit value and decoder list
                    if modfit==True:
                        print("fit params work for each of the surround for given rms threshold")
                        params.append({"kci":kci[m],"ksi":ksi[i],"depval":depval[j],"dif":dif})
                    scannum=scannum+1
        print(scannum)
        return params
        
"""
The parameter scan
"""    
phInt=np.linspace(0,157.5,8)#phase of depmod and stdInt (center units) can be scanned as well if wished.    
fit=10;errType="rms";date=date.today();decod="vecsum"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
#The line 255 is for the scan, change values here for changing the scan attributes.
"""
Save the filtered model values after parameter scan with pickle if the folders are absent.
Try is the statement to check if the code below raises exception. If so, the line in except is computed.
This part is written later on for possible parameter scans in the next time, so the name etc is specified for possible multiple paramscan
runs. 
"""
try: 
    with open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s.pckl'%(fit,decod,errType,date),"rb") as file:
        pickl=pickle.load(file)
except FileNotFoundError:    
    print("creating the pickle file")
    f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s.pckl'%(fit,decod,errType,date), 'wb')
    pickle.dump(paraml, f)
    print("pickle file is created")
except EOFError:
    print("filling the empty pickle file")
    f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s.pckl'%(fit,decod,errType,date), 'wb')
    pickle.dump(paraml, f)
    print("pickle file is filled")

"""
Same thing for the dictTot. As dicttot is the same for all cases, no naming differentiantion is done.
"""
try: 
    with open('dicttot.pckl',"rb") as file:
        pickl=pickle.load(file)
except FileNotFoundError:    
    print("creating the pickle file")
    g = open('dicttot.pckl', 'wb')
    pickle.dump(dictTot, g)
    print("pickle file is created")
except EOFError:
    print("filling the empty pickle file")
    g = open('dicttot.pckl', 'wb')
    pickle.dump(dictTot, g)
    print("pickle file is filled")

"""
Find the maximum and minimum values of the variables for the next and better scan, where threshold is smaller
This function is not used as the first scan yielded very similar model fits for the best 10 models, a further scan with smaller increments
was unnecessary.
"""

"""
Scan for each individual now instead of the average observer
"""
dicts=["dictHN","dictLH","dictMH","dictSU","dictTW"]
fit=10;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
for i in dicts:#change to dicts if you wanna redo the 1st scan
    paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,dicti=i)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
    f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s.pckl'%(fit,decod,errType,date,i), 'wb')
    pickle.dump(paraml, f)

decod="vecsum"
for i in dicts:#change to dicts if you wanna redo the 1st scan
    paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,dicti=i)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
    f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s.pckl'%(fit,decod,errType,date,i), 'wb')
    pickle.dump(paraml, f)

"""
Scan the average observer without the se normalization
"""
fit=15;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,se=False)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s.pckl'%(fit,decod,errType,date,"nose"), 'wb')
pickle.dump(paraml, f)
decod="vecsum"
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,se=False)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s.pckl'%(fit,decod,errType,date,"nose"), 'wb')
pickle.dump(paraml, f)

"""
Scan the average observer se normalization with phase=30°
"""
#phInt=np.linspace(0,157.5,8)#phase of depmod and stdInt (center units) can be scanned as well if wished.    
fit=10;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
ksi=np.linspace(1.0,2.3,14);kbs=0.7 ;kus=2 ;depbs=0.2 ;depus=0.8 ;kstep=0.2 ;depstep=0.2 ;phInt=[45]
paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=phInt,deco=decod)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s°.pckl'%(fit,decod,errType,date,phInt), 'wb')
pickle.dump(paraml, f)

decod="vecsum"

paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=[30],deco=decod)
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s°.pckl'%(fit,decod,errType,date,phInt), 'wb')
pickle.dump(paraml, f)

"""
Scan the phase as well (preliminary, make the steps shorter to see whats up)
"""
fit=7;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
ksi=np.linspace(1.0,2.3,14);kbs=0.7 ;kus=2 ;depbs=0.2 ;depus=0.8 ;kstep=0.2 ;depstep=0.2 ;phInt=np.linspace(0,157.5,8)
paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=phInt,deco=decod)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s°.pckl'%(fit,decod,errType,date,phInt), 'wb')
pickle.dump(paraml, f)

decod="vecsum"

paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=phInt,deco=decod)
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s°.pckl'%(fit,decod,errType,date,phInt), 'wb')
pickle.dump(paraml, f)

"""
Popvec decoder scan without error correction, expected is worsened data fit. fingers crossed :) Not quite as error correction not always dampens the error
"""
fit=10;errType="rms";date=date.today();decod="vecsum"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s.pckl'%(fit,decod,errType,date,"nocorr"), 'wb')#no decoder correction
pickle.dump(paraml, f)

"""
Popvec decoder scan without error correction, but model maximum activity normalized so vecsum error is dampened
"""
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False,bwType="gradient/max")#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s_maxnorm.pckl'%(fit,decod,errType,date,"nocorr"), 'wb')#no decoder correction
pickle.dump(paraml, f)

"""
Uniform model scan for both decoder types, should be pretty fast as only 3 parameters and expected is worsened fit.
"""
fit=10;errType="rms";date=date.today();decod="ml";bwType="regular"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
ksi=np.linspace(1.0,2.3,14);kci=np.linspace(0.5,2.5,11);depval=np.linspace(0,1,6);kbs=None ;kus=None ;depbs=None ;depus=None ;kstep=None ;depstep=None ;phInt=None
paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=phInt,deco=decod,kci=kci,depval=depval,bwType=bwType)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_uni.pckl'%(fit,decod,errType,date), 'wb')#no decoder correction
pickle.dump(paraml, f)


decod="vecsum" #the scan is run for 14.02.2019
paraml=scan_params(fit,ksi=ksi,kbs=kbs,kus=kus,depbs=depbs,depus=depus,kstep=kstep,depstep=depstep,errType=errType,phInt=phInt,deco=decod,kci=kci,depval=depval,bwType=bwType)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_%s_uni.pckl'%(fit,decod,errType,date,"nocorr"), 'wb')#no decoder correction
pickle.dump(paraml, f)

"""
Scan only with surround modulation
"""
phInt=np.linspace(0,157.5,8)#phase of depmod and stdInt (center units) can be scanned as well if wished.    
fit=10;errType="rms";date=date.today();decod="vecsum"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False,ckapun=True)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_nocorr_unicent.pckl'%(fit,decod,errType,date), 'wb')#no decoder correction
pickle.dump(paraml, f)

decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,10),0.5,2,0,1,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False,ckapun=True)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_unicent.pckl'%(fit,decod,errType,date), 'wb')#no decoder correction
pickle.dump(paraml, f)


"""
Do surround kappa modulated scan on ml, then see if something better is up, and do the same for popvec if yea
"""
fit=7;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,12),0.8,2,0.2,0.6,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False,KsurInt=[2.5,0.5],KsurStep=0.2,ksurphase=112.5,kcentphase=22.5)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_%s_surkap_modulated.pckl'%(fit,decod,errType,date), 'wb')#no decoder correction
pickle.dump(paraml, f)

"""
Scan where kapa surround and suppression strength phases are same but center kappa phase differs.
sur phase 22.5 cent phase orthogonal and other way around
"""
fit=7;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
ksurphase=22.5;kcentphase=112.5
paraml=scan_params(fit,np.linspace(0.1,2.3,12),0.8,2,0.2,0.6,0.2,0.2,errType=errType,phInt=[22.5],deco=decod,errN=False,KsurInt=[2.5,0.5],KsurStep=0.2,ksurphase=ksurphase,kcentphase=kcentphase)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_ksurphase=%s_kcentphase=%s_%s.pckl'%(fit,decod,errType,ksurphase,kcentphase,date),'wb')#no decoder correction
pickle.dump(paraml, f)

fit=7;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
ksurphase=112.5;kcentphase=22.5
paraml=scan_params(fit,np.linspace(0.1,2.3,12),0.8,2,0.2,0.6,0.2,0.2,errType=errType,phInt=[112.5],deco=decod,errN=False,KsurInt=[2.5,0.5],KsurStep=0.2,ksurphase=ksurphase,kcentphase=kcentphase)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
f = open(sp+'\\paraml_fit_%s_decoder_%s_errType_%s_ksurphase=%s_kcentphase=%s_%s.pckl'%(fit,decod,errType,ksurphase,kcentphase,date),'wb')#no decoder correction
pickle.dump(paraml, f)

def auto_scan(thr,ksistep,kstep,depstep,param):
    """Parameter scanner for the next iteration:
        This function takes the values of the previous parameter scan and uses them for the next scan as lower-upper limits, respectively.
        
        Parameters
        ----------
        thr: float. The fit threshold value of the next iteration.
        ksistep: float. The binning of surround Kappa interval for the next run. The function itself finds out the ksi interval to be scanned.
        kstep: float. The binning of the center unit tuning curve Kappa for the next run. The function itself finds out the center kappa interval to be scanned.
        depstep: float. The binning of the maximum surround suppression interval for the next run. The function itself finds out 
        itself the lower and upper limits of the maximum surround suppression.
        param: list. The output params of the function scan_params()
        
        Returns
        -------
        decl2: list. The list of decoder objects which yield a good data fit in the next iteration.
        paraml2: list. The list of dictionaries including parameters of the models giving good model fits in the next iteration.
    """
    
    """
    Give arbitrary starting values for the upper and lower bounds to find the biggest and smallest parameter values
    """
    depbt=100#depb total
    deput=0#depu total
    ksib=100#ksi max
    ksiu=0#ksi min
    kut=0#center kappa max total
    kbt=100#center kappa min total
     
    """
    Find the max&min values for parameters
    """
    for i in range(0,len(paraml)):
        if depbt>param[i]["depb"]:
            depbt=param[i]["depb"]
        if deput<param[i]["depu"]:
            deput=param[i]["depu"]
        if kbt>param[i]["kb"]:
            kbt=param[i]["kb"]
        if kut<param[i]["ku"]:
            kut=param[i]["ku"]
        if ksiu<param[i]["ksi"]:
            ksiu=param[i]["ksi"]
        if ksib>param[i]["ksi"]:
            ksib=param[i]["ksi"]
    print(depbt,deput,kbt,kut,ksib,ksiu)#The parameter values for the next scan.
    """
    Run the second scan
    """
    decl2,paraml2=scan_params(thr,np.linspace(ksib,ksiu,ksistep),kbt,kut,depbt,deput,kstep,depstep)
    return decl2,paraml2

"""
TW 20.03.20 14.00 skype
"""

"""
test
fit=1;errType="rms";date=date.today();decod="ml"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,2),0.8,2,0.2,0.6,0.8,0.8,errType=errType,phInt=[22.5],deco=decod,errN=False,KsurInt=[2.5,0.5],KsurStep=0.8,ksurphase=112.5,kcentphase=22.5)

phInt=np.linspace(0,157.5,8)#phase of depmod and stdInt (center units) can be scanned as well if wished.    
fit=1;errType="rms";date=date.today();decod="vecsum"#These values are used to specify the pickle file name. date.today() gives the date of today in a pretty straightforward way.
paraml=scan_params(fit,np.linspace(0.1,2.3,2),0.5,2,0,1,0.8,0.8,errType=errType,phInt=[22.5],deco=decod)#threshold=10, run it once, do the hist and LOOK AT THE FITTED CURVES FOR ALL CASES, if they reproduce the data mechanistically, all is well, do the subplot for the best fits.
"""
"""
*Development notes
Run the scan only once, keep intervals same, threshold is 2, and for the values below threshold take also rms value into consideration,
(append it in a list), then first look at the distribution of error with density plot (histogram), where we hope to get low frequency of
low error values, use it for putting a threshold on error value, after that plot the error values in parameter raum to see if theres clustering
,where low error values (according to threshold) have different color than others 
QUESTIONS:  -ROOT MEAN SQUARE OR MEAN ABSOLUTE ERROR????, what are other options to check the performance of my model? RMS is not working for a fit in 2 SE, shall i compare my model
            with a null model?
            -DEPMOD (look at colclass script colMod if depMod==True): should i also modulate below inhibition rate? if yes, how? This is due to von Mises distribution assumption, nothing to do
            -ML decoder sometimes gives 2 different decoded angle as output, meaning that the population activity is bimodal with 2 same global peaks (or better, that there
            are more than 1 look up table entries with same distance to the population activity after inhibition.), what does it mean? 
            does it directly mean that model parameters are wrongly chosen? Yes irrelevant parameters
            (example run this code):
             colMod=col.colmod(1,0.5,1,[2,0.5],bwType="gradient/sum",phase=0,avgSur=0,depInt=[0.1,0.9],depmod=True,stdtransform=False)
             dec=col.decoder.ml(colMod.x,colMod.centery,colMod.resulty,colMod.unitTracker,avgSur=0,dataFit=True,tabStep=1)

Analysis of parameter values: First set a threshold for mean rms values, then look if parameters cluster, lastly check the fits in data by plotting to see where is the fail mainly,
all ok if fits can capture important aspects like mmanshi etc. well.
"""



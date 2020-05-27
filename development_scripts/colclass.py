# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:19:20 2019

@author: Ibrahim Alperen Tunc
"""
'''
Class for the model and decoder objects. This script contains all essential codes and functions for the model and decoders.
This script is imported to other analysis scripts.

Specify the path here once.

Used libraries and modules:
NumPy, cmath, SciPy
'''
import numpy as np
import cmath as c
from scipy.optimize import curve_fit as fit
import matplotlib.pyplot as plt
from supplementary_functions import std2kappa, depth_modulator, plotter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

class pathes():
    """Specify the directory pathways for saving figures (figpath), saving scans (scanpath) or running other scripts (runpath)
    """
    figpath=r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\thesis_figures" #//TODO change the figure paths in scripts DONE
    runpath=r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python" #//TODO change the paths in scripts DONE
    scanpath=r"C:\Users\Ibrahim Alperen Tunc\.spyder-py3\bachelor_arbeit\python\scans"   #//TODO change the scan paths in scan scripts DONE
    
class colmod:#add here the kappa phase variable.
    """The model class:
        This class contains all necessary model elements (center unit tuning curves, surround modulation rate etc.) for the given model parameters. 
        The output objects of this class are then further used for the decoder functions below.
        See also model_tester.py 
        TO ADD: THE SURROUND MODULATION KAPPA PHASE DEPENDENCE!
    """
    x=np.ndarray.round(np.linspace(-60,420,num=4801),2)#round all these steps to .1 decimals
    def __init__(self,Kcent,Ksur,maxInhRate,stdInt=[60,70],bwType="regular",phase=0,avgSur=180,startAvg=1,endAvg=360,depInt=[0.2,0.6],depmod=False,stdtransform=True,KsurInt=[None],ksurphase=None,kcentphase=None):
        """Parameters
        -------------
        Kcent: float. The concentration parameter Kappa of the center unit tuning curves. This parameter is of relevance only for the uniform model.
        Ksur: float. The concentration parameter Kappa of the surround modulation curve. This parameter is relevant only when KsurInt=None
        maxInhRate: float. The maximum surround suppression value, can be between 0-1. Zero means no surround suppression, 1 means in maximum suppression
        case the unit is completely silenced. This parameter is only relevant for the uniform model.
        stdInt: list, optional. The lower and upper limits of the standard deviation of the unit tuning curves for the non-uniform model. The values are given in degrees.
        The parameter is given as [stdb,stdu] where stdb is the smallest value and stdu is the biggest value. The given std values are then transformed to Kappa by 
        using the std2kappa() function.
        bwType: string, optional. This parameter determines the model type. The model is by default uniform ("regular"). For non-uniform maximum activity normalized
        model the input is "gradient/max", for the non-uniform model total unit activity normalized is "gradient/sum"
        phase: float, optional. The angular phase of the non-uniformity in the model in degrees. Set to 0° by default but in the study the phase 22.5° was used. 
        When for example phase 0° is, the unit with the preferred hue angle of 0° and 180° have the narrowest tuning and the surround suppression is the strongest for
        the surround hue angle=0° and 180, beacuse the phase is periodic within 180°. If other phase variables are denoted, this variable is only responsible for the phase
        of the surround suppression (depmod=True).
        avgSur: float, optional. The hue angle of the surround stimulus. Set to 180° by default.
        startAvg: integer, optional. The smallest preferred hue angle of the center unit population. Set to 1° as default.
        endAvg: integer, optional. The biggest preferred hue angle of the center unit population. Set to 360° as default.
        depInt: list, optional. The interval of the modulation depth for different surround hues in the non-uniform model. Given as [depb,depu] where
        depb>=0 and depu<=1. The maximum inhibition rate is for example 0.5 when depu=0.5 for the surround hue angle the same as the phase angle.
        depmod: boolean, optional. When True, the surround modulation depth is non-uniform, and it is otherwise uniform.
        stdtransform: boolean, optional. If True, stdInt should be given in standard deviation angle, otherwise stdInt is given as kappa. Note that when
        stdtransform==False, stdInt should be given as [ku,kb] where ku is the biggest kappa value and kb is the smallest kappa value
        KsurInt: list, optional. If a list value is given, then surround kappa is also phase modulated, meaning Ksur is irrelevant. Default is None. Give as [ku,kb]
        ksurphase: integer, optional. Default is 0. If a number is specified, the phase of surround modulation kappa  is changed by the given value in degrees.
        kcentphase: integer, optional. Default is None. If a number is specified, the phase of center unit kappa is changed by the given value in degrees. If the value
        is None (as in default), then the center unit phase modulation is via the variable "phase" (see above).
    
        Returns
        -------
        All returns can be called by colmod."the_object_name" . The return values are as follows:
        
        x: The array of center stimulus hue angles in degrees.
        totalAvg: The list of preferred hue angles of the center units.
        centery: The list of tuning curve activities of the center units without surround modulation. Each list element is an array of activity
        values of the center unit tuning curve. 
        resulty: The list of tuning curve activities of the center units after surround modulation. Each list element is an array of activity
        values of the center unit tuning curve.
        unitTracker: The preferred hue angle list of the center units. This list is used for indexing in centery/resulty. 
        For example centery[np.where(unitTracker==100)[0][0]] gives the centery value of the unit with the preferred hue angle of 100°.
        surroundy: The surround modulation values on center units. These values are aligned with x (see above), in this case x is the preferred
        hue angle of the unit under suppression and surroundy gives the modulation rate.
        errorRate: Dictionary of the error values of the centery values. Each element in centery should have a sum of 1, this dictionary gives the
        possible deviations.
        """
        self.totalAvg=[]
        self.centery=[]
        self.resulty=[]
        if bwType=="regular":#uniform model
            for i in range(0,endAvg):
                avg=startAvg+i#in each iteration the preferred hue angle of the unit increases
                self.totalAvg.append(avg)#totalAvg to track the values.
                y=1/(2*np.pi)*np.e**(Kcent*np.cos(np.deg2rad(colmod.x)-np.deg2rad(avg)))#The von Mises distribution, the activity is rescaled with sum, that total area is 1
                self.centery.append(y/sum(y[np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]]))#normalize the curve by dividing the sum from x=0 to x=359.9 so that total area=1
            self.unitTracker=np.arange(startAvg,startAvg+len(self.totalAvg))#The index list for all other lists in the class.
        
        if bwType=="gradient/max":#non-uniform model maximum activity normalized
            if stdtransform==False:#When false, stdInt is treated as Kappa values.
                kappaDown=stdInt[0]#!stdInt[0] is the bigger kappa and stdInt[1] is smaller kappa
                kappaUp=stdInt[1]
            else:
                kappaDown=std2kappa(stdInt[0],1,1.5)[0]#highest kappa value of the lowest std value
                kappaUp=std2kappa(stdInt[1],1,1.5)[0]#lowest kappa value of the highest std value
            if kcentphase==None:
                kapMod=(kappaDown-kappaUp)/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,360)-phase))+kappaUp+(kappaDown-kappaUp)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py
            elif kcentphase!=None:
                kapMod=(kappaDown-kappaUp)/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,360)-kcentphase))+kappaUp+(kappaDown-kappaUp)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py
            for i in range(0,endAvg):#Create the unitTracker
                avg=startAvg+i
                self.totalAvg.append(avg)
                y=1/(2*np.pi)*np.e**(kapMod[i]*np.cos(np.deg2rad(colmod.x)-np.deg2rad(avg)))
                self.centery.append(y/max(y))#Normalize the tuning curves by their maximum activity
            self.unitTracker=np.arange(startAvg,startAvg+len(self.totalAvg))
        
        if bwType=="gradient/sum":#non-uniform model total activity normalized
            if stdtransform==False:#When false, stdInt is treated as Kappa values.
                kappaDown=stdInt[0]#!stdInt[0] is the bigger kappa and stdInt[1] is smaller kappa
                kappaUp=stdInt[1]
            else:
                kappaDown=std2kappa(stdInt[0],1,1.5)[0]#here kappa down is the bigger kappa
                kappaUp=std2kappa(stdInt[1],1,1.5)[0]#other way around, as std up is kappa down
            if kcentphase==None:
                kapMod=(kappaDown-kappaUp)/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,360)-phase))+kappaUp+(kappaDown-kappaUp)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py
            elif kcentphase!=None:
                kapMod=(kappaDown-kappaUp)/2*np.cos(2*np.deg2rad(np.linspace(startAvg,endAvg,360)-kcentphase))+kappaUp+(kappaDown-kappaUp)/2#Kappa Modulator, see also depth_modulator() in supplementary_functions.py
            for i in range(0,endAvg):
                avg=startAvg+i
                self.totalAvg.append(avg)
                y=1/(2*np.pi)*np.e**(kapMod[i]*np.cos(np.deg2rad(colmod.x)-np.deg2rad(avg)))
                self.centery.append(y/sum(y[np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]]))#Normalize the units by their total activity.
            self.unitTracker=np.arange(startAvg,startAvg+len(self.totalAvg))
        
        
        if KsurInt[0]!=None:#this one is for kappa surround, look at the first entry of KsurInt if that is none
            ksurDown=KsurInt[0]#big kappa limit
            ksurUp=KsurInt[1]#small kappa limit
            kSurMod=(ksurDown-ksurUp)/2*np.cos(2*np.deg2rad(np.linspace(0,359.5,720)-(ksurphase)))+ksurUp+(ksurDown-ksurUp)/2#Kappa Modulator
            kapval=kSurMod[np.where(np.linspace(0,359.5,720)==avgSur)[0][0]]
            self.surroundy=1/(2*np.pi)*np.e**(kapval*np.cos(np.deg2rad(colmod.x)-np.deg2rad(avgSur)))#Surround modulation curve, von Mises distributed.
        if KsurInt[0]==None:
            self.surroundy=1/(2*np.pi)*np.e**(Ksur*np.cos(np.deg2rad(colmod.x)-np.deg2rad(avgSur)))#Surround modulation curve, von Mises distributed.
        
        if depmod==False:#Surround suppression is the same for all surround stimulus conditions
            self.surroundy=self.surroundy/max(self.surroundy)*maxInhRate#first divide by max value to get a 1 in peak, then multiply with max inh. rate to get the desired maximum inhibition.     
        if depmod==True: #if surround modulation depth is gradient, then the corresponding modulation value is taken via function depth_modulator.
            self.surroundy=self.surroundy/max(self.surroundy)*depth_modulator(depInt,avgSur,phase)#first divide by max value to get a 1 in peak, then multiply with max inh. rate to get the desired maximum inhibition.
            #!HERE AS WELL: below modulatory depth is not manipulated, only the max modulatory depth, below mod. depth interaction between kSur and upper modulatory depth limit of the given surround
            """
            Supplementary figure 5:
            colMod1=col.colmod(1,0.7,0,[1.5,0.5],bwType="gradient/sum",phase=0,avgSur=0,depInt=[0,0.5],depmod=True,stdtransform=False)
            colMod2=colMod=col.colmod(1,0.1,0,[1.5,0.5],bwType="gradient/sum",phase=0,avgSur=0,depInt=[0,0.5],depmod=True,stdtransform=False)
            plt.plot(colMod1.surroundy)
            plt.figure()
            plt.plot(colMod2.surroundy)
            """

        for i in range(0,len(self.unitTracker)):#modulate the center tuning curves.
            ymodulated=self.centery[i]*(1-self.surroundy[np.where(colmod.x==self.unitTracker[i])[0][0]])
            self.resulty.append(ymodulated) 
        self.errorRate={}#error of the center units before surround modulation, the dictionary includes the center unit sum deviation from 1. 
        for i in range(0,len(self.centery)):
                if 1-sum(self.centery[i][np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]])!=0:#taking the units whose curve area is not exactly 1
                    self.errorRate.update({self.unitTracker[i]:1-sum(self.centery[i][np.where(colmod.x==0)[0][0]:np.where(colmod.x==360)[0][0]])})#list of unit ID and error rate

class decoder:
    """The decoder class:
        This class includes all functions and output elements of the decoders used. Decoders are population vector (called vecsum), maximum likelihood (ml),
        maximum fire rate (maxfr) and the von Mises fit (vmFit). Additionally there is also a class to simply return the model elements without surround
        modulation. The decoder subclass inputs are the specific outputs of the colmod.
    """
    class nosurround:
        """Return the population activity with/out surround modulation
        """
        def __init__(self,stimulusAngle,x,act):
            """Parameters
            -------------
            stimulusAngle: float. The hue angle of the center stimulus.
            x: array. The colmod.x variable should be given here
            act: array. The activity of interest, can be population activity without surround modulation (colmod.centery) 
            or population activity with surround modulation (colmod.resulty)
            
            Returns
            -------
            noSur: list. The population activity for the given center stimulus hue angle without surround suppression.
            """
            self.noSur=[]
            for i in range(0,len(act)):
                self.noSur.append(act[i][np.where(x==stimulusAngle)[0][0]])
    
    class vecsum:
        """The population vector decoder (Georgopoulos et al. 1986):
            Here, each unit is transformed into a vector, the vector angle is the preferred hue angle of the unit, the vector length is the
            activity of the unit. Then each unit is addded together and the angle of the resulting vector is the decoded hue angle. The addition
            and retrieval of the resulting vector angle is done in complex plane due to its simplicity.
        """
        def __init__(self,x,resulty,unitTracker,avgSur=180,errNorm=False,centery=None,dataFit=False):
            """Parameters
            -------------
            x: array. The colmod.x variable should be given here.
            resulty: list. The colmod.resulty variable should be given here.
            unitTracker: list. The colmod.unitTracker variable should be given here.
            avgSur: float, optional. The surround hue angle in degrees. 180° by default.
            errNorm: boolean, optional. If True, the vectors of the tuning curve (in maximum value) are multiplied by a value, so that each
            vector is within a circle with the x axis radius of the population vector ellipsis. Note that the error correction only dampens the
            magnitude of the error relative to the uncorrected population activity.
            centery: list. The colmod.centery variable should be given here, when errNorm is True. Else this variable can be ignored.
            dataFit: boolean, optional. If True, only the measured center hue angles (0,45,90,...,315) in the psychophysics data are computed by
            the decoder. This parameter is to reduce computational workload during data fit scan.


            Returns
            -------
            centSurDif: list. The center-surround stimulus hue angle difference in degrees.
            angShift: list. The induced hue shift in degrees. This list is aligned with the centSurDif list, so that the angShift[0] value
            is at centSurDif[0] value.
            SurDecoder: list. The list of population activities for different center stimuli. Note that in each case the avgSur is constant.
            Here, the colmod.x can be used to track each of the center stimulus hue angle (each array element in the list), and 
            colmod.unitTracker is used for each of the center stimulus array to track the activity of each unit. 
            E.g. surDecoder[np.where(colmod.unitTracker==100)[0][0]][np.where(colmod.unitTracker==10)[0][0]] gives the unit activity with preferred hue angle of 10° at center stimulus
            hue angle of 100°. In other words, each element in SurDecoder is for a center stimulus hue, each value inside this array element is for a unit
            with a definite preferred hue. Both angles are tracked with unitTracker index list.
            popSurVec: list. The vectorized version of SurDecoder.
            """
            self.centSurDif=[]
            self.angShift=[]
            self.popSurVec=[]
            self.surDecoder=[]
           
            if dataFit==True:#If model fit to the data is to be done, only the measured angles are taken into consideration
                datAng=avgSur+np.linspace(-180,157.5,16)#transform the data center-surround angle difference into absolute angular values
                datAng[np.where(datAng>360)]=datAng[np.where(datAng>360)]-360#Ensure there is no angle bigger than 360°
                datAng[np.where(datAng<0)]=datAng[np.where(datAng<0)]+360#Ensure there is no angle smaller than 0°

            if errNorm==True:
                popNoSurVecx=[]#The tuning curve maximum activity vector x values without surround modulation.
                for i in range(0,len(unitTracker)):
                    popNoSur=decoder.nosurround(unitTracker[i],x,centery).noSur[i]#the unit activity without surround
                                                                                  #for center stimulus=preferred hue angle
                    popNoSurVecx.append(popNoSur*np.e**(1j*np.deg2rad(unitTracker[i])).real)
                circRad=max(popNoSurVecx)-min(popNoSurVecx)#The radius of the circle for the vector population error correction
                normVal=np.array(circRad)/np.array(popNoSurVecx)
            def vector_decoder(stimulusAngle,unitStep=1):#vector sum decoding (used in Gilbert et al. 1990).
                surDecoder=[]
                for i in range(0,len(resulty),unitStep):
                    surDecoder.append(resulty[i][np.where(x==stimulusAngle)[0][0]])#create the list of population activity for each center stimulus condition.
                popSurVector=[]#Make the surDecoder values vector in complex plane.
                for i in range(0,len(surDecoder)):
                    popSurVector.append(surDecoder[i]*np.e**(1j*np.deg2rad(unitTracker[i*unitStep])))#each unit is transformed into vectors in complex plain with vector length equal to
                    #unit activity and angle equal to preferred unit angle (in form of z=r*e**(i*angle)), !Computations for angle are done here in radiants  
                if errNorm==True:
                    popSurVector=popSurVector*normVal
                np.sum(popSurVector)#vector sum of neural population
                decodedAngleSur=np.rad2deg(c.phase(np.sum(popSurVector)))#Take the angle of the resulting vector with c.phase()
                if decodedAngleSur<0:
                    decodedAngleSur=360+decodedAngleSur #transform the negative angles into angles>180.
                
                if avgSur!=180:#For the surround stimulus different than 180°, reference point on the angle cirle is shifted so, that the surround angle 
                               #in the middle. This is important to transform the center surround angle hue difference, that center hue angles within the
                               #interval [surround angle,surround angle+180] result in positive center-surround difference, other center hue angles on the
                               #other hand cause the center-surround difference to be negative.
                    if avgSur<180:#The transformation is done by upper limit if average surround<180, so that center hue angles>surround+180 are expressed
                                  #as negative angles with the correct value.
                        upperLim=avgSur+180
                        if stimulusAngle>upperLim:
                            stimulusAngle=stimulusAngle-360#center hue angle is negative if bigger than surround+180, this is for center-surround difference
                            decodedAngleSur=decodedAngleSur-360#decoded angle is also expressed as 1 cycle less value, this is for induced hue shift. For example,
                                                               #for surround=0 and center hue=270°, center hue is taken as -90°, and the decoded hue angle 265° is taken
                                                               #as -95. Resultingly, the hue shift is decoded-center hue angle and is negative.
                    if avgSur>180:#Similar transformation for when surround>180°. In this case it works with below limit. For example, surround=200° means center hue of 0°
                                  #has the value of 360, so that the center-surround difference is -160°
                        belowLim=avgSur-180
                        if stimulusAngle<belowLim:
                            stimulusAngle=stimulusAngle+360#Change the center hue angle value smaller than the lower limit 1 cycle bigger (+360°)
                            decodedAngleSur=decodedAngleSur+360#Same transformation as above also to the decoded angle so that the hue shift stays the same.
                if decodedAngleSur-stimulusAngle>180:#If induced hue shift is inferred to be bigger than 180°, same angle is given in negative counterpart.
                    decodedAngleSur=decodedAngleSur-360#transform nonsensical values (so far very lame way!)
                if decodedAngleSur-stimulusAngle<-180:#Same as above, if hue shift is smaller than -180°, hue shift should is given as positive value.
                    decodedAngleSur=decodedAngleSur+360
                return decodedAngleSur-stimulusAngle , stimulusAngle-avgSur,surDecoder,popSurVector #returns the induced hue shift and center-surround hue angle difference

            if dataFit==True:#If True, then only angles measured in psychophysics experiments are given as inputs to the decoder function.
                for i in range(0,len(datAng)):#i here is the stimulusAngle index 
                    angs,csd,surdec,popsurv = vector_decoder(stimulusAngle=datAng[i])
                    self.angShift.append(angs)
                    self.centSurDif.append(csd)
                    self.surDecoder.append(surdec)
                    self.popSurVec.append(popsurv)
                self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!

            else:
                for i in range(0,len(unitTracker)):#Form the list of decoder outputs for the center hue angles 0°-360°
                    angs,csd,surdec,popsurv = vector_decoder(stimulusAngle=unitTracker[i])#angs is the angular shift, csd is center-surround difference,surdec is surround decoder, 
                                                                                          #popsurv is the population vector. As cen ter stimuli, the preferred hue angles of the filters
                                                                                          #are given (1°-...-360°)
                    self.angShift.append(angs)
                    self.centSurDif.append(csd)
                    self.surDecoder.append(surdec)
                    self.popSurVec.append(popsurv)
                self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!
                    
    class ml:
        """The maximum likelihood decoder:
            This decoder assumes that the filter population is "unaware" if there is a surround modulation happening or not. Resultingly, the idea is that,
            the decoded angle with surround modulation should be the same as the angle encoded by the population activity (without surround modulation) which
            has the most similar activity profile as the population activity with surround (Kellner manuscript, Pouget et al 2000). For that, a look-up table is 
            created for the population activity without surround for a various center stimuli. Then, by using the root mean square difference between the population
            activity with surround modulation and the look-up table entries, the most similar population activity in the look-up table (least root mean square difference)
            is found. The decoded angle is the encoded angle by the chosen look up table entry.
        """
        def __init__(self,x,centery,resulty,unitTracker,avgSur=180,tabStep=5,norm=False,dataFit=False):#tabStep minimum 1 where stimulus angle binning 0.1 ist 
            """Parameters
            -------------
            x: array. The colmod.x variable should be given here. This parameter is used for center stimulus hue angle.
            centery: list. colmod.centery variable should be given. The center unit tuning curves without surround modulation.
            resulty: list. colmod.resulty variable should be given. The center unit tuning curves after surround modulation.
            unitTracker: list. colmod.unitTracker variable should be given. The index list for tracking the center units.
            avgSur: float, optional. The surround stimulus hue angle in degrees. Set to 180° as default.
            tabStep: integer, optional. The binning of the hue angle for creating the look-up table. tabStep=1 means center hues are binned
            0.1° (look-up table entries 0,0.1,0.2,...359.9) and for tabStep=10 is the binning 1°. In other words, this value determines how many
            colmod.x elements are to be skipped (in case of tabStep=10 colmod.x elements with indices of 0,10,20 are chosen etc.)
            norm: boolean, optional. If True, the look-up table entries are  normalized so that each entry has a sum of 1.
            dataFit: boolean, optional. If True, only the measured center hue angles (0,45,90,...,315) in the psychophysics data are computed by
            the decoder. This parameter is to reduce computational workload during data fit scan.
            
            Returns
            -------
            centSurDif: list. The center-surround stimulus hue angle difference in degrees.
            angShift: list. The induced hue shift in degrees. This list is aligned with the centSurDif list, so that the angShift[0] value
            is at centSurDif[0] value.
            MLindex: list. The root mean square difference between each center stimulus with surround modulation and each look-up table entry. Note
            that surround stimulus is constant. MLindex[np.where(colmod.x==100)[0][0]/tabStep] (for norm=False) gives the complete RMS difference between each entry and
            the population activity for center=100° with surround, the encoded angle by the population activity with the minimum RMS value is then the 
            decoded angle by the ML-decoder. Again, colmod.x is used for finding the decoded angle value (see code comments for a more detailed explanation).
            surDecoder: list. The population activity after surround modulation. Note that the indexing of this list depends on the tabStep. The
            first hue angle is always 0°, the last one is <360° and its exact value depends on the tabStep 
            (359.9° for tabStep=1, 359° for tabStep=10 and 359.5° for tabStep=5). colmod.x to track the center stimulus hue angle, colmod.unitTracker to
            track the activity of each chromatic filter unit.
            """
            """
            Create the empty lists of the return variables.
            """
            self.angShift=[]
            self.centSurDif=[]
            self.MLindex=[]
            self.surDecoder=[]

            def tab_look(stimulusAngle):#lookup-table generator function.
                tableML={}
                tableInd=[]
            #stimulusAngle=100
                for i in range(1,len(unitTracker)+1):
                    tableInd.append(centery[np.where(unitTracker==i)[0][0]][np.where(x==stimulusAngle)[0][0]])#create the lookup-table entry by taking the population 
                                                                                                              #activity in the given center stimulus angle without 
                                                                                                              #surround modulation.
                tableML.update({stimulusAngle:tableInd})#values here are unit activities in given stimulus for each unit 
                return tableML
            
            if dataFit==True:#If model fit to the data is to be done, only the measured angles are taken into consideration
                datAng=avgSur+np.linspace(-180,157.5,16)#transform the data center-surround angle difference into absolute angular values
                datAng[np.where(datAng>360)]=datAng[np.where(datAng>360)]-360#Ensure there is no angle bigger than 360°
                datAng[np.where(datAng<0)]=datAng[np.where(datAng<0)]+360#Ensure there is no angle smaller than 360°
            tabLook={}
            for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],tabStep):#i here is the stimulusAngle index
                tabLook.update(tab_look(x[i]))#This loop creates the whole look-up table for all center stimulus hues.
            
            def ML_decoder(stimulusAngle,normalizer):#Decoder function
                popInh=[]
                #stimulusAngle=130
                for i in range(1,len(unitTracker)+1):
                    popInh.append(resulty[np.where(unitTracker==i)[0][0]][np.where(x==stimulusAngle)[0][0]])#Population activity after surround
                distLUT=[]
                
                if normalizer==False:#Do not normalize the look-up table entry.
                    for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],tabStep):
                        inhVal=np.asarray(tabLook[x[i]])-np.asarray(popInh)
                        distLUT.append(np.sqrt(np.mean(np.asarray(inhVal)**2)))
                else:#Normalize the look-up table entry before taking the RMS value.
                    for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],tabStep):
                        inhVal=np.asarray(tabLook[x[i]])/sum(tabLook[x[i]])-np.asarray(popInh)/sum(popInh)
                        distLUT.append(np.sqrt(np.mean(np.asarray(inhVal)**2)))#both areas of activity before modulation and after modulation is normalized to 1 
                #plt.plot(distLUT)
                
                decodedAng=x[range(np.where(x==0)[0][0],np.where(x==360)[0][0],tabStep)][np.where(np.asarray(distLUT)==min(distLUT))[0]]
                #Decoded angle is the smallest RMS value in the look-up table, x is here limited to 0 and 360 degrees.
                if len(decodedAng)>1:#If the decoded angle is ambigious (more than 1 minimum value), then the model paramters are not realistic,
                                     #so the model is not further considered and return values are nonsensical.
                    print("stimAng=%s,decAng=%s"%(stimulusAngle,decodedAng))
                    print("decoded angle is ambigious, model possibly has irrealistic parameter values")
                    #raise Exception("The decoded angle is ambigious, model possibly has irrealistic parameter values")
                    return [99999],9999999999,999999999,9999999999 #to make data fit non functional in any case
                if avgSur!=180:#For the surround stimulus different than 180°, reference point on the angle cirle is shifted so, that the surround angle 
                               #in the middle. This is important to transform the center surround angle hue difference, that center hue angles within the
                               #interval [surround angle,surround angle+180] result in positive center-surround difference, other center hue angles on the
                               #other hand cause the center-surround difference to be negative.
                               #SEE also the same codes in vecsum.
                    if avgSur<180:
                        upperLim=avgSur+180
                        if stimulusAngle>upperLim:
                            stimulusAngle=stimulusAngle-360
                            decodedAng=decodedAng-360
                    if avgSur>180:
                        belowLim=avgSur-180
                        if stimulusAngle<belowLim:
                            stimulusAngle=stimulusAngle+360
                            decodedAng=decodedAng+360                
                if decodedAng-stimulusAngle>180:
                    decodedAng=decodedAng-360# transform nonsensical values (so far very lame way!)
                if decodedAng-stimulusAngle<-180:
                    decodedAng=decodedAng+360
                return decodedAng-stimulusAngle, stimulusAngle-avgSur, distLUT, popInh
            
            if dataFit==True:#If True, then only angles measured in psychophysics experiments are given as inputs to the decoder function.
              for i in range(0,len(datAng)):#i here is the stimulusAngle index 
                    angShift,csd,MLindex,surDec=ML_decoder(datAng[i],norm)
                    self.angShift.append(angShift[0])
                    self.centSurDif.append(csd)
                    self.MLindex.append(MLindex)
                    self.surDecoder.append(surDec)
              self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!
  
            else:
                for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],tabStep):
                    angShift,csd,MLindex,surDec=ML_decoder(x[i],norm)
                    self.angShift.append(angShift)
                    self.centSurDif.append(csd)
                    self.MLindex.append(MLindex)
                    self.surDecoder.append(surDec)
                self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!
    
    class maxfr:
        """Maximum fire rate decoder:
            In this decoder, the decoded hue angle is the preferred hue angle of the unit with maximum activity among the population. 
        """
        def __init__(self,x,resulty,unitTracker,avgSur=180):
            """Parameters
            -------------
             x: array. The colmod.x variable should be given here. This parameter is used for center stimulus hue angle.
            resulty: list. colmod.resulty variable should be given. The center unit tuning curves after surround modulation.
            unitTracker: list. colmod.unitTracker variable should be given. The index list for tracking the center units.
            avgSur: float, optional. The surround stimulus hue angle in degrees. Set to 180° as default.
            
            Returns
            -------
            centSurDif: list. The center-surround stimulus hue angle difference in degrees.
            angShift: list. The induced hue shift in degrees. This list is aligned with the centSurDif list, so that the angShift[0] value
            is at centSurDif[0] value.
            """
            self.centSurDif=[]
            self.angShift=[]
            def max_fr(stimulusAngle, unitStep=1, avgSur=avgSur):
               surDecoder=[]#Population activity in the given stimulus angle.
               for i in range(0,len(resulty),unitStep):
                   surDecoder.append(resulty[i][np.where(x==stimulusAngle)[0][0]])
               
               decodedAng=unitTracker[np.where(surDecoder==max(surDecoder))[0][0]]#Preferred hue of the maximum active unit is the readout.
                
               if avgSur!=180:#Same transformation as in other decoders (see vecsum)
                   if avgSur<180:
                       upperLim=avgSur+180
                       if stimulusAngle>upperLim:
                           stimulusAngle=stimulusAngle-360
                           decodedAng=decodedAng-360
                   if avgSur>180:
                       belowLim=avgSur-180
                       if stimulusAngle<belowLim:
                           stimulusAngle=stimulusAngle+360
                           decodedAng=decodedAng+360
               if decodedAng-stimulusAngle>180:
                   decodedAng=decodedAng-360# transform nonsensical values (so far very lame way!)
               if decodedAng-stimulusAngle<-180:
                   decodedAng=decodedAng+360
               return decodedAng-stimulusAngle , stimulusAngle-avgSur
            
            for i in range(0,len(unitTracker)):#Create the list of decoder output values.
                angs,csd=max_fr(unitTracker[i])
                self.angShift.append(angs) 
                self.centSurDif.append(csd)
            self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!
 
    class vmfit:
        """von Mises fit decoder:
            In this decoder, the population activity is fitted by a von Mises distribution, the decoded angle is the angular location of the 
            global maximum of the fit. Levenberg-Mardquadt algorithm is used for the fit.
        """
        def __init__(self,x,resulty,unitTracker,avgSur=180):
            """Parameters
            -------------
            x: array. The colmod.x variable should be given here.
            resulty: list. The colmod.resulty variable should be given here.
            unitTracker: list. The colmod.unitTracker variable should be given here.
            avgSur: float, optional. The surround hue angle in degrees. 180° by default.
            
            Returns
            -------
            centSurDif: list. The center-surround stimulus hue angle difference in degrees.
            angShift: list. The induced hue shift in degrees. This list is aligned with the centSurDif list, so that the angShift[0] value
            is at centSurDif[0] value.
            popFit: list. The von Mises fit to the given population activity. This list includes fits for center hue angles starting from 1° to 359° with 1° binning,
            each center stimulus fit array then contains the von Mises values for the center unit preferred hue stimuli (1° to 360° with 1° binning.)
            surDecoder: list. The population activity for the given center hue after surround modulation. Similar to the variable in class vecsum.
            parameters: list. The 3 parameters of the von Mises fit. These parameters are the curve amplitude (amp), concentration parameter (kappa) and global maximum
            of the distribution (avg). The list is returned as 3 independent sublists for amp,kappa,avg for all fits, where indices of all these lists are aligned together,
            so that parameters[0][10] and parameters[1][10] are the amp and kappa of the same fit. 
            """
            self.angShift=[]
            self.centSurDif=[]
            self.popFit=[]
            self.surDecoder=[]
            self.parameters=[]
            def fit_func(ang,amp,kappa,avg):#Fit function, returns the von Mises distributio for given angle (array), amp, kappa and avg values (last parameters are float).
                return amp/(2*np.pi)*np.e**(kappa*np.cos(np.deg2rad(ang)-np.deg2rad(avg)))
            
            def von_mises_fit(stimulusAng,avgSur=avgSur):#Decoder function
                #stimulusAng=180
                surDec=[]
                for i in range(0,len(resulty)):
                    surDec.append(resulty[i][np.where(x==stimulusAng)[0][0]])#Population activity after surround modulation for the given center hue.
                xx=x[np.where(x==0)[0][0]:np.where(x==360)[0][0]]    
                popt, pcov = fit(fit_func, unitTracker, surDec,method='lm')#Levenberg-Marquardt algorithm using the von Mises distribution as template (fit_func) and 
                                                                           # where x value is given by unitTracker and fit is done on surDec.
                popFit=fit_func(xx,*popt)#Create the fit values as an array.
                if popt[2]>360:#If the global maximum of the fit is predicted to be bigger than 360°, then it is transformed by subtracting 360°
                    popt[2]=popt[2]-360    
                if popt[2]<-360:#If the global maximum of the fit is predicted to be smaller than -360°, then it is transformed by adding 360°
                    popt[2]=popt[2]+360
                if popt[1]<0:#If the kappa of the fit is inferred as negative, it is transformed to its absolute value and global maximum is shifted by 180°
                    popt[1]=abs(popt[1])
                    popt[2]=popt[2]+180
                if popt[2]<-0.1 and popt[1]>0:
                    popt[2]=360+popt[2]
#THESE IF STATEMENTS are to solve the irregular parameters problem: due to the fact: cos(x)=cos(-x) and -a*cos(x)=acos(x+180), if kappa is negative it is changed
#to its positive value before adding +180 to the readout angle. Similarly, if kappa is positive but angle is negative, 360 degrees are added (in that part threshold
#is given as -0.1 because in the beginning (stimulus angle=0) there is a small error that decoded angle is in -0.000001 range.)
                
                if avgSur!=180:#Same transformation, see vecsum.
                   if avgSur<180:
                       upperLim=avgSur+180
                       if stimulusAng>upperLim:
                           stimulusAng=stimulusAng-360
                           popt[2]=popt[2]-360
                   if avgSur>180:
                       belowLim=avgSur-180
                       if stimulusAng<belowLim:
                           stimulusAng=stimulusAng+360
                           popt[2]=popt[2]+360
                           
                if popt[2]-stimulusAng>180:
                   popt[2]=popt[2]-360# transform nonsensical values (so far very lame way!)
                if popt[2]-stimulusAng<-180:
                   popt[2]=popt[2]+360
                return popt[2]-stimulusAng, stimulusAng-avgSur, popFit,surDec, popt
            

            for i in range(np.where(x==0)[0][0],np.where(x==360)[0][0],10):#Create the decoder outputs as lists for different center hue angles.
                allvals1=von_mises_fit(x[i])
                self.angShift.append(allvals1[0])
                self.centSurDif.append(allvals1[1])
            
            for i in range(0,len(resulty)):
                allvals2=von_mises_fit(i)
                self.popFit.append(allvals2[2])
                self.surDecoder.append(allvals2[3])
                self.parameters.append(allvals2[4])
            
            self.centSurDif, self.angShift = zip(*sorted(zip(self.centSurDif,self.angShift)))#sort the values from centsurdif=-180 to +180!
            self.parameters=list(map(list,zip(*self.parameters)))#Transpose the list, so that each sublist corresponds to each parameter (amp, kappa, avg)


"""
Effect of surround kappa phase modulation
"""
class figures():
    def surphase():
        surs=[135,90,45,180,0,225,270,315]
        cols=["red","green","brown","yellow","blue"]
        fig=plt.figure(1)
        fig2=plt.figure(2)
        ax2=fig2.gca()
        for j in surs:
            ax=plotter.subplotter(fig,surs.index(j))
            for i in np.linspace(0,90,5):
                print(i,j)
                colm=colmod(Kcent=None,Ksur=None,maxInhRate=None,stdInt=[2,1.8],bwType="gradient/sum",phase=22.5,avgSur=j,depInt=[0.4,0.6],stdtransform=False,depmod=True,KsurInt=[2.5,0.5],ksurphase=i)
                ax2.plot(colm.x,colm.surroundy,label=j)
                maxl=decoder.ml(colm.x,colm.centery,colm.resulty,colm.unitTracker,tabStep=5,avgSur=j)
                ax.plot(maxl.centSurDif,maxl.angShift,label=i+22.5,color=cols[int(i/22.5)])
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
        ax.legend(loc="best",bbox_to_anchor=(1,1))
        ax2.legend()
        return
    
    """
    Effect of center kappa phase modulation. To bypass the surround modulation effect in the non uniform model, the surround modulation is kept constant over all
    surround hues by giving the min and max values the same.
    """
    def centphase():
        surs=[135,90,45,180,0,225,270,315]
        cols=["red","green","brown","yellow","blue"]
        fig=plt.figure(3)
        fig2=plt.figure(4)
        ax2=fig2.gca()
        for j in surs:
            ax=plotter.subplotter(fig,surs.index(j))
            for i in np.linspace(0,90,5):
                print(i,j)
                colm=colmod(Kcent=None,Ksur=2.3,maxInhRate=None,stdInt=[2,1.8],bwType="gradient/sum",phase=i,avgSur=j,depInt=[0.4,0.4],stdtransform=False,depmod=True)
                maxl=decoder.ml(colm.x,colm.centery,colm.resulty,colm.unitTracker,tabStep=5,avgSur=j)
                ax2.plot(colm.x,colm.surroundy)
                ax.plot(maxl.centSurDif,maxl.angShift,label=i,color=cols[int(i/22.5)])
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
        ax.legend(loc="best",bbox_to_anchor=(1,1))
        return


"""
Try if the different phase variable work correct by plotting each setting
fig1=plt.figure(1)
fig2=plt.figure(2)

for i in np.linspace(0,180,9):
    a=colmod(1,2,1,stdInt=[2,1],bwType="gradient/sum",phase=0,depmod=True,stdtransform=False,KsurInt=[3,0.5],kcentphase=90,ksurphase=45,avgSur=i)
    for j in range(0,18):#cneter independent of what the surround hue angle is
        ax1=fig1.gca()
        ax1.plot(a.x,a.centery[j*10])
    ax2=fig2.gca()
    ax2.plot(a.x,a.surroundy)
#All seems fine, ready for the scan.
"""
  
"""
If necessary, below is plot of surround suppression curves as a function of the surround hue
"""
"""
plt.figure()
for i in np.linspace(0,315,15):
    colMod=colmod(1,2.3,1,stdInt=[1.2,0.9],bwType="gradient/sum",phase=22.5,depInt=[0.2,0.3999999999999999],depmod=True,stdtransform=False,KsurInt=[2.5,0.5],avgSur=i,ksurphase=90)
    plt.plot(colMod.x,colMod.surroundy,color="black")
"""   
    
"""
Thesis figure supplement 2
Also in thesis_figures.py
"""
"""
a=colmod(1.5,1,0.5,[1,10],bwType="regular")
fig=plt.figure()
plt.title("Center unit tuning curves and surround modulation in the uniform model")
plt.xticks([])
plt.yticks([])
plt.box(False)
ax1=fig.add_subplot(1,3,1)
for i in range(45,len(a.centery)+1,45):#range of 90-270 also possible
    print(i)
    ax1.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1.0)
ax1.set_xlabel('Hue angle of center stimulus')
ax1.set_xticks(np.linspace(0,360,9))
ax1.set_ylabel('Neuronal activity (arbitrary units)')

ax2=fig.add_subplot(1,3,2)
ax2.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],1-a.surroundy[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]])
ax2.set_xlabel('Preferred hue angle of center filter')
ax2.set_xticks(np.linspace(0,360,9))
ax2.set_yticks(np.linspace(0.4,1,7))
ax2.set_ylabel('Modulation rate')

ax3=fig.add_subplot(1,3,3)
for i in range(45,len(a.centery)+1,45):
    ax3.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.centery[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="gray",linewidth=0.5,label="before surround modulation")
    ax3.plot(a.x[np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],a.resulty[i-1][np.where(a.x==0)[0][0]:np.where(a.x==360)[0][0]],color="black",linewidth=1,label="after surround modulation")
ax3.set_xlabel('Hue angle of center stimulus')
ax3.set_xticks(np.linspace(0,360,9))
ax3.set_ylabel('Neuronal activity (arbitrary units)')
ax3.legend(["before surround modulation","after surround modulation"],loc="best", bbox_to_anchor=(1,1))
"""

"""
*Development notes!
vmFit amplitude Problem
colMod=colmod(1,1.5,0.5,[60,80],bwType="regular")
dec=decoder.vmfit(colMod.x,colMod.resulty,colMod.unitTracker)
dec.parameters
plt.plot(colMod.unitTracker,dec.parameters[0])#180 with local maximum, adjacent 2 minima???
plt.plot(colMod.unitTracker,dec.parameters[1])#here 180 has the minimum kappa=> max deviation, meaning rather flat which also affects amp? what is the meaning of amp here???
"""
"""
!!!FEW QUESTIONS:
-In case of vmFit, the inferred apmlitude of population activity has a maximum in 180 degrees (regular BW uniform surdepth and surr=180), and minima in 109 degrees, does this make any sense 
at all? As the peak activity is lowest in 180 degrees. My conclusion: Peak activity mediated both by amp variable and by the width of population activity, 180 degrees has the widest population
activity which eliminates the effect of amp variable.
-In Klauke paper: V1 has dense blue-yellow axis representations but the sensitivity in this axis is low and activity is variable: Why? Shouldn't adding more units increase the sensitivity? SENSITIVITY BEGRENZT DURCH S CONE MENGE+RETINAL and LGN representations
-In general: color coding LGN and RGCs(small midget/parasol) color opponent, also in V1 color opponent (HUBEL&Wiesel mit Vorsicht, die Kategorien sind nicht unbedingt stark erklärt) in CO blobs in L2-3 (with S upsampling leading to axis rotation) but there are broadband, beides eher continuum 
neurons as well, which V1 neuron am i modelling? How is the tuning of V1 color neurons constituted by LGN cells? selective synaptic weight? or different number of LGN cells from parvo-koino
layers to the color selective neuron?
"""
'''
!!!FOR very big modulation depth (>=0.7) the population activity after surround effect gets bimodal, especially when Kcent near Ksur!

TRY TO FIND A RELATIONSHIP BETWEEN KAPPA AND PERCENTAGE OF DATAPOINTS IN INTERVAL [AVG-KAPPA;AVG+KAPPA]!
To get an idea about relationship between kappa and stdev, plot the sd of distribution with 180 deg. mean against kappa (numerical approach to get 1st feeling of relationship)
Done: Linear regression to std and Kappa, when correlation is >0.99 all is fine. Here are but some questions, like e.g std cannot be bigger than 105° even when Kappa=0 and 
distribution is uniform, which is ok as my interval is 0 to 360 and for that interval uniform distribution has std <105! 
In worst case again try the apporaches in the literature (circular data papers/books) , done with supplementary function std2kappa which interpolates.

!choose surround modulation widths (80-140 with some binning) as well as modulation depth (0.2-0.8 binning) in a constant center bandwith (60° pro unit)
and analyse the parameters by scanning through. done in parameter scan script
Learn how to work with gits (windows) for dummies

*Gauss Fit auf population activity (minimize the difference between population activity and gauß function numerically there are python functions)
use von mises to fit, there are automatic tools for that! non linear levenberg marquardt method for example, look for python modules.
(try e.g import scipy.optimize.curve_fit as sp_op#a very nice possibility but i somehow need a function in advance as model!
sp_op.curvefit()) !!!done 

*Maximum fire rate decoder (via recycling codes from vector sum). !!!done

*Learn some git!
*Change the model so, that the tuning width of center units changes gradiently in a given frequency:
    0: 60 BW, 90:70 bw 180: 60 ..., you can apart from that use the surround parameters you got from parameter_scan file. you can use sinus curve for differential center unit gradient (quadrat or rectified!)
    !!!done
*Also investigate the population activity without surround modulation in stimulus angle and compare it with activity with surround modulation!    

*Decoding of population activity of gradient units in different surrounds (0, 45/2, 45 etc as binning!) !the models are ready for different surround modulations!
*Normalizing to maximum against area (literature check, e.g Schwartz, we keep both at the time being if they differ)
-Keep both atm, normalizing to max FR increases the involvement of bigger bandwith neurons, normalizing to area assumes higher BW neurons have lower FR (adaptation, homeostasis) 
!Could it be the a good argument for areal normalization to keep the whole population coding stable? or via hebb that smaller BW neurons are driven stronger by downstream targets and thus higher FR?

*When looking at surround decoder compare with without surround pop activity.

*Make subplots for all different surrAngs for each model type (regular and bw gradient types)
*other possibility: uniform center units but surround modulation depth is different at different surround angles
*simpler alternative for surround modulation axis shift: before any computation add 360 to every angular value (x, stimulusAng, avgSur) and then subtract 360 from all angle related variables!
'''
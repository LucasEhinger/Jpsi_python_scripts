#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm

from pystyle.annotate import placeText

from scipy.optimize import curve_fit
import os

plt.style.use("SRC_CT_presentation")

xmin=0
xmax=5
# histname = "alpha_miss_SB_lower"
histname = "t_kin_minus_ROI"
directoryname=".ROI"

pTCut=""
tracks="noTracks"
version="v7"

Egamma_subt="8p2"
Egamma_thresh="8p2"
rebin_data=50
rebin_sim=10
xoffset=0.01
def getXY(infiles,weights,histname, rebin):
    x=0
    y=0
    yerr=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=directoryname)
        h = f.get(histname, rebin=rebin)
        x = h.x
        y += h.y*weights[i]
        yerr = np.sqrt(yerr**2 +(h.yerr*weights[i])**2)
    return x,y,yerr

# <editor-fold desc="Get Data">
simWeights=[0.366,0.069,1.13,0.29] #Not including 2H (0.242 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{version}/PoverE/m3_p2_sigma/{tracks}/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_lower/"
if pTCut!="":
    filepath+="/pTCuts/"+pTCut+"_lower/"
x_data_subt,y_data_subt,yerr_data_subt=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname,rebin=rebin_data)

x_sim_subt,y_sim_subt,yerr_sim_subt=getXY(infiles=[filepath+tree for tree in simFiles],
                              weights=simWeights,histname=histname,rebin=rebin_sim)

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{version}/PoverE/m3_p2_sigma/{tracks}/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_upper/"
if pTCut!="":
    filepath+="/pTCuts/"+pTCut+"_lower/"
x_data_thresh,y_data_thresh,yerr_data_thresh=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname,rebin=rebin_data)

x_sim_thresh,y_sim_thresh,yerr_sim_thresh=getXY(infiles=[filepath+tree for tree in simFiles],
                              weights=simWeights,histname=histname,rebin=rebin_sim)
# </editor-fold>


# <editor-fold desc="Scaling">
# Scaling
scaleFactor=sum(y_data_thresh+y_data_subt)/sum(y_sim_thresh+y_sim_subt)
scaleFactor_subt=sum(y_data_subt)/sum(y_sim_subt)
scaleFactor_thresh=sum(y_data_thresh)/sum(y_sim_thresh)
scaleFactor_subt=scaleFactor
scaleFactor_thresh=scaleFactor

y_sim_thresh *= scaleFactor_thresh*rebin_data/rebin_sim
yerr_sim_thresh *= scaleFactor_thresh*rebin_data/rebin_sim
y_sim_subt *= scaleFactor_subt*rebin_data/rebin_sim
yerr_sim_subt *= scaleFactor_subt*rebin_data/rebin_sim
# </editor-fold>


# Plotting
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_data_subt,y_data_subt,yerr=yerr_data_subt,fmt='.k',capsize=0,label="Data")

plt.errorbar(x_sim_subt,y_sim_subt,yerr=yerr_sim_subt,fmt='.b',capsize=0,label="Sim.")
plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
if tracks=="noTracks":
    placeText("No Sideband Sub."+"\n"+"Mixed"+"\n"+version+" No Extra Tracks",loc=2,yoffset=-60)
else:
    placeText("No Sideband Sub." + "\n" + "Mixed" + "\n" + version + " Incl. Extra Tracks", loc=2, yoffset=-60)

if pTCut=="":
    placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
else:
    placeText(r"$3<m(e^+e^-)<3.2$"
          +"\n"+r"$p_T<$"+pTCut.replace("p","."),loc=1,yoffset=-45)




plt.subplot(2,1,2)
plt.errorbar(x_data_thresh,y_data_thresh,yerr=yerr_data_thresh,fmt='.k',capsize=0)

plt.errorbar(x_sim_thresh,y_sim_thresh,yerr=yerr_sim_thresh,fmt='.b',capsize=0)

plt.ylabel("Counts")
plt.xlabel(r"$-t$ [GeV]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
placeText(r"$E_{\gamma}>$"+Egamma_thresh.replace("p",".")+" GeV",loc=1)

plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/t/t_ROI_{version}_pT{pTCut}_{tracks}_mixed.pdf")
plt.show()

print(f"Scale difference: {1/scaleFactor_subt: .3f}")
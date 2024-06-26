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
xmax=10
histname = "jpsi_theta"
directoryname=".All.jpsi.ROI"
vers="v8"

rebin_data=5
rebin_sim=2
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
simWeights=[0.242,0.366,0.069,1.13,0.29]
dataFiles=["data_hist2_D.root","data_hist2_He.root", "data_hist2_C.root"]
simFiles=["hist_DSelector_2H_MF_helicity_mixed.root","hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"

x_data_D,y_data_D,yerr_data_D=getXY(infiles=[filepath+tree for tree in [dataFiles[0]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_D,y_sim_D,yerr_sim_D=getXY(infiles=[filepath+tree for tree in [simFiles[0]]],
                              weights=simWeights,histname=histname,rebin=rebin_sim)

x_data_He,y_data_He,yerr_data_He=getXY(infiles=[filepath+tree for tree in [dataFiles[1]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_He,y_sim_He,yerr_sim_He=getXY(infiles=[filepath+tree for tree in simFiles[1:3]],
                              weights=simWeights[3:5],histname=histname,rebin=rebin_sim)

x_data_C,y_data_C,yerr_data_C=getXY(infiles=[filepath+tree for tree in [dataFiles[2]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_C,y_sim_C,yerr_sim_C=getXY(infiles=[filepath+tree for tree in simFiles[3:5]],
                              weights=simWeights[3:5],histname=histname,rebin=rebin_sim)

# </editor-fold>


# <editor-fold desc="Scaling">
# Scaling
scaleFactor=sum(y_data_D+y_data_He+y_data_C)/sum(y_sim_D+y_sim_He+y_sim_C)
scaleFactorD=sum(y_data_D)/sum(y_sim_D)
scaleFactorHe=sum(y_data_He)/sum(y_sim_He)
scaleFactorC=sum(y_data_C)/sum(y_sim_C)


y_sim_D *= scaleFactorD*rebin_data/rebin_sim
yerr_sim_D *= scaleFactorD*rebin_data/rebin_sim
y_sim_He *= scaleFactorHe*rebin_data/rebin_sim
yerr_sim_He *= scaleFactorHe*rebin_data/rebin_sim
y_sim_C *= scaleFactorC*rebin_data/rebin_sim
yerr_sim_C *= scaleFactorC*rebin_data/rebin_sim
# </editor-fold>


# Plotting
plt.figure(figsize=(5,8))
# Subthreshold Plots
plt.subplot(3,1,1)
plt.errorbar(x_data_D,y_data_D,yerr=yerr_data_D,fmt='.k',capsize=0,label="Data")

plt.errorbar(x_sim_D,y_sim_D,yerr=yerr_sim_D,fmt='.b',capsize=0,label="Sim.")

plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText("No Sideband Sub." + "\n" + "Mixed" + "\n" + vers + " Incl. Extra Tracks", loc=2, yoffset=-60)


placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
# placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".") + " GeV",loc=1)

placeText("D",loc=1)


plt.subplot(3,1,2)
plt.errorbar(x_data_He,y_data_He,yerr=yerr_data_He,fmt='.k',capsize=0,label="Data")

plt.errorbar(x_sim_He,y_sim_He,yerr=yerr_sim_He,fmt='.b',capsize=0,label="Sim.")

plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
# placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".") + " GeV",loc=1)
placeText("He",loc=1)


plt.subplot(3,1,3)
plt.errorbar(x_data_C,y_data_C,yerr=yerr_data_C,fmt='.k',capsize=0,label="Data")

plt.errorbar(x_sim_C,y_sim_C,yerr=yerr_sim_C,fmt='.b',capsize=0,label="Sim.")
plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText("C",loc=1)
plt.xlabel(r"$\theta_p$ [Deg]")

# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/alpha_miss/alpha_miss_ROI_fitted_{version}_pT{pTCut}_{tracks}_mixed.pdf")
plt.show()

# print(f"Scale difference: {scaleFactor_subt/scaleFactor_thresh: .3f}")

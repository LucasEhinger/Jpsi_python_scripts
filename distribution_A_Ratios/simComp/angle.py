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
xmax=80
A='D'
# histname = "alpha_miss_SB_lower"
histname = "theta_p"

directoryname=".ROI"

# h_em_ep_angle
rebin=40
xoffset=2
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
    normFactor=sum(y)
    y/=normFactor
    yerr/=normFactor
    return x,y,yerr

# <editor-fold desc="Get Data">
simWeights=[0.242,0.366,0.069,1.13,0.29] #Not including 2H (0.242 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_2H_MF_helicity_mixed.root","hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]
if A=='D':
    simFiles=[simFiles[0]]
    simWeights=[simWeights[0]]
if A=='He':
    simFiles=simFiles[1:3]
    simWeights=simWeights[1:3]
if A=='C':
    simFiles=simFiles[4:6]
    simWeights=simWeights[4:6]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/"
x_data,y_data,yerr_data=getXY(infiles=[filepath+f"data_hist_D.root"],
                              weights=[1],histname=histname,rebin=rebin)

x_sim,y_sim,yerr_sim=getXY(infiles=[filepath+tree for tree in simFiles],
                              weights=simWeights,histname=histname,rebin=rebin)

x_sim+=xoffset
# </editor-fold>


# Plotting
plt.figure()
# plt.figure(figsize=(5,6))
# # Subthreshold Plots
# plt.subplot(2,1,1)
plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.r',capsize=0,label="data")
plt.errorbar(x_sim,y_sim,yerr=yerr_sim,fmt='.g',capsize=0,label="sim")
plt.legend(frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

# plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Normalized Counts")
plt.xlabel(r'$\theta_p$')
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
# plt.ylim(ymin,ymax*1.3)
# placeText("Sideband Sub.",loc=2,yoffset=-25)
# placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)



# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/alpha_miss_SBS_fitted.pdf")
plt.show()

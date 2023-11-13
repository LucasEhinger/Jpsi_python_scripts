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

xmin=6
xmax=11
MF2SRC_factor=6.5
# histname = "alpha_miss_SB_lower"
histname = "Egamma_ROI"
Egamma_subt="8p2"
Egamma_thresh="8p2"
rebin_data=10
rebin_sim=2
xoffset=0.01
def getXY(infiles,weights,histname, rebin):
    x=0
    y=0
    yerr=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,".ROI")
        h = f.get(histname, rebin=rebin)
        x = h.x
        y += h.y*weights[i]
        yerr = np.sqrt(yerr**2 +(h.yerr*weights[i])**2)
    return x,y,yerr

# <editor-fold desc="Get Data">
simWeights=[0.54,0.08,1.64,0.33] #Not including 2H (0.35 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_4He_MF.root", "hist_DSelector_4He_SRC.root", "hist_DSelector_12C_MF.root", "hist_DSelector_12C_SRC.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/"
x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname,rebin=rebin_data)

x_sim,y_sim,yerr_sim=getXY(infiles=[filepath+tree for tree in simFiles],
                              weights=simWeights,histname=histname,rebin=rebin_sim)
# </editor-fold>

# <editor-fold desc="Scaling">
# Scaling
scaleFactor=sum(y_data)/sum(y_sim)

y_sim *= scaleFactor*rebin_data/rebin_sim
yerr_sim *= scaleFactor*rebin_data/rebin_sim
# </editor-fold>

# <editor-fold desc="Plotting">
plt.figure(figsize=(5,3))
plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0,label="He+C data")
plt.errorbar(x_sim,y_sim,yerr=yerr_sim,fmt='.b--',capsize=0,label="He Sim.")
plt.legend(frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlabel(r"$E_{\gamma}$ [GeV]")
plt.xlim(xmin,xmax)
placeText("No Sideband Sub.",loc=2,yoffset=-25)
placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)

plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Egamma_ROI.pdf")
plt.show()
# </editor-fold>

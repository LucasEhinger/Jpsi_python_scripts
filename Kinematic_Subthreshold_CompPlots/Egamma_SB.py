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

# histname = "alpha_miss_SB_lower"
histname_ROI = "Egamma_ROI"
histname_SB = "Egamma_SB"
Egamma_subt="8p2"
Egamma_thresh="8p2"
rebin=10
xoffset=0.05
def getXY(infiles,weights,histname, rebin,dir):
    x=0
    y=0
    yerr=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=dir)
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
x_data_ROI,y_data_ROI,yerr_data_ROI=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_ROI,rebin=rebin,dir=".ROI")

x_data_SBS,y_data_SBS,yerr_data_SBS=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_SB,rebin=rebin,dir=".SB")
# </editor-fold>


# <editor-fold desc="Plotting">
plt.figure(figsize=(5,3))
plt.errorbar(x_data_ROI,y_data_ROI,yerr=yerr_data_ROI,fmt='.k',capsize=0,label="ROI")
plt.errorbar(x_data_SBS+xoffset,y_data_SBS,yerr=yerr_data_SBS,fmt='.b',capsize=0,label="Side-band")
plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)
plt.ylabel("Counts")
plt.xlabel(r"$E_{\gamma}$ [GeV]")
plt.xlim(xmin,xmax)
placeText("He+C Data",loc=2,yoffset=-25)
# placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)

plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Egamma_SB.pdf")
plt.show()
# </editor-fold>

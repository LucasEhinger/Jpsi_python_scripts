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
xmax=1
MF2SRC_factor=6.5
histname_ROI = "Pt_ROI"
histname_SB = "Pt_SB"

Egamma_subt="8p2"
Egamma_thresh="8p2"
rebin=4
xoffset=0.01


def getXY(infiles,weights,histname, rebin, dir):
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
dataFiles=["data_hist_He.root", "data_hist_C.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_lower/"
x_ROI_subt,y_ROI_subt,yerr_ROI_subt=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_ROI,rebin=rebin,dir=".ROI")

x_SB_subt,y_SB_subt,yerr_SB_subt=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_SB,rebin=rebin,dir=".SB")

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_thresh}_upper/"
x_ROI_thresh,y_ROI_thresh,yerr_ROI_thresh=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_ROI,rebin=rebin,dir=".ROI")

x_SB_thresh,y_SB_thresh,yerr_SB_thresh=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_SB,rebin=rebin,dir=".SB")
# </editor-fold>

# <editor-fold desc="Plotting">
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_ROI_subt,y_ROI_subt,yerr=yerr_ROI_subt,fmt='.k',capsize=0,label="ROI")
plt.errorbar(x_SB_subt+xoffset,y_SB_subt,yerr=yerr_SB_subt,fmt='.b',capsize=0,label="Side-band")
plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.2)
placeText("He+C Data",loc=2,yoffset=-25)
# placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".")+" GeV",loc=1)
plt.legend(loc='center right',frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)
# Above threshold Plots
# -----------------------------------------------------------------
plt.subplot(2,1,2)
plt.errorbar(x_ROI_thresh,y_ROI_thresh,yerr=yerr_ROI_thresh,fmt='.k',capsize=0)
plt.errorbar(x_SB_thresh+xoffset,y_SB_thresh,yerr=yerr_SB_thresh,fmt='.b',capsize=0)


plt.ylabel("Counts")
plt.xlabel(r"$p_t$ [GeV]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
# plt.ylim(ymin,ymax*1.2)
placeText(r"$E_{\gamma}>$"+Egamma_thresh.replace("p",".")+" GeV",loc=1)
# </editor-fold>

plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Pt_SB.pdf")
plt.show()

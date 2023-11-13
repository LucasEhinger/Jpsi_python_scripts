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
xmax=6
MF2SRC_factor=6.5
# histname = "alpha_miss_SB_lower"
histname = "p_p"

directoryname=".Proton"
# h_em_ep_angle
rebin=50
xoffset=0.1
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
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/"
x_data_D,y_data_D,yerr_data_D=getXY(infiles=[filepath+f"data_hist_D.root"],
                              weights=[1],histname=histname,rebin=rebin)

x_data_He,y_data_He,yerr_data_He=getXY(infiles=[filepath+f"data_hist_He.root"],
                              weights=[1],histname=histname,rebin=rebin)

x_data_C,y_data_C,yerr_data_C=getXY(infiles=[filepath+f"data_hist_C.root"],
                              weights=[1],histname=histname,rebin=rebin)

y_data_CoverHe=y_data_C/y_data_He
yerr_data_CoverHe=np.abs(y_data_CoverHe*(yerr_data_C/y_data_C+yerr_data_He/y_data_He))
y_data_CoverD=y_data_C/y_data_D
yerr_data_CoverD=np.abs(y_data_CoverD*(yerr_data_C/y_data_C+yerr_data_D/y_data_D))
y_data_HeoverD=y_data_He/y_data_D
yerr_data_HeoverD=np.abs(y_data_HeoverD*(yerr_data_He/y_data_He+yerr_data_D/y_data_D))
x_data_He+=xoffset
x_data_C+=2*xoffset
# </editor-fold>


# Plotting
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_data_D,y_data_D,yerr=yerr_data_D,fmt='.r',capsize=0,label="D")
plt.errorbar(x_data_He,y_data_He,yerr=yerr_data_D,fmt='.g',capsize=0,label="He")
plt.errorbar(x_data_C,y_data_C,yerr=yerr_data_D,fmt='.b',capsize=0,label="C")
plt.legend(frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

# plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Normalized Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
# plt.ylim(ymin,ymax*1.3)
# placeText("Sideband Sub.",loc=2,yoffset=-25)
# placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)


plt.subplot(2,1,2)
plt.errorbar(x_data_D,y_data_CoverHe,yerr=yerr_data_CoverHe,fmt='.r',capsize=0,label="C/He")
plt.errorbar(x_data_He,y_data_CoverD,yerr=yerr_data_CoverD,fmt='.g',capsize=0,label="C/D")
plt.errorbar(x_data_C,y_data_HeoverD,yerr=yerr_data_HeoverD,fmt='.b',capsize=0,label="He/D")
plt.plot([xmin,xmax],[1,1],'k:')
plt.legend(frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)
plt.ylabel("Ratio")
plt.xlabel(r"$p_{p}$ [GeV]]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(0,3)

# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/alpha_miss_SBS_fitted.pdf")
plt.show()

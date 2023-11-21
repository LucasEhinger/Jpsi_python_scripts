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
xmax=1.2
histname_ROI = "pT_ROI"
sub_directoryname_ROI=".Kin.ROI"
histname_SB = "pT_SB"
sub_directoryname_SB=".Kin.SB"
vers="v8"
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
pTCut=False

rebin=20
xoffset=0.03

def getXY(infiles,weights,histname, rebin,directoryname):
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
dataFiles=["data_hist_He.root", "data_hist_C.root"]

directoryname=".SubThresh"
if pTCut:
    directoryname=".SubThresh_pt03_lower"
x_ROI_subt,y_ROI_subt,yerr_ROI_subt= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                           histname=histname_ROI, rebin=rebin, directoryname=directoryname+sub_directoryname_ROI)

x_SB_subt,y_SB_subt,yerr_SB_subt= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                        histname=histname_SB, rebin=rebin, directoryname=directoryname+sub_directoryname_SB)


directoryname=".AboveThresh"
if pTCut:
    directoryname=".AboveThresh_pt03_lower"
x_ROI_thresh,y_ROI_thresh,yerr_ROI_thresh= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                                 histname=histname_ROI, rebin=rebin, directoryname=directoryname+sub_directoryname_ROI)

x_SB_thresh,y_SB_thresh,yerr_SB_thresh= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                              histname=histname_SB, rebin=rebin, directoryname=directoryname+sub_directoryname_SB)
# </editor-fold>




# Plotting
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_ROI_subt,y_ROI_subt,yerr=yerr_ROI_subt,fmt='.k',capsize=0,label="ROI")

plt.errorbar(x_SB_subt+xoffset,y_SB_subt,yerr=yerr_SB_subt,fmt='.r',capsize=0,label="Side-band")
plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)
placeText("He+C"+"\n"+r"$m(e^+e^-)$ Sideband",loc=2,yoffset=-40)
placeText(r"$E_{\gamma}<8.2$ GeV",loc=1)


plt.subplot(2,1,2)
plt.errorbar(x_ROI_thresh,y_ROI_thresh,yerr=yerr_ROI_thresh,fmt='.k',capsize=0)
plt.errorbar(x_SB_thresh+xoffset,y_SB_thresh,yerr=yerr_SB_thresh,fmt='.r',capsize=0)
plt.ylabel("Counts")
plt.xlabel(r"$p_T$ [GeV]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)


plt.savefig(f"../../files/figs/kin/subthreshold_comp/pT/p_T_SB_{vers}_{'pT03_' if pTCut else ''}mixed.pdf")
plt.show()

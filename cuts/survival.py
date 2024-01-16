#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

A="D"
ROI_min=2.9
ROI_max=3.2

filepath="/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/v8/filtered/noTrackShower/"
histname = "mass_pair_"
directoryname = ".MassPairPlots"
cuts=["All","Fiducial","TrackShower","preBCal","PoverE","Emiss"]
cut_counts=np.zeros(len(cuts))
def getXY(infiles, weights, histname, rebin,directoryname):
    x = 0
    y = 0
    yerr = 0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile, dir=directoryname)
        h = f.get(histname, rebin=rebin)
        x = h.x
        y += h.y * weights[i]
        yerr = np.sqrt(yerr ** 2 + (h.yerr * weights[i]) ** 2)
    return x, y, yerr


for i,cut_name in enumerate(cuts):
    x_data, y_data, yerr_data = getXY(infiles=[filepath+f"data_hist_cutflow_{A}.root"],
                                      weights=[1], histname=histname+cut_name, rebin=1, directoryname=directoryname)
    counts=0
    for j,x_val in enumerate(x_data):
        if x_val>ROI_min and x_val<ROI_max:
            counts=counts+y_data[j]
    cut_counts[i]=counts

plt.bar(range(len(cuts)),cut_counts)
plt.xticks(range(len(cuts)),["All","Fiducial","TrackShower","preBCal","PoverE","Emiss"],rotation=45)
plt.yscale('log')
plt.ylabel("Events")
plt.title(r"Surviving Events")
placeText(r"2.9< m($e^+e^-$) < 3.2 GeV",loc=1)
placeText(A,loc=2,yoffset=-25)
plt.savefig(f"../../files/figs/cuts/cutflow_{A}.pdf")
plt.show()
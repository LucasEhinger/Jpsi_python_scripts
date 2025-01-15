#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

A="He"
vers="v8"
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
directoryname=".PreShowerPlots.Before"
histname = "em_eprebcal_sinTheta"
f = root2mpl.File(filepath+f"data_hist2_cutflow_{A}.root",dir=directoryname)

rebin=20

plt.figure()
f.plotPoints(histname,rebin=rebin,ls='--',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([0.03,0.03],[ymin,ymax],'r-')
plt.ylim(ymin,ymax)
plt.xlim(0,0.25)
plt.xlabel(r"$E_{preBCal}$ sin($\theta$) [GeV]")
plt.ylabel("Counts")
# plt.title("Vertex z Cuts")
placeText(A+"\nData",loc=1)

plt.savefig(f"../../files/figs/cuts/preBcut_data_{A}.pdf", bbox_inches = 'tight')

plt.show()

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
directoryname=".PreShowerPlots.Before"
histname = "em_eprebcal_sinTheta"

f2 = root2mpl.File(filepath+f"hist2_cutflow_DSelector_4He_MF_helicity_mixed.root",dir=directoryname)

rebin=20

plt.clf()
plt.figure()
f2.plotPoints(histname,rebin=rebin,ls='--',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([0.03,0.03],[ymin,ymax],'r-')
plt.ylim(ymin,ymax)
plt.xlim(0,0.25)
plt.xlabel(r"$E_{preBCal}$ sin($\theta$) [GeV]")
plt.ylabel("Counts")
# plt.title("Vertex z Cuts")
placeText(A+"\nSim",loc=1)
plt.savefig(f"../../files/figs/cuts/preBcut_sim_{A}.pdf", bbox_inches = 'tight')
plt.show()
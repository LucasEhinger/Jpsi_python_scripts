#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

vers="v8"
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
A="He"
directoryname=".EmissPlots.Before"
histname = "Emiss"
# f = root2mpl.File(filepath+f"data_hist2_cutflow_{A}.root",dir=directoryname)
f = root2mpl.File(filepath+f"hist_cutflow_DSelector_4{A}_MF_helicity_mixed_2.root",dir=directoryname)
rebin=4

plt.figure()
f.plotPoints(histname,rebin=rebin,ls='--',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([-1,-1],[ymin,ymax],'r-')
plt.plot([1,1],[ymin,ymax],'r-')
plt.ylim(ymin,ymax)
plt.xlim(-4,4)
plt.xlabel(r"$E_{miss}$ [GeV]")
plt.ylabel("Counts")
# plt.title("Vertex z Cuts")
placeText(A+"\nSim",loc=1)
plt.savefig(f"../../files/figs/cuts/Emiss_sim_{A}.pdf", bbox_inches = 'tight')
plt.show()
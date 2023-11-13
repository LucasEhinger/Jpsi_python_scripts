#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

filepath= "/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/fiducialCuts/negWeights/"
A="He"
f = root2mpl.File(filepath+f"data_hist_{A}.root")
histname = "em_eprebcal_sinTheta"
rebin=7

# plt.figure()
# f.plotPoints(histname,rebin=rebin,ls='--',color='k',capsize=0,marker='.')
# xmin, xmax, ymin, ymax = plt.axis()
# plt.plot([0.03,0.03],[ymin,ymax],'r-')
# plt.ylim(ymin,ymax)
# plt.xlim(0,0.25)
# plt.xlabel(r"$E_{preBCal}$ sin($\theta$) [GeV]")
# plt.ylabel("Counts")
# # plt.title("Vertex z Cuts")
# placeText(A+"\nData",loc=1)
# plt.savefig("figures/"+f"preBcut_data_{A}.pdf", bbox_inches = 'tight')
# plt.show()

f2 = root2mpl.File(filepath+f"sim_hist_He_MF.root")
histname = "em_eprebcal_sinTheta"
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
placeText(A+"\nSimulation",loc=1)
plt.savefig("figures/"+f"preBcut_sim_{A}.pdf", bbox_inches = 'tight')
plt.show()
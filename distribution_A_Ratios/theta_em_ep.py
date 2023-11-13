#!/usr/bin/env python3.10


import numpy as np

import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm, multivariate_normal

from pystyle.annotate import placeText

from scipy.optimize import curve_fit
import os
plt.style.use("SRC_CT_presentation")

A = "He"
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_0/Emiss_100/"

directoryname=".ROI"

f = root2mpl.File(filepath+f"data_hist_{A}.root",dir=directoryname)
cutLoc="p/E Cuts"

rebinx=1
rebiny=1

histname = "theta_em_ep"

plt.figure()
f.plotHeatmap(histname,rebinx=rebinx,rebiny=rebiny)
plt.xlabel(r"$\theta_{e^-}$")
plt.ylabel(r"$\theta_{e^+}$")
xmin,xmax,ymin,ymax=plt.axis()
plt.plot([xmin,xmax],[11,11],'r--')
plt.plot([11,11],[ymin,ymax],'r--')
# plt.title(rf"{A} Data")

placeText(A+"\n"+cutLoc,loc=1)

# plt.savefig(f"noSubtractionOrigCuts/figures/{A}_cut_noBackgroundSubtraction.png",dpi=1000)

plt.savefig(f"../figures/Cuts/theta_em_ep_pOverE.pdf")
plt.show()
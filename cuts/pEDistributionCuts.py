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

A = "C"
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE_Fitting/preB_03/noCalCuts/"

f = root2mpl.File(filepath+f"data_hist_{A}.root")


rebinx=3
rebiny=3

bkd_min_bcal=1.2450
bkd_max_bcal=1.3893

sig_min_bcal=0.8123
sig_max_bcal=1.1729


histname = "Cal_PoverE"
# histname = "PoverE_2D_3_inf"

plt.figure()
plt.gca().set_aspect('equal')
f.plotHeatmap(histname,rebinx=rebinx,rebiny=rebiny)
plt.plot([0.6,1.4],[bkd_min_bcal,bkd_min_bcal],'r--',linewidth=2)
plt.plot([0.6,1.4],[bkd_max_bcal,bkd_max_bcal],'r--',linewidth=2)

plt.plot([0.6,1.4],[sig_min_bcal,sig_min_bcal],'r-',linewidth=2)
plt.plot([0.6,1.4],[sig_max_bcal,sig_max_bcal],'r-',linewidth=2)

plt.xlim(0.6,1.4)
plt.ylim(0.6,1.4)
plt.xlabel(r"FCAL p/E")
plt.ylabel(r"BCAL p/E")
# plt.title(rf"{A} Data")


# placeText(A,loc=2,yoffset=-23)

# plt.savefig(f"noSubtractionOrigCuts/figures/{A}_cut_noBackgroundSubtraction.png",dpi=1000)


plt.savefig(f"figures/pOverE/PoverE_2D_data_{A}.pdf", bbox_inches = 'tight')
plt.show()
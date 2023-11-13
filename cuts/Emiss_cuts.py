#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_0/Emiss_100/"
A="C"
f = root2mpl.File(filepath+f"data_hist_{A}.root")
histname = "missing Energy: stationary proton"
rebin=10

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
placeText(A+"\nData",loc=1)
plt.savefig("figures/"+f"Emiss_{A}.pdf", bbox_inches = 'tight')
plt.show()

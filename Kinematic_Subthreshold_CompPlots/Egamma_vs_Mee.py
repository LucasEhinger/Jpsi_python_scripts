#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

filepath= "/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/8p2_lower/"
f_He = root2mpl.File(filepath+f"data_hist_He.root",".SubThresh2D")
f_C = root2mpl.File(filepath+f"data_hist_C.root",".SubThresh2D")
histname = "Egamma_v_mee"
rebinx=1
rebiny=1
Egamma="8p2"

plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
h_He=f_He.get(histname,rebinx=rebinx,rebiny=rebiny)
h_C=f_C.get(histname,rebinx=rebinx,rebiny=rebiny)
plt.pcolormesh(h_He.xedge,h_He.yedge,h_He.z+h_C.z)
plt.xlabel(r'$E_{\gamma}$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
plt.plot([8.2,8.2],[ymin,ymax],'k:')
placeText("He+C",loc=2,yoffset=-30)
placeText(r"$E_{\gamma}<$"+Egamma.replace("p",".")+" GeV",loc=1,yoffset=-30)


plt.subplot(2,1,2)
plt.xlim(xmin,xmax)
filepath= "/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/8p2_upper/"
f2_He = root2mpl.File(filepath+f"data_hist_He.root",".SubThresh2D")
f2_C = root2mpl.File(filepath+f"data_hist_C.root",".SubThresh2D")
h2_He=f2_He.get(histname,rebinx=rebinx,rebiny=rebiny)
h2_C=f2_C.get(histname,rebinx=rebinx,rebiny=rebiny)
plt.pcolormesh(h2_He.xedge,h2_He.yedge,h2_He.z+h2_C.z)
plt.xlabel(r'$E_{\gamma}$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
plt.plot([8.2,8.2],[ymin,ymax],'k:')
placeText(r"$E_{\gamma}>$"+Egamma.replace("p",".")+" GeV",loc=1,yoffset=-30)



# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Egamma_vs_Mee_He+C.pdf")
plt.show()


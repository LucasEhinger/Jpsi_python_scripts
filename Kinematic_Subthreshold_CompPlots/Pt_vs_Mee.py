#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

Egamma_subt="8p2"
Egamma_thresh="8p2"
target="He+C"
histname = "Pt_v_mee"
dir=".SubThresh2D"
rebinx=1
rebiny=1
xmin=0
xmax=1

def getXYZ(infiles,weights,histname, rebinx,rebiny):
    x=0
    y=0
    z=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=dir)
        h = f.get(histname, rebinx=rebinx, rebiny=rebiny)
        x = h.xedge
        y = h.yedge
        z += h.z*weights[i]
    return x,y,z

# <editor-fold desc="Get Data">
dataFiles=["data_hist_He.root", "data_hist_C.root"]
filepath= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_lower/"
x_subt,y_subt,z_subt=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=[1,1],histname=histname,rebinx=rebinx,rebiny=rebiny)
filepath= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_thresh}_upper/"
x_thresh,y_thresh,z_thresh=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=[1,1],histname=histname,rebinx=rebinx,rebiny=rebiny)
# </editor-fold>


# <editor-fold desc="Plotting">
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.pcolormesh(x_subt,y_subt,z_subt)
plt.xlim(xmin,xmax)
plt.xlabel(r'$P_t$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText(target,loc=2,yoffset=-30)
placeText(r"$E_{\gamma}<$"+Egamma_thresh.replace("p",".")+" GeV",loc=1,yoffset=-30)


plt.subplot(2,1,2)
plt.pcolormesh(x_thresh,y_thresh,z_thresh)
plt.xlim(xmin,xmax)
plt.xlabel(r'$P_t$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText(r"$E_{\gamma}>$"+Egamma_thresh.replace("p",".")+" GeV",loc=1,yoffset=-30)
# </editor-fold>

# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Pt_vs_Mee_{target}.pdf")
plt.show()


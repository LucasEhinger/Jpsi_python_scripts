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
target="He + C"
histname = "alpha_miss_vs_pT_ROI"
MF2SRC_factor=6.5
rebinx=2
rebiny=2
xmin=0.7
xmax=2
ymin=0
ymax=1

def getXYZ(infiles,weights,histname, rebinx,rebiny):
    x=0
    y=0
    z=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,".ROI")
        h = f.get(histname, rebinx=rebinx, rebiny=rebiny)
        x = h.xedge
        y = h.yedge
        z += h.z*weights[i]
    return x,y,z


simWeights=[0.366,0.069,1.13,0.29] #Not including 2H (0.242 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]


filepath= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_lower/"
x_subt,y_subt,z_subt=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=[1,1],histname=histname,rebinx=rebinx,rebiny=rebiny)

filepath= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_thresh}_upper/"
x_thresh,y_thresh,z_thresh=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=[1,1],histname=histname,rebinx=rebinx,rebiny=rebiny)


plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_subt,y_subt,z_subt)
xlin=np.linspace(xmin,xmax,1000)
plt.plot([xmin,xmax],[0.3,0.3],'r--')
plt.plot([1.2,1.2],[ymin,ymax],'r--')
plt.plot(xlin,0.8-0.8/1.4*xlin,'r--')
plt.ylabel(r'$P_t$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
placeText(target+"\n"+r"$3<m(e^+e^-)<3.2$",loc=2,yoffset=-45)
placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".")+" GeV",loc=1,yoffset=-30)


plt.subplot(2,1,2)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_thresh,y_thresh,z_thresh)
plt.plot([xmin,xmax],[0.3,0.3],'r--')
plt.plot([1.2,1.2],[ymin,ymax],'r--')
plt.plot(xlin,0.8-0.8/1.4*xlin,'r--')
plt.xlabel(r'$\alpha_{miss}$ [GeV]')
plt.ylabel(r'$P_t$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
placeText(r"$E_{\gamma}>$"+Egamma_thresh.replace("p",".")+" GeV",loc=1,yoffset=-30)
# plt.colorbar()

plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Alpha_Miss_vs_Pt_{target}_mixed.pdf")
plt.show()


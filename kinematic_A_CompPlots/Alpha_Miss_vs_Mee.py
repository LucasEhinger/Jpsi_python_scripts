#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

histname = "alpha_miss_vs_mee"
# histname = "Pt_v_mee"
rebinx=1
rebiny=1
xmin=0.7
xmax=2
ymin=2.8
ymax=3.4
version="v7"
tracks="noTracks"

def getXYZ(infiles,weights,histname, rebinx,rebiny):
    x=0
    y=0
    z=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=".SubThresh2D")
        h = f.get(histname, rebinx=rebinx, rebiny=rebiny)
        x = h.xedge
        y = h.yedge
        z += h.z*weights[i]
    return x,y,z

filepath= filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{version}/PoverE/m3_p2_sigma/{tracks}/preB_03/Emiss_1/"
x_D,y_D,z_D=getXYZ([filepath+"data_hist_D.root"],
                            weights=[1],histname=histname,rebinx=rebinx,rebiny=rebiny)
x_He,y_He,z_He=getXYZ([filepath+"data_hist_He.root"],
                            weights=[1],histname=histname,rebinx=rebinx,rebiny=rebiny)
x_C,y_C,z_C=getXYZ([filepath+"data_hist_C.root"],
                            weights=[1],histname=histname,rebinx=rebinx,rebiny=rebiny)


plt.figure(figsize=(6,10))
plt.subplot(3,1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_D,y_D,z_D)
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText('D',loc=1,yoffset=-25)
if tracks=="noTracks":
    placeText("No Sideband Sub."+"\n"+"Mixed"+"\n"+version+" No Extra Tracks",loc=2,yoffset=-60)
else:
    placeText("No Sideband Sub." + "\n" + "Mixed" + "\n" + version + " Incl. Extra Tracks", loc=2, yoffset=-60)


plt.subplot(3,1,2)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_He,y_He,z_He)
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText('He',loc=1,yoffset=-25)

plt.subplot(3,1,3)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_C,y_C,z_C)
plt.xlabel(r'$\alpha_{miss}$ [GeV]')
# plt.xlabel(r'$p_{T}$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot([xmin,xmax],[3,3],'r:')
plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText('C',loc=1,yoffset=-25)

# plt.savefig(f"../figures/p2_preB03_Emiss1/AComp/2D/Alpha_Miss_vs_Mee_comp.pdf")
# plt.savefig(f"../figures/p2_preB03_Emiss1/AComp/2D/pT_vs_Mee_comp.pdf")
plt.show()


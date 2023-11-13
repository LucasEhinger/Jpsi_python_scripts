#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

histname = "mpair_v_mkin"
directory=".All_alpha1p2_lower.Kin.Jpsi_mass"
# histname = "Pt_v_mee"
rebinx=5
rebiny=5
xmin=2.8
xmax=3.4
ymin=2.8
ymax=3.4
vers="v7"

def getXYZ(infiles,weights,histname, rebinx,rebiny):
    x=0
    y=0
    z=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=directory)
        h = f.get(histname, rebinx=rebinx, rebiny=rebiny)
        x = h.xedge
        y = h.yedge
        z += h.z*weights[i]
    return x,y,z

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/"

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
plt.ylabel(r'$m(e^+e^-)$ True [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
xlin=np.linspace(xmin,xmax,100)
plt.plot(xlin,xlin,'r--',linewidth=2)
placeText('D',loc=1,yoffset=-25)

placeText("No Sideband Sub."+"\n"+"Mixed"+"\n"+vers+" Extra Track",loc=2,yoffset=-60)

plt.subplot(3,1,2)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_He,y_He,z_He)
plt.ylabel(r'$m(e^+e^-)$ True [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.plot(xlin,xlin,'r--',linewidth=2)
# plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText('He',loc=1,yoffset=-25)

plt.subplot(3,1,3)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.pcolormesh(x_C,y_C,z_C)
plt.xlabel(r'$m(e^+e^-)$ [GeV]')
# plt.xlabel(r'$p_{T}$ [GeV]')
plt.ylabel(r'$m(e^+e^-)$ True [GeV]')
plt.plot(xlin,xlin,'r--',linewidth=2)

placeText('C',loc=1,yoffset=-25)

# plt.savefig(f"../figures/p2_preB03_Emiss1/AComp/2D/Alpha_Miss_vs_Mee_comp.pdf")
plt.savefig(f"../figures/p2_preB03_Emiss1/AComp/2D/Mee_vs_Mee_True_zoomed_comp.pdf")
plt.show()


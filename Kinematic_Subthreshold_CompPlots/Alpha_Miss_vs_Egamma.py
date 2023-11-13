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
target="He+C Sim"
histname = "alpha_miss_vs_Egamma_ROI"
rebinx=1
rebiny=1
xmin=0.8
xmax=1.8
ymin=0
ymax=1

def getXYZ(infiles,weights,histname, rebinx,rebiny):
    x=0
    y=0
    z=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=".ROI")
        h = f.get(histname, rebinx=rebinx, rebiny=rebiny)
        x = h.xedge
        y = h.yedge
        z += h.z*weights[i]
    return x,y,z

simWeights=[0.366,0.069,1.13,0.29] #Not including 2H (0.242 nb)
simFiles=["hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]

filepath= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/"
x,y,z=getXYZ(infiles=[filepath+tree for tree in simFiles],
             weights=simWeights,histname=histname,rebinx=rebinx,rebiny=rebiny)



def EgammaLimit(alpha):
    # m_jpsi=3.096
    m_jpsi=3.686
    m_n=0.938272
    return ((m_jpsi+m_n)**2-m_n**2)/(2*m_n*alpha)


plt.figure(figsize=(6,5))
plt.xlim(xmin,xmax)
plt.pcolormesh(x,y,z)
plt.xlabel(r'$\alpha_{miss}$ [GeV]')
plt.ylabel(r'$E_{\gamma}$ [GeV]')
xmin,xmax,ymin,ymax =plt.axis()
plt.ylim(ymin,ymax)
# plt.plot([xmin,xmax],[3,3],'r:')
# plt.plot([xmin,xmax],[3.2,3.2],'r:')
# plt.plot([8.2,8.2],[ymin,ymax],'k:')
# placeText("He Simulation",loc=2,yoffset=-25)
placeText(target,loc=2,yoffset=-25)
placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
# placeText(r"$E_{\gamma}<$"+Egamma.replace("p",".")+" GeV",loc=1,yoffset=-30)


alpha_val=np.linspace(0.8,1.4,100)
plt.plot(alpha_val,EgammaLimit(alpha_val),'r')
# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Alpha_Miss_vs_Egamma_He_sim.pdf")
# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Alpha_Miss_vs_Egamma_He+C_sim_mixed.pdf")
plt.show()


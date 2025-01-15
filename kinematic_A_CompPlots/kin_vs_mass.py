#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText
import matplotlib.colors as colors

rebinx=2
rebiny=2
version="v8"
tracks="noTracks"
A="C"

def getXYZ(infiles,weights,histname, rebinx,rebiny,dir):
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

mass_var=["jpsi_m_pair","jpsi_m_kin"]

for kin_var in ["t","alpha_miss","pT","kmiss"]:
    for A in ["D","He","C"]:
        filepath= filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{version}/filtered/noTrackShower/simHists/"
        if A=="D":
            filenames=["hist_mass_DSelector_2H_MF_helicity_mixed_2.root"]
            weights=[1]
        elif A=="He":
            filenames=[f"hist_mass_DSelector_4{A}_MF_helicity_mixed_2.root",f"hist_mass_DSelector_4{A}_SRC_helicity_mixed_2.root"]
            weights=[1,1]
        elif A=="C":
            filenames=[f"hist_mass_DSelector_12C_MF_helicity_mixed_2.root",f"hist_mass_DSelector_12C_SRC_helicity_mixed_2.root"]
            weights=[1,1]
        x_lc,y_lc,z_lc=getXYZ([filepath+filename for filename in filenames],
                                    weights=weights,histname=mass_var[0]+"_vs_"+kin_var,rebinx=rebinx,rebiny=rebiny,dir="."+mass_var[0])
        x,y,z=getXYZ([filepath+filename for filename in filenames],
                                    weights=weights,histname=mass_var[1]+"_vs_"+kin_var,rebinx=rebinx,rebiny=rebiny,dir="."+mass_var[1])

        min_val= min(z.min(),z_lc.min())
        min_val=1
        max_val= max(z.max(),z_lc.max())

        plt.figure(figsize=(6,6))
        plt.subplot(2,1,1)
        plt.pcolormesh(x, y, z, norm=colors.LogNorm(min_val, max_val))
        # plt.pcolormesh(x,y,z,norm=colors.LogNorm(vmin=min_val, vmax=max_val))
        plt.ylabel(r'$m(e^+e^-)$ [GeV/c$^2$]')
        plt.xticks([])
        placeText(A, loc=1, yoffset=-25)

        plt.subplot(2,1,2)
        plt.pcolormesh(x_lc,y_lc,z_lc,norm=colors.LogNorm(min_val, max_val))
        plt.ylabel(r'$m(e^+e^-)$ Lightcone [GeV/c$^2$]')

        if kin_var=="t":
            plt.xlabel("-t [GeV/c]")
        elif kin_var=="alpha_miss":
            plt.xlabel(r'$\alpha_{miss}$')
        elif kin_var=="pT":
            plt.xlabel(r'$p_T$ [GeV/c]')
        elif kin_var=="kmiss":
            plt.xlabel(r'$k_{miss}$ [GeV/c]')

        plt.savefig(f"../../files/figs/simFigs/2D_mass_vs_{kin_var}_{A}_log.pdf", bbox_inches = 'tight')
        plt.show()


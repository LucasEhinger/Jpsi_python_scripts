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

targets=["D","He","C"]
histname = "kin_p"
dir_subt= ".SubThresh.p.ROI"
dir_abovet= ".AboveThresh.p.ROI"
vers="v8"

rebinx=1*1
rebiny=5*1
xmin=0
xmax=5
ymin=0
ymax=90

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

# <editor-fold desc="Get Data">
dataFiles=[f"data_hist2_{A}.root" for A in targets]
simFiles=[f"hist2_DSelector_2H_MF_helicity_mixed.root"]+[f"hist2_DSelector_{A}_{MFSRC}_helicity_mixed.root" for A in ["4He","12C"] for MFSRC in ["MF","SRC"]]
dataFiles = simFiles
weight_arr=np.ones(len(dataFiles))
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
x_subt,y_subt,z_subt=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=weight_arr,histname=histname,rebinx=rebinx,rebiny=rebiny,dir=dir_subt)
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
x_thresh,y_thresh,z_thresh=getXYZ(infiles=[filepath+tree for tree in dataFiles],
                            weights=weight_arr,histname=histname,rebinx=rebinx,rebiny=rebiny,dir=dir_abovet)
# </editor-fold>


min_val = min(np.min(z_subt[z_subt>0]),np.min(z_thresh[z_thresh>0]))
min_val=0.1
max_val = max(np.max(z_subt),np.max(z_thresh))
# <editor-fold desc="Plotting">
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.pcolormesh(x_subt,y_subt,z_subt,norm=colors.LogNorm(min_val, max_val))
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel(r'$p_p$ [GeV/c]')
plt.ylabel(r'$\theta_p$ [Deg]')
xmin,xmax,ymin,ymax =plt.axis()
# plt.plot([xmin,xmax],[3,3],'r:')
# plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText(targets,loc=2,yoffset=-30)
placeText(r"$E_{\gamma}<8.2$ GeV",loc=1,yoffset=0)


plt.subplot(2,1,2)
plt.pcolormesh(x_thresh,y_thresh,z_thresh,norm=colors.LogNorm(min_val, max_val))
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel(r'$p_p$ [GeV/c]')
plt.ylabel(r'$\theta_p$ [Deg]')
xmin,xmax,ymin,ymax =plt.axis()
# plt.plot([xmin,xmax],[3,3],'r:')
# plt.plot([xmin,xmax],[3.2,3.2],'r:')
placeText(r"$E_{\gamma}>8.2$ GeV",loc=1,yoffset=0)
# </editor-fold>

# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/Pt_vs_Mee_{target}.pdf")
plt.savefig(f"../../files/figs/kin/p_rescattering/sim_theta_p_vs_p_p_noTrackShower_{vers}.pdf")
plt.show()


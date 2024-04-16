#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
import os
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

A="He"
vers="v8"
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
directoryname=".VertexPlots.Before"
# f = root2mpl.File(filepath+f"data_hist2_cutflow_{A}.root",dir=directoryname)
f = root2mpl.File(filepath+f"hist_cutflow_DSelector_4{A}_MF_helicity_mixed.root",dir=directoryname)

histname = "xyVertex"
rebinx=4
rebiny=4


plt.figure()
plt.gca().set_aspect('equal')
f.plotHeatmap(histname,rebinx=rebinx,rebiny=rebiny)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel("x [cm]")
plt.ylabel("y [cm]")
# plt.title(A)

tmpvar=np.linspace(0,2*3.15,100)
x=np.cos(tmpvar)
y=np.sin(tmpvar)
plt.plot(x,y,'r-',linewidth=2)
# placeText(A,loc=2,yoffset=-25)
plt.title(A+" Sim")
plt.savefig(f"../../files/figs/cuts/xyCut_sim_{A}.pdf", bbox_inches = 'tight')
plt.show()



rebin=100
plt.clf()
plt.figure()
f.plotPoints("Z-location",rebin=rebin,ls='--',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([51,51],[ymin,ymax],'r-')
plt.plot([79,79],[ymin,ymax],'r-')
plt.xlim(40,90)
plt.ylim(ymin,ymax)
plt.xlabel("z [cm]")
plt.ylabel("Counts")
# plt.title(A)
# placeText(A,loc=2)
plt.title(A+" Sim")
plt.savefig(f"../../files/figs/cuts/zCut_sim_{A}.pdf", bbox_inches = 'tight')

plt.show()

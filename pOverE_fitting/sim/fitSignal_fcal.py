#!/usr/bin/env python310root


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import root2mpl
import pystyle
from scipy.stats import norm
from pystyle.annotate import placeText
from scipy.optimize import curve_fit
import sys

plt.style.use("SRC_CT_presentation")

A="D"
vers="v8"
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
directoryname=".PoverEPlots.Before"

f = root2mpl.File(filepath+f"hist2_cutflow_DSelector_4He_MF_helicity_mixed.root",dir=directoryname)
histname = "fcal_PoverE"

rebin=30

histname = "fcal_PoverE"

plt.figure(figsize=(8,5))
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
plt.xlim(0.7,1.5)



plt.xlabel(r"fcal p/E")
plt.ylabel(r"Counts")
# plt.title("Background + Signal Fit")

h = f.get(histname,rebin=rebin)

xmin=0.8
xmax=1.4
min_index=int((xmin-min(h.x))/(max(h.x)-min(h.x))*len(h.x))
max_index=int((xmax-min(h.x))/(max(h.x)-min(h.x))*len(h.x))

x = h.x[min_index:max_index]
y = h.y[min_index:max_index]
yerr = np.sqrt(h.yerr[min_index:max_index]**2+1)
dx = x[1]-x[0]



def fun(x,A,mu,sigma):
    return (A * norm.pdf(x,loc=mu,scale=sigma))*dx

p0 = [10**7,1,0.1]
popt, pcov = curve_fit(fun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
N = popt[0]
mu = popt[1]
sigma = popt[2]
#
N_err = np.sqrt(pcov[0][0])
mu_err = np.sqrt(pcov[1][1])
sigma_err = np.sqrt(pcov[2][2])
#
# print("N: ", N)
# print("mu: ", mu)
# print("sigma: ", sigma)
print(mu)
print(sigma)

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin, fun(xlin,*popt),zorder=100,color='r')

xmin, xmax, ymin, ymax= plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText(rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"+"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")#+"\n"+r"")
# # placeText(A+"\n"+f"{cut}",loc=2)
# # placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
placeText(A+" Sim"+"\n"+r"$-3\sigma$ $<$ $p/E_{BCAL}-\langle p/E_{BCAL} \rangle$ $<$ $2\sigma$" ,loc=2)
# "\n"+ "Iteration " +str(iter)+

plt.savefig(f"../../../files/figs/cuts/{A}_fcal_sig_bkd_fit_sim.pdf", bbox_inches = 'tight')
plt.show()

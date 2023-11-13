#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm

from pystyle.annotate import placeText

from scipy.optimize import curve_fit
import os

plt.style.use("SRC_CT_presentation")

xmin=0.5
xmax=2
MF2SRC_factor=6.5
# histname = "alpha_miss_SB_lower"
histname_ROI = "alpha_miss_ROI"
histname_SB = "alpha_miss_SB"
Egamma_subt="8p2"
Egamma_thresh="8p2"
rebin=5
xoffset=0.03
def getXY(infiles,weights,histname, rebin,dir):
    x=0
    y=0
    yerr=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=dir)
        h = f.get(histname, rebin=rebin)
        x = h.x
        y += h.y*weights[i]
        yerr = np.sqrt(yerr**2 +(h.yerr*weights[i])**2)
    return x,y,yerr
def fitGaussian(x,y,yerr,xstart,xend):
    first = int((xstart - min(x)) / (max(x) - min(x)) * len(x))
    last = int((xend - min(x)) / (max(x) - min(x)) * len(x))
    x = x[first:last]
    y = y[first:last]
    yerr=np.sqrt(yerr[first:last]**2+1)
    dx = x[1] - x[0]
    def fun(x, A, mu, sigma):
        return (A * norm.pdf(x, loc=mu, scale=sigma)) * dx
    p0 = [5, 1.2, 0.2]
    popt, pcov = curve_fit(fun, x, y, sigma=yerr, absolute_sigma=True, p0=p0, maxfev=10000)
    N = popt[0]
    mu = popt[1]
    sigma = popt[2]
    N_err = np.sqrt(pcov[0][0])
    mu_err = np.sqrt(pcov[1][1])
    sigma_err = np.sqrt(pcov[2][2])
    xlin = np.linspace(x[0], x[-1], num=1000)
    yfit=fun(xlin,*popt)

    return xlin, yfit, N, N_err, mu, mu_err, sigma, sigma_err


# <editor-fold desc="Get Data">
dataFiles=["data_hist_He.root", "data_hist_C.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_subt}_lower/"
x_ROI_subt,y_ROI_subt,yerr_ROI_subt=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_ROI,rebin=rebin,dir=".ROI")

x_SB_subt,y_SB_subt,yerr_SB_subt=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_SB,rebin=rebin,dir=".SB")

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/{Egamma_thresh}_upper/"
x_ROI_thresh,y_ROI_thresh,yerr_ROI_thresh=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_ROI,rebin=rebin,dir=".ROI")

x_SB_thresh,y_SB_thresh,yerr_SB_thresh=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname_SB,rebin=rebin,dir=".SB")
# </editor-fold>



# <editor-fold desc="Fit Data">
xlin_ROI_subt, yfit_ROI_subt, N_ROI_subt, N_err_ROI_subt, \
    mu_ROI_subt, mu_err_ROI_subt, sigma_ROI_subt, sigma_err_ROI_subt \
    = fitGaussian(x=x_ROI_subt,y=y_ROI_subt,yerr=yerr_ROI_subt,xstart=1,xend=1.7)

xlin_SB_subt, yfit_SB_subt, N_SB_subt, N_err_SB_subt, \
    mu_SB_subt, mu_err_SB_subt, sigma_SB_subt, sigma_err_SB_subt \
    = fitGaussian(x=x_SB_subt,y=y_SB_subt,yerr=yerr_SB_subt,xstart=0.8,xend=1.7)

xlin_ROI_thresh, yfit_ROI_thresh, N_ROI_thresh, N_err_ROI_thresh, \
    mu_ROI_thresh, mu_err_ROI_thresh, sigma_ROI_thresh, sigma_err_ROI_thresh \
    = fitGaussian(x=x_ROI_thresh,y=y_ROI_thresh,yerr=yerr_ROI_thresh,xstart=0.8,xend=1.7)

xlin_SB_thresh, yfit_SB_thresh, N_SB_thresh, N_err_SB_thresh, \
    mu_SB_thresh, mu_err_SB_thresh, sigma_SB_thresh, sigma_err_SB_thresh \
    = fitGaussian(x=x_SB_thresh,y=y_SB_thresh,yerr=yerr_SB_thresh,xstart=0.8,xend=1.7)
# </editor-fold>

# Plotting
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_ROI_subt,y_ROI_subt,yerr=yerr_ROI_subt,fmt='.k',capsize=0,label="ROI")
# plt.plot(xlin_ROI_subt,yfit_ROI_subt,'k--')

plt.errorbar(x_SB_subt+xoffset,y_SB_subt,yerr=yerr_SB_subt,fmt='.b',capsize=0,label="Side-band")
# plt.plot(xlin_SB_subt+xoffset,yfit_SB_subt,'b--')
plt.legend(loc="upper left",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
placeText("He+C Data",loc=2,yoffset=-25)
# placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".") + " GeV",loc=1)
# placeText(r"$\mu_{ROI}"+rf"={mu_ROI_subt:.2f}\pm{mu_err_ROI_subt:.2f}$",loc=2)
# placeText("\n"+r"$\mu_{SB}"+rf"={mu_SB_subt:.2f}\pm{mu_err_SB_subt:.2f}$",loc=2,color="blue")
plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)

plt.subplot(2,1,2)
plt.errorbar(x_ROI_thresh,y_ROI_thresh,yerr=yerr_ROI_thresh,fmt='.k',capsize=0)
# plt.plot(xlin_ROI_thresh,yfit_ROI_thresh,'k--')

plt.errorbar(x_SB_thresh+xoffset,y_SB_thresh,yerr=yerr_SB_thresh,fmt='.b',capsize=0)
# plt.plot(xlin_SB_thresh+xoffset,yfit_SB_thresh,'b--')



plt.ylabel("Counts")
plt.xlabel(r"$\alpha_{miss}$ [GeV]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
placeText(r"$E_{\gamma}>$"+Egamma_thresh.replace("p",".")+" GeV",loc=1)
# placeText(r"$\mu_{ROI}"+rf"={mu_ROI_thresh:.2f}\pm{mu_err_ROI_thresh:.2f}$",loc=2)
# placeText("\n"+r"$\mu_{SB}"+rf"={mu_SB_thresh:.2f}\pm{mu_err_SB_thresh:.2f}$",loc=2,color="blue")
plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)

# plt.savefig(f"../figures/p2_preB03_Emiss1/subthreshold/alpha_miss_SB.pdf")
plt.show()

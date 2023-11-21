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
histname = "alpha_miss_ROI"
sub_directoryname=".Kin.ROI"
vers="v8"
pTCut=False

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"

rebin_data=20
rebin_sim=10
xoffset=0.01
def getXY(infiles,weights,histname, rebin,directoryname):
    x=0
    y=0
    yerr=0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile,dir=directoryname)
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
simWeights=[0.366,0.069,1.13,0.29] #Not including 2H (0.242 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]


directoryname=".SubThresh"+sub_directoryname
if pTCut:
    directoryname=".SubThresh_pt03_lower"+sub_directoryname
x_data_subt,y_data_subt,yerr_data_subt= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                              histname=histname, rebin=rebin_data, directoryname=directoryname)

x_sim_subt,y_sim_subt,yerr_sim_subt= getXY(infiles=[filepath + tree for tree in simFiles], weights=simWeights,
                                           histname=histname, rebin=rebin_sim, directoryname=directoryname)

directoryname=".AboveThresh"+sub_directoryname
if pTCut:
    directoryname=".AboveThresh_pt03_lower"+sub_directoryname
x_data_thresh,y_data_thresh,yerr_data_thresh= getXY(infiles=[filepath + tree for tree in dataFiles], weights=[1, 1],
                                                    histname=histname, rebin=rebin_data, directoryname=directoryname)

x_sim_thresh,y_sim_thresh,yerr_sim_thresh= getXY(infiles=[filepath + tree for tree in simFiles], weights=simWeights,
                                                 histname=histname, rebin=rebin_sim, directoryname=directoryname)
# </editor-fold>


# <editor-fold desc="Scaling">
# Scaling
scaleFactor=sum(y_data_thresh+y_data_subt)/sum(y_sim_thresh+y_sim_subt)
scaleFactor_subt=sum(y_data_subt)/sum(y_sim_subt)
scaleFactor_thresh=sum(y_data_thresh)/sum(y_sim_thresh)
scaleFactor_subt=scaleFactor
scaleFactor_thresh=scaleFactor


y_sim_thresh *= scaleFactor_thresh*rebin_data/rebin_sim
yerr_sim_thresh *= scaleFactor_thresh*rebin_data/rebin_sim
y_sim_subt *= scaleFactor_subt*rebin_data/rebin_sim
yerr_sim_subt *= scaleFactor_subt*rebin_data/rebin_sim
# </editor-fold>

# <editor-fold desc="Fit Data">
xlin_data_subt, yfit_data_subt, N_data_subt, N_err_data_subt, \
    mu_data_subt, mu_err_data_subt, sigma_data_subt, sigma_err_data_subt \
    = fitGaussian(x=x_data_subt,y=y_data_subt,yerr=yerr_data_subt,xstart=1,xend=1.7)

xlin_sim_subt, yfit_sim_subt, N_sim_subt, N_err_sim_subt, \
    mu_sim_subt, mu_err_sim_subt, sigma_sim_subt, sigma_err_sim_subt \
    = fitGaussian(x=x_sim_subt,y=y_sim_subt,yerr=yerr_sim_subt,xstart=0.8,xend=1.7)

xlin_data_thresh, yfit_data_thresh, N_data_thresh, N_err_data_thresh, \
    mu_data_thresh, mu_err_data_thresh, sigma_data_thresh, sigma_err_data_thresh \
    = fitGaussian(x=x_data_thresh,y=y_data_thresh,yerr=yerr_data_thresh,xstart=0.8,xend=1.7)

xlin_sim_thresh, yfit_sim_thresh, N_sim_thresh, N_err_sim_thresh, \
    mu_sim_thresh, mu_err_sim_thresh, sigma_sim_thresh, sigma_err_sim_thresh \
    = fitGaussian(x=x_sim_thresh,y=y_sim_thresh,yerr=yerr_sim_thresh,xstart=0.8,xend=1.7)
# </editor-fold>

# Plotting
plt.figure(figsize=(5,6))
# Subthreshold Plots
plt.subplot(2,1,1)
plt.errorbar(x_data_subt,y_data_subt,yerr=yerr_data_subt,fmt='.k',capsize=0,label="Data")
# plt.plot(xlin_data_subt,yfit_data_subt,'k--')

plt.errorbar(x_sim_subt,y_sim_subt,yerr=yerr_sim_subt,fmt='.b',capsize=0,label="Sim.")
plt.plot(xlin_sim_subt,yfit_sim_subt,'b--')
plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
# if tracks=="noTracks":
#     placeText("No Sideband Sub."+"\n"+"Mixed"+"\n"+version+" No Extra Tracks",loc=2,yoffset=-60)
# else:
#     placeText("No Sideband Sub." + "\n" + "Mixed" + "\n" + version + " Incl. Extra Tracks", loc=2, yoffset=-60)

if pTCut:
    placeText("No Extra Tracks/Showers" + "\n" + vers +"\n"+r"  $p_T<0.3$ ", loc=1, yoffset=-60)
else:
    placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)


placeText("He+C"+"\n"+r"3<$m(e^+e^-)$<3.2",loc=2,yoffset=-40)
placeText(r"$E_{\gamma}<8.2$ GeV",loc=1)
placeText(r"$\mu_{data}"+rf"={mu_data_subt:.2f}\pm{mu_err_data_subt:.2f}$" +"\n"
          +r"$\mu_{sim}"+rf"={mu_sim_subt:.2f}\pm{mu_err_sim_subt:.2f}$",loc=2)
# plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
# plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)


plt.subplot(2,1,2)
plt.errorbar(x_data_thresh,y_data_thresh,yerr=yerr_data_thresh,fmt='.k',capsize=0)
plt.plot(xlin_data_thresh,yfit_data_thresh,'k--')

plt.errorbar(x_sim_thresh,y_sim_thresh,yerr=yerr_sim_thresh,fmt='.b',capsize=0)
plt.plot(xlin_sim_thresh,yfit_sim_thresh,'b--')


plt.ylabel("Counts")
plt.xlabel(r"$\alpha_{miss}$ [GeV]")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
placeText(r"$E_{\gamma}<8.2$ GeV",loc=1)
placeText(r"$\mu_{data}"+rf"={mu_data_thresh:.2f}\pm{mu_err_data_thresh:.2f}$" +"\n"
          +r"$\mu_{sim}"+rf"={mu_sim_thresh:.2f}\pm{mu_err_sim_thresh:.2f}$",loc=2)
# plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
# plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)

plt.savefig(f"../../files/figs/kin/subthreshold_comp/alpha_miss/alpha_miss_ROI_fitted_{vers}_{'pT03_' if pTCut else ''}mixed.pdf")

plt.show()

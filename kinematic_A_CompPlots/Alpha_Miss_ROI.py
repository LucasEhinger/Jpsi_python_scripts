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
histname = "alpha_miss_SB"
directoryname=".SB"
pTCut=""
tracks="noTracks"
version="v7"
# sideband="No Sideband Sub."
sideband="Sidebands"
rebin_data=2
rebin_sim=1
xoffset=0.01
def getXY(infiles,weights,histname, rebin):
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
simWeights=[0.242,0.366,0.069,1.13,0.29]
dataFiles=["data_hist_D.root","data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_2H_MF_helicity_mixed.root","hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root", "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]

filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{version}/PoverE/m3_p2_sigma/{tracks}/preB_03/Emiss_1/"
if pTCut!="":
    filepath+="/pTCuts/"+pTCut+"_lower/"
x_data_D,y_data_D,yerr_data_D=getXY(infiles=[filepath+tree for tree in [dataFiles[0]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_D,y_sim_D,yerr_sim_D=getXY(infiles=[filepath+tree for tree in [simFiles[0]]],
                              weights=simWeights,histname=histname,rebin=rebin_sim)

x_data_He,y_data_He,yerr_data_He=getXY(infiles=[filepath+tree for tree in [dataFiles[1]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_He,y_sim_He,yerr_sim_He=getXY(infiles=[filepath+tree for tree in simFiles[1:3]],
                              weights=simWeights[3:5],histname=histname,rebin=rebin_sim)

x_data_C,y_data_C,yerr_data_C=getXY(infiles=[filepath+tree for tree in [dataFiles[2]]],
                              weights=[1],histname=histname,rebin=rebin_data)

x_sim_C,y_sim_C,yerr_sim_C=getXY(infiles=[filepath+tree for tree in simFiles[3:5]],
                              weights=simWeights[3:5],histname=histname,rebin=rebin_sim)


# </editor-fold>


# <editor-fold desc="Scaling">
# Scaling
scaleFactor=sum(y_data_D+y_data_He+y_data_C)/sum(y_sim_D+y_sim_He+y_sim_C)
scaleFactorD=sum(y_data_D)/sum(y_sim_D)
scaleFactorHe=sum(y_data_He)/sum(y_sim_He)
scaleFactorC=sum(y_data_C)/sum(y_sim_C)


y_sim_D *= scaleFactorD*rebin_data/rebin_sim
yerr_sim_D *= scaleFactorD*rebin_data/rebin_sim
y_sim_He *= scaleFactorHe*rebin_data/rebin_sim
yerr_sim_He *= scaleFactorHe*rebin_data/rebin_sim
y_sim_C *= scaleFactorC*rebin_data/rebin_sim
yerr_sim_C *= scaleFactorC*rebin_data/rebin_sim
# </editor-fold>

# <editor-fold desc="Fit Data">
xlin_data_D, yfit_data_D, N_data_D, N_err_data_D, \
    mu_data_D, mu_err_data_D, sigma_data_D, sigma_err_data_D \
    = fitGaussian(x=x_data_D,y=y_data_D,yerr=yerr_data_D,xstart=0.8,xend=1.7)

xlin_sim_D, yfit_sim_D, N_sim_D, N_err_sim_D, \
    mu_sim_D, mu_err_sim_D, sigma_sim_D, sigma_err_sim_D \
    = fitGaussian(x=x_sim_D,y=y_sim_D,yerr=yerr_sim_D,xstart=0.8,xend=1.7)

xlin_data_He, yfit_data_He, N_data_He, N_err_data_He, \
    mu_data_He, mu_err_data_He, sigma_data_He, sigma_err_data_He \
    = fitGaussian(x=x_data_He,y=y_data_He,yerr=yerr_data_He,xstart=0.8,xend=1.7)

xlin_sim_He, yfit_sim_He, N_sim_He, N_err_sim_He, \
    mu_sim_He, mu_err_sim_He, sigma_sim_He, sigma_err_sim_He \
    = fitGaussian(x=x_sim_He,y=y_sim_He,yerr=yerr_sim_He,xstart=0.8,xend=1.7)

xlin_data_C, yfit_data_C, N_data_C, N_err_data_C, \
    mu_data_C, mu_err_data_C, sigma_data_C, sigma_err_data_C \
    = fitGaussian(x=x_data_C,y=y_data_C,yerr=yerr_data_C,xstart=0.8,xend=1.7)

xlin_sim_C, yfit_sim_C, N_sim_C, N_err_sim_C, \
    mu_sim_C, mu_err_sim_C, sigma_sim_C, sigma_err_sim_C \
    = fitGaussian(x=x_sim_C,y=y_sim_C,yerr=yerr_sim_C,xstart=0.8,xend=1.7)


# </editor-fold>

# Plotting
plt.figure(figsize=(5,8))
# Subthreshold Plots
plt.subplot(3,1,1)
plt.errorbar(x_data_D,y_data_D,yerr=yerr_data_D,fmt='.k',capsize=0,label="Data")
plt.plot(xlin_data_D,yfit_data_D,'k--')

plt.errorbar(x_sim_D,y_sim_D,yerr=yerr_sim_D,fmt='.b',capsize=0,label="Sim.")
plt.plot(xlin_sim_D,yfit_sim_D,'b--')
plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
if tracks=="noTracks":
    placeText(sideband+"\n"+"Mixed"+"\n"+version+" No Extra Tracks",loc=2,yoffset=-60)
else:
    placeText(sideband + "\n" + "Mixed" + "\n" + version + " Incl. Extra Tracks", loc=2, yoffset=-60)

if pTCut=="":
    placeText(r"$3<m(e^+e^-)<3.2$",loc=1,yoffset=-25)
else:
    placeText(r"$3<m(e^+e^-)<3.2$"
          +"\n"+r"$p_T<$"+pTCut.replace("p","."),loc=1,yoffset=-45)
# placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".") + " GeV",loc=1)
placeText(r"$\mu_{data}"+rf"={mu_data_D:.2f}\pm{mu_err_data_D:.2f}$" +"\n"
          +r"$\mu_{sim}"+rf"={mu_sim_D:.2f}\pm{mu_err_sim_D:.2f}$",loc=2)
placeText("D",loc=1)
# plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
# plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)


plt.subplot(3,1,2)
plt.errorbar(x_data_He,y_data_He,yerr=yerr_data_He,fmt='.k',capsize=0,label="Data")
plt.plot(xlin_data_He,yfit_data_He,'k--')

plt.errorbar(x_sim_He,y_sim_He,yerr=yerr_sim_He,fmt='.b',capsize=0,label="Sim.")
plt.plot(xlin_sim_He,yfit_sim_He,'b--')
plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)
# placeText(r"$E_{\gamma}<$"+Egamma_subt.replace("p",".") + " GeV",loc=1)
placeText(r"$\mu_{data}"+rf"={mu_data_He:.2f}\pm{mu_err_data_He:.2f}$" +"\n"
          +r"$\mu_{sim}"+rf"={mu_sim_He:.2f}\pm{mu_err_sim_He:.2f}$",loc=2)
placeText("He",loc=1)
# plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
# plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)


plt.subplot(3,1,3)
plt.errorbar(x_data_C,y_data_C,yerr=yerr_data_C,fmt='.k',capsize=0,label="Data")
plt.plot(xlin_data_C,yfit_data_C,'k--')

plt.errorbar(x_sim_C,y_sim_C,yerr=yerr_sim_C,fmt='.b',capsize=0,label="Sim.")
plt.plot(xlin_sim_C,yfit_sim_C,'b--')
plt.legend(loc="center right",frameon=True,fontsize=12,labelspacing=0.25,handletextpad=0)

plt.ylabel("Counts")
plt.xlim(xmin,xmax)
xmin,xmax,ymin,ymax=plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText(r"$\mu_{data}"+rf"={mu_data_C:.2f}\pm{mu_err_data_C:.2f}$" +"\n"
          +r"$\mu_{sim}"+rf"={mu_sim_C:.2f}\pm{mu_err_sim_C:.2f}$",loc=2)
placeText("C",loc=1)
# plt.axvspan(xmin, 1.2, facecolor='b', alpha=0.3)
# plt.axvspan(1.2, xmax, facecolor='yellow', alpha=0.3)
plt.xlabel(r"$\alpha_{miss}$ [GeV]")

plt.savefig(f"../figures/p2_preB03_Emiss1/AComp/alpha_miss_SB_fitted_{version}_pT{pTCut}_{tracks}_mixed.pdf")
plt.show()

# print(f"Scale difference: {scaleFactor_subt/scaleFactor_thresh: .3f}")
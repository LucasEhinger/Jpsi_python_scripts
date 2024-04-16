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
directoryname=".PoverEPlots"

f = root2mpl.File(filepath+f"hist2_pE_DSelector_4He_MF_helicity_mixed.root",dir=directoryname)
rebin=30

theta_bins=[0,3,5,8,11,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125]
theta_bins=theta_bins[0:26]
p_bins=[0.4,1.6,2.8,4.0,5.2,6.4,7.6,8.8,10.0]
p_bins=p_bins[3:8]

def fun(x,A,mu,sigma):
    return (A * norm.pdf(x,loc=mu,scale=sigma))*dx
xmin=0.8
xmax=1.2
mu=np.zeros(len(p_bins)-1)
sigma=np.zeros(len(p_bins)-1)
mu_err=np.zeros(len(p_bins)-1)
sigma_err=np.zeros(len(p_bins)-1)


for i in range(len(p_bins) - 1):
    x=0
    y=0
    yerr=0
    for j in range(len(theta_bins[:5]) - 1):
        histname = f"PoverE_{theta_bins[j]}_{theta_bins[j + 1]}_{p_bins[i]}_{p_bins[i + 1]}"
        h = f.get(histname,rebin=rebin)
        min_index=int((xmin-min(h.x))/(max(h.x)-min(h.x))*len(h.x))
        max_index=int((xmax-min(h.x))/(max(h.x)-min(h.x))*len(h.x))

        x = h.x[min_index:max_index]
        y += h.y[min_index:max_index]
        yerr = np.sqrt(yerr**2 + h.yerr[min_index:max_index]**2)
    dx = x[1]-x[0]

    p0 = [10**7,1,0.1]

    popt, pcov = curve_fit(fun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
    N = popt[0]
    mu[i] = popt[1]
    sigma[i] = popt[2]
    #
    N_err = np.sqrt(pcov[0][0])
    mu_err[i] = np.sqrt(pcov[1][1])
    sigma_err[i] = np.sqrt(pcov[2][2])




    # plt.errorbar(theta_bins[:-1],mu,yerr=mu_err,fmt='o',label=r"$\mu$")

p_avg = (np.array(p_bins[:-1])+np.array(p_bins[1:]))/2
plt.errorbar(p_avg,sigma,yerr=sigma_err,fmt='o',label=r"FCal")

for i in range(len(p_bins) - 1):
    x=0
    y=0
    yerr=0
    for j in range(4,len(theta_bins) - 1):
        histname = f"PoverE_{theta_bins[j]}_{theta_bins[j + 1]}_{p_bins[i]}_{p_bins[i + 1]}"
        h = f.get(histname,rebin=rebin)
        min_index=int((xmin-min(h.x))/(max(h.x)-min(h.x))*len(h.x))
        max_index=int((xmax-min(h.x))/(max(h.x)-min(h.x))*len(h.x))

        x = h.x[min_index:max_index]
        y += h.y[min_index:max_index]
        yerr = np.sqrt(yerr**2 + h.yerr[min_index:max_index]**2)
    yerr = np.sqrt(yerr**2 + 1)
    dx = x[1]-x[0]

    p0 = [sum(y),1,0.1]

    popt, pcov = curve_fit(fun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
    N = popt[0]
    mu[i] = popt[1]
    sigma[i] = popt[2]
    #
    N_err = np.sqrt(pcov[0][0])
    mu_err[i] = np.sqrt(pcov[1][1])
    sigma_err[i] = np.sqrt(pcov[2][2])




    # plt.errorbar(theta_bins[:-1],mu,yerr=mu_err,fmt='o',label=r"$\mu$")

p_avg = (np.array(p_bins[:-1])+np.array(p_bins[1:]))/2+0.1
plt.errorbar(p_avg,sigma,yerr=sigma_err,fmt='o',label=r"BCal")
# plt.ylim(0.5,max(sigma)*1.1)
plt.xlabel(r"$p$ [GeV]")
plt.ylabel(r"$p/E$" + r" $\sigma$")
plt.legend()
# placeText(rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"+"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")#+"\n"+r"")
# # # placeText(A+"\n"+f"{cut}",loc=2)
# # # placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
# placeText(A+" Sim"+"\n"+r"$-3\sigma$ $<$ $p/E_{FCAL}-\langle p/E_{FCAL} \rangle$ $<$ $2\sigma$" ,loc=2)
# "\n"+ "Iteration " +str(iter)+

# plt.savefig(f"../../../files/figs/cuts/{A}_bcal_sig_bkd_fit_sim.pdf", bbox_inches = 'tight')
plt.show()

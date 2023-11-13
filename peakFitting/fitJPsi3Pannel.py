#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm
import scipy.integrate as integrate

from pystyle.annotate import placeText

from scipy.optimize import curve_fit
import os

plt.style.use("SRC_CT_presentation")

A = "D"

plt.figure(figsize=(7,13))
rebin=3
jPsiMass=3.096916

def fun(x,a0,a1,A,mu, sigma):
    return (a0 + x*a1 + A * norm.pdf(x,loc=mu,scale=sigma))

def integratedfun(x,a0,a1,A,mu,sigma):
    df=[]
    for xval in x:
        df_val = integrate.quad(fun, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
        df.append(df_val[0])
    return df

p0 = [0,0,5,jPsiMass,0.1]


# <editor-fold desc="Upper">
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/9p5_upper/"
plt.subplot(3,1,3)

f = root2mpl.File(filepath+f"data_hist_{A}.root")
histname = "mass_pair"
directoryname=".JPsi"
f.setDirectory(dirName=directoryname)

plt.xlim(2.2,3.4)
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax=plt.axis()
plt.ylim(ymin,1.25*ymax)
plt.ylabel("Counts")
plt.xlabel(r"$m(e^+e^-)$ [GeV]")

h = f.get(histname,rebin=rebin)

first = int(280/rebin)
last = int(330/rebin)
x = h.x[first:last]
y = h.y[first:last]
yerr = np.sqrt(h.yerr[first:last]**2+1)

dx = x[1]-x[0]


# print(y)
# print(yerr)

popt, pcov = curve_fit(integratedfun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0, maxfev=10000)

a0 = popt[0]
a1 = popt[1]
N = popt[2]
mu = popt[3]
sigma = popt[4]

N_err = np.sqrt(pcov[2][2])
mu_err = np.sqrt(pcov[3][3])
sigma_err = np.sqrt(pcov[4][4])


JPsiTot=0.9545*N
BackgroundTot=(a0+a1*mu)*4*sigma

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,integratedfun(xlin,*popt),zorder=100,color='r')

placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
placeText(r"$9.5<E_{\gamma}<10.8$ GeV",loc=2)
# </editor-fold>

# <editor-fold desc="Middle">
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/9p5_lower/"
plt.subplot(3,1,2)

f = root2mpl.File(filepath+f"data_hist_{A}.root")
histname = "mass_pair"
directoryname=".JPsi"
f.setDirectory(dirName=directoryname)

plt.xlim(2.2,3.4)
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax=plt.axis()
plt.ylim(ymin,1.25*ymax)
plt.ylabel("Counts")

h = f.get(histname,rebin=rebin)

first = int(280/rebin)
last = int(330/rebin)
x = h.x[first:last]
y = h.y[first:last]
yerr = np.sqrt(h.yerr[first:last]**2+1)

dx = x[1]-x[0]


# print(y)
# print(yerr)

popt, pcov = curve_fit(integratedfun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0, maxfev=10000)

a0 = popt[0]
a1 = popt[1]
N = popt[2]
mu = popt[3]
sigma = popt[4]

N_err = np.sqrt(pcov[2][2])
mu_err = np.sqrt(pcov[3][3])
sigma_err = np.sqrt(pcov[4][4])


JPsiTot=0.9545*N
BackgroundTot=(a0+a1*mu)*4*sigma

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,integratedfun(xlin,*popt),zorder=100,color='r')

placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
placeText(r"$8.2 <E_{\gamma}<9.5$ GeV",loc=2)
plt.xticks([])
# </editor-fold>

# <editor-fold desc="Lower">
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/EgammaCuts/8p2_lower/"
plt.subplot(3,1,1)

f = root2mpl.File(filepath+f"data_hist_{A}.root")
histname = "mass_pair"
directoryname=".JPsi"
f.setDirectory(dirName=directoryname)

plt.xlim(2.2,3.4)
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax=plt.axis()
plt.ylim(ymin,1.25*ymax)
plt.ylabel("Counts")

h = f.get(histname,rebin=rebin)

first = int(280/rebin)
last = int(330/rebin)
x = h.x[first:last]
y = h.y[first:last]
yerr = np.sqrt(h.yerr[first:last]**2+1)

dx = x[1]-x[0]


# print(y)
# print(yerr)

popt, pcov = curve_fit(integratedfun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0, maxfev=10000)

a0 = popt[0]
a1 = popt[1]
N = popt[2]
mu = popt[3]
sigma = popt[4]

N_err = np.sqrt(pcov[2][2])
mu_err = np.sqrt(pcov[3][3])
sigma_err = np.sqrt(pcov[4][4])


JPsiTot=0.9545*N
BackgroundTot=(a0+a1*mu)*4*sigma

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,integratedfun(xlin,*popt),zorder=100,color='r')

placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
placeText(r"$7<E_{\gamma}<8.2$ GeV",loc=2)
placeText(A,loc=2,yoffset=-25)
plt.xticks([])
# </editor-fold>



plt.savefig(f"figures/p2_preB03_Emiss1/mass_pair_final/Mee_fitted_3pannel_{A}_bin3.pdf")
plt.show()
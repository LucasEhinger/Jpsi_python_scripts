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

Egamma="8p2"
histname="mass_pair"
directoryname=".JPsi"
A="He + C"

rebin=3
jPsiMass=3.096916

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

# <editor-fold desc="Get Data">
simWeights=[0.35,0.54,0.08,1.64,0.33] #Not including 2H (0.35 nb)
dataFiles=["data_hist_He.root", "data_hist_C.root"]
simFiles=["hist_DSelector_4He_MF.root", "hist_DSelector_4He_SRC.root", "hist_DSelector_12C_MF.root", "hist_DSelector_12C_SRC.root"]
simFiles=simFiles[0:2]
simFiles=["hist_DSelector_2H_MF.root"]
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/v5/PoverE/m3_p2_sigma/noTracks/preB_03/Emiss_1/EgammaCuts/8p2_lower/"
x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                              weights=[1,1],histname=histname,rebin=rebin)
# </editor-fold>


plt.figure()
plt.xlim(2.6,3.4)
plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0,marker='.')
xmin, xmax, ymin, ymax=plt.axis()
plt.ylim(0.5*ymin,1.5*ymax)
plt.ylabel("Counts")
plt.xlabel(r"$m(e^+e^-)$ [GeV]")

first = int(280/rebin)
last = int(330/rebin)
x = x_data[first:last]
y = y_data[first:last]
yerr = np.sqrt(yerr_data[first:last]**2+1)

dx = x[1]-x[0]

def fun(x,a0,a1,A,mu, sigma):
    return (a0 + x*a1 + A * norm.pdf(x,loc=mu,scale=sigma))

def integratedfun(x,a0,a1,A,mu,sigma):
    df=[]
    for xval in x:
        df_val = integrate.quad(fun, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
        df.append(df_val[0])
    return df

p0 = [0,0,7,jPsiMass-0.1,0.05]

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

# print("N: ",N)
# print("mu: ",mu)
# print("sigma: ",sigma)

print(N,N_err)
# print(mu,mu_err)
print(sigma,sigma_err)

JPsiTot=0.9545*N
BackgroundTot=(a0+a1*mu)*4*sigma

# print('Total Background: ', BackgroundTot)

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,integratedfun(xlin,*popt),color='r')

# placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"
#           +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
placeText(A+"\n"+r"$7<E_{\gamma}<8.2 GeV$"+"\n"+"No Extra Tracks"\
        +"\n"+"v05",loc=2)
# placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
# placeText(A+"\n"+f"No subtracted events",loc=2)
# placeText(A+"\n"+r"$\theta_p< 80 ^\circ$"  + "\n" + " $E_{miss \;stat}$<0.5 GeV" +"\n"+r"$E_{preBCal}$ Sin($\theta$)>0.04 GeV",loc=2)
# r"$E_{preBCal}$ Sin($\theta$)<0.04 GeV"
# + "\n" + " $E_{miss \;stat}$<2 GeV"
# +"\n"+r"$E_{preBCal}$ Sin($\theta$)>0.04 GeV"
# "$p_{miss stat}$<1 GeV"
#\n"+str(int(sum(h.y)))+" events"
# isExist = os.path.exists("figures")
# if not isExist:
#    os.makedirs("figures/p2_preB03_Emiss1/8p2/")

# plt.savefig(f"figures/p2_preB03_Emiss1/mass_pair_final/Mee_fitted_{A}_v05_noTracksrebin1.pdf")
plt.show()
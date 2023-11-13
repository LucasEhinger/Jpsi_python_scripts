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

# filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE/m3_p2_sigma/preB_03/Emiss_1/"
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/v7/PoverE/m3_p2_sigma/noTracks/preB_03/Emiss_1/"

f = root2mpl.File(filepath+f"data_hist_{A}.root")
rebin=4

histname = "mass_pair"
directoryname=".JPsi"
f.setDirectory(dirName=directoryname)
jPsiMass=3.096916

plt.figure()
plt.xlim(2.2,3.7)
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
xmin, xmax, ymin, ymax=plt.axis()
plt.ylim(ymin,1.25*ymax)
plt.ylabel("Counts")
plt.xlabel(r"$m(e^+e^-)$ [GeV]")

h = f.get(histname,rebin=rebin)

first = int(260/rebin)
last = int(340/rebin)
x = h.x[first:last]
y = h.y[first:last]
yerr = np.sqrt(h.yerr[first:last]**2+1)

dx = x[1]-x[0]

def fun(x,a0,a1,A,mu, sigma):
    return (a0*10**6 * np.exp(a1*x) + A * norm.pdf(x,loc=mu,scale=sigma))

def integratedfun(x,a0,a1,A,mu,sigma):
    df=[]
    for xval in x:
        df_val = integrate.quad(fun, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
        df.append(df_val[0])
    return df

p0 = [13,-4,40,jPsiMass,0.04]

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
print(mu,mu_err)
print(sigma,sigma_err)

JPsiTot=0.9545*N
BackgroundTot=(a0+a1*mu)*4*sigma

# print('Total Background: ', BackgroundTot)

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,integratedfun(xlin,*popt),zorder=100,color='r')
# plt.plot(xlin,integratedfun(xlin,*p0),zorder=100,color='r')

# placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"
#           +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
# placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
#           +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
# placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
# placeText(A+"\n"+"No Extra Tracks"+"\n"+"v07",loc=2)
# placeText(A+"\n"+r"$E_{\gamma}>8.2$ GeV",loc=2)
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

# plt.savefig(f"figures/p2_preB03_Emiss1/mass_pair_final/integratedFits/trackComp/Mee_fitted_{A}_noExtraTrack_v7.pdf")
plt.show()
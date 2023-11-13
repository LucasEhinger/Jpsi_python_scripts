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

# A = ["D", "He", "C"]
A="He"
filepath="/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/noTargetHits/xyCuts/"

f = root2mpl.File(filepath+f"data_hist_{A}.root")


rebin=5

histname = "Z-location"

plt.figure()
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
# plt.xlim(35,40)
# plt.xlim(80,83)
plt.xlim(83,86.5)
plt.ylim(-0.1*rebin,45*rebin)
plt.xlabel(r"z Vertex [cm]")
plt.ylabel("Counts")

h = f.get(histname,rebin=rebin)
minx=84.2
maxx=85.2
# minx=81.
# maxx=82.2
# minx=36
# maxx=38.5
diff=max(h.x)-min(h.x)

first = int(len(h.x/rebin)*(minx-min(h.x))/diff)
last = int(len(h.x/rebin)*(maxx-min(h.x))/diff)
x = h.x[first:last]
y = h.y[first:last]
yerr = np.sqrt(h.yerr[first:last]**2+1)

dx = x[1]-x[0]

def fun(x,A,mu,sigma):
    return A * norm.pdf(x,loc=mu,scale=sigma)*dx

p0 = [200,84,1]

# print(y)
# print(yerr)

popt, pcov = curve_fit(fun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0, maxfev=10000)

N = popt[0]
mu = popt[1]
sigma = popt[2]

N_err = np.sqrt(pcov[0][0])
mu_err = np.sqrt(pcov[1][1])
sigma_err = np.sqrt(pcov[2][2])

print("N: ",N)
print("mu: ",mu)
print("sigma: ",sigma)

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin,fun(xlin,*popt),zorder=100,color='b')

placeText(r"$A$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")  # ##+"\n"+r"")
# placeText(A+"\n"+f"{cut}",loc=2)
# placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
# placeText(A+"\n"+f"No subtracted events",loc=2)
placeText(A,loc=2)
#\n"+str(int(sum(h.y)))+" events"
isExist = os.path.exists(filepath+"figures")
if not isExist:
   os.makedirs(filepath+"figures")

plt.savefig(filepath+"figures/"+f"zVertex_84_fitted_{A}_noNegs.png",dpi=1000)
plt.show()
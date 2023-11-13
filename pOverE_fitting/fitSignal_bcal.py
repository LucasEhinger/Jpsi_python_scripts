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

iter=5
plt.style.use("SRC_CT_presentation")

file= "/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE_Fitting/preB_03/"
A="D"
f = root2mpl.File(f"{file}/iter_{iter}/fcal_cut/data_tree_{A}.root")


rebin=30

histname = "bcal_PoverE"

plt.figure(figsize=(8,5))
f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
plt.xlim(0.7,1.5)



plt.xlabel(r"BCAL p/E")
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


# print(y)
# print(yerr)
# No background sub
# polyparam_D = [692.85610611, -3664.19021488,  6964.93306998, -5531.43883289, 1563.91791482]
# polyparam_He = [ -3896.04433747,  16235.76180499, -24973.7009865 ,  16962.44434297, -4298.46918862]
# polyparam_C = [ 4151.15068832, -18604.43565006,  30830.78119648, -22218.20361439, 5873.25499144]

#Background Sub preB Cut 03 4v 3.5-5.5
# polyparam_D = [ -1971.37669946,   8550.45010341, -13747.78683495,   9745.16073673, -2568.78527671]
# polyparam_He = [ -2573.86857946,  10834.20539306, -16887.49930606,  11596.8722719 , -2961.50039745]
# polyparam_C = [ -2116.90291358,   8967.74183493, -14073.63797011,   9745.26378838, -2514.20331149]

polyparam_D = [  769.23154584, -3151.83345908,  4760.55972791, -3111.81075902, 741.27448886]
polyparam_He = [ -2031.60579772,   8556.33555922, -13344.11118059,   9165.9191112 ,  -2340.34625657]
polyparam_C = [ 1215.60354244, -5200.24473647,  8235.19846525, -5686.65903366, 1443.65161087]

polyparam=None
if A=="D":
    polyparam=polyparam_D
elif A=="He":
    polyparam=polyparam_He
elif A=="C":
    polyparam=polyparam_C


bkgd = np.poly1d(polyparam)
def fun(x,A,B,mu,sigma):
    return (A * norm.pdf(x,loc=mu,scale=sigma)+B*bkgd(x))*dx

p0 = [0,50,1,0.1]
popt, pcov = curve_fit(fun,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
N = popt[0]
B = popt[1]
mu = popt[2]
sigma = popt[3]
#
N_err = np.sqrt(pcov[0][0])
mu_err = np.sqrt(pcov[2][2])
sigma_err = np.sqrt(pcov[3][3])
#
# print("N: ", N)
# print("mu: ", mu)
# print("sigma: ", sigma)
print(mu)
print(sigma)

bkd_file= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE_Fitting/preB_03/background_3p5_5p5/fcal_cut/data_tree_{A}.root"
f2 = root2mpl.File(bkd_file)
h2 = f2.get(histname,rebin=rebin)
x_bkd = h2.x+0.01
y_bkd = h2.y*popt[1]*dx
yerr_bkd = np.sqrt((h2.yerr)**2+1)*popt[1]*dx
plt.errorbar(x_bkd,y_bkd,yerr=yerr_bkd,fmt='b.',capsize=0)

xlin = np.linspace(x[0],x[-1],num=1000)

plt.plot(xlin, fun(xlin,*popt),zorder=100,color='r')
plt.plot(xlin,bkgd(xlin)*popt[1]*dx,zorder=100,color='b')

xmin, xmax, ymin, ymax= plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText(rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"+"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")#+"\n"+r"")
# # placeText(A+"\n"+f"{cut}",loc=2)
# # placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
placeText(A+"\n"+r"$-3\sigma$ $<$ $p/E_{FCAL}-\langle p/E_{FCAL} \rangle$ $<$ $2\sigma$" ,loc=2)
# "\n"+ "Iteration " +str(iter)+

plt.savefig(f"figures/pOverE/{A}_bcal_sig_bkd_fit_iter_{iter}.pdf")
plt.show()


# exit()
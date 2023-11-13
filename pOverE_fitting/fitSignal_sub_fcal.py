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
f = root2mpl.File(f"{file}/iter_{iter}/bcal_cut/data_tree_{A}.root")


rebin=30

histname = "fcal_PoverE"

plt.figure(figsize=(8,5))
# f.plotPoints(histname,rebin=rebin,ls='',color='k',capsize=0,marker='.')
plt.xlim(0.7,1.5)



plt.xlabel(r"FCAL p/E")
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
#No background sub
# polyparam_D = [692.85610611, -3664.19021488,  6964.93306998, -5531.43883289, 1563.91791482]
# polyparam_He = [ -3896.04433747,  16235.76180499, -24973.7009865 ,  16962.44434297, -4298.46918862]
# polyparam_C = [ 4151.15068832, -18604.43565006,  30830.78119648, -22218.20361439, 5873.25499144]

#Background Sub preB Cut 03 4v 3.5-5.5
# polyparam_D = [  390.22537398, -1798.49254489,  2954.82635783, -2029.8453727 ,  493.7326465 ]
# polyparam_He = [  391.48755848, -2076.07215969,  3882.97867018, -3038.20510082, 849.43682808]
# polyparam_C = [ 1210.21542329, -5252.48310654,  8326.63683818, -5675.70063555, 1404.70942793]

polyparam_D = [  2.09449457, -43.34304982,  36.48644061,  81.76116897,  -67.69642232]
polyparam_He = [  553.16372693, -2636.57940353,  4570.02688344, -3392.29947197, 913.1889841 ]
polyparam_C = [  753.85962932, -3197.20954399,  4948.21536204, -3284.00888154, 788.4706425 ]

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
print(mu_err)
print(sigma)
print(sigma_err)

xlin = np.linspace(x[0],x[-1],num=1000)

bkd_file= f"/Users/lucasehinger/CLionProjects/untitled/Files/JPsi2/files/PoverE_Fitting/preB_03/background_3p5_5p5/bcal_cut/data_tree_{A}.root"
f2 = root2mpl.File(bkd_file)
h2 = f2.get(histname,rebin=rebin)
x_bdk = h2.x[min_index:max_index]
y_bkd = h2.y[min_index:max_index]
yerr_bkd = np.sqrt(h2.yerr[min_index:max_index]**2+1)


yscaled=y-y_bkd*popt[1]*dx
yerr+=yerr_bkd

plt.errorbar(x,yscaled,yerr=yerr,fmt='k.')

plt.plot(xlin, N*norm.pdf(xlin,loc=mu,scale=sigma)*dx,zorder=100,color='r')
plt.plot([0.5,1.5],[0,0],'k--')
# plt.plot(xlin,bkgd(xlin)*popt[1]*dx,zorder=100,color='b')

xmin, xmax, ymin, ymax= plt.axis()
plt.ylim(ymin,ymax*1.3)

placeText(rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"+"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")#+"\n"+r"")
# # placeText(A+"\n"+f"{cut}",loc=2)
# # placeText(A+"\n"+f"|E/p-1|<{float(cut)/10} events",loc=2)
placeText(A+"\n"+r"$-3\sigma$ $<$ $p/E_{BCAL}-\langle p/E_{BCAL}\rangle < 2\sigma$" ,loc=2)
# "\n"+ "Iteration " +str(iter)+

plt.savefig(f"figures/pOverE/{A}_fcal_sig_fit_iter_{iter}.pdf")
plt.show()


# exit()
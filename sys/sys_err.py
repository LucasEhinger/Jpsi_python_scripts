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

# N=[7.2,27.1,15.8]
# cuts=[22, 12, 15]
# fitting=[16, 7, 10]

N=[7.2,27.1,15.8,40.3, 45.8]
cuts=[22, 12, 15, 9, 10]
fitting=[16, 7, 10, 8, 6]

def fun(x,A,B):
    return np.sqrt(A+B/x)

popt, pcov = curve_fit(fun,N,cuts)
A=popt[0]
B=popt[1]
A_err = np.sqrt(pcov[0][0])
B_err = np.sqrt(pcov[1][1])
print("Cuts")
print(f"A = {A:.2f} +- {A_err:.2f}")
print(f"B = {B:.2f} +- {B_err:.2f}")

x=np.linspace(min(N)*0.9,max(N)*1.1,100)
plt.plot(N,cuts,'ko')
plt.plot(x,fun(x,*popt))
plt.xlabel(r"$N_{J/\psi}$")
plt.ylabel("Cut Uncertainty [%]")
# plt.savefig(f"../../files/figs/sys/He_sys_cuts.pdf", bbox_inches = 'tight')
plt.show()


popt, pcov = curve_fit(fun,N,fitting)
A=popt[0]
B=popt[1]
A_err = np.sqrt(pcov[0][0])
B_err = np.sqrt(pcov[1][1])
print("Fitting")
print(f"A = {A:.2f} +- {A_err:.2f}")
print(f"B = {B:.2f} +- {B_err:.2f}")

plt.plot(N,fitting,'ko')
plt.plot(x,fun(x,*popt))
plt.xlabel(r"$N_{J/\psi}$")
plt.ylabel("Fitting Uncertainty [%]")
# plt.savefig(f"../../files/figs/sys/He_sys_fitting.pdf", bbox_inches = 'tight')
plt.show()
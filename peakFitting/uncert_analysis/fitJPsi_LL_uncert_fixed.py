#!/usr/bin/env python3.10


# import numpy as np
import autograd.numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm
import scipy.integrate as integrate
from autograd import grad, jacobian, hessian
import numpy.linalg as lin
import pandas as pd

from pystyle.annotate import placeText

from scipy.optimize import curve_fit, minimize, basinhopping
import os

plt.style.use("SRC_CT_presentation")

A = "C"
vers="v8"
rebin=30
histname="mass_pair"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3
x_fit_min=2.6
x_fit_max=3.3

mu_vals=[3.082, 3.069, 3.041]
sigma_vals=[0.03,0.031,0.053]


def getXY(infiles, weights, histname, rebin,directoryname):
    x = 0
    y = 0
    yerr = 0
    for i, infile in enumerate(infiles):
        f = root2mpl.File(infile, dir=directoryname)
        h = f.get(histname, rebin=rebin)
        x = h.x
        y += h.y * weights[i]
        yerr = np.sqrt(yerr ** 2 + (h.yerr * weights[i]) ** 2)
    return x, y, yerr
# for A in ["D", "He", "C"]:
for A in ["He", "C"]:
    if A == "D":
        mu = mu_vals[0]
        sigma = sigma_vals[0]
    if A == "He":
        mu = mu_vals[1]
        sigma = sigma_vals[1]
    if A == "C":
        mu = mu_vals[2]
        sigma = sigma_vals[2]

    filepath = f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/cutVary/noTrackShower/loose_medium/"

    range_vals=("loc0","loc1","loc2","loc3","loc4")
    df = pd.DataFrame(columns=['preB','sigma_min','sigma_max','E_miss','N', 'N_err'])

    for preB in range_vals:
        for sigma_min in range_vals:
            for sigma_max in range_vals:
                for E_miss in range_vals:
                    # <editor-fold desc="Get Data">


                    directoryname = f".preB_{preB}_sigma_{sigma_min}_{sigma_max}_Emiss_{E_miss}.Jpsi_mass"
                    dataFiles = [f"data_hist_cutVary_{A}.root"]
                    #
                    # dataFiles=[f"data_hist_{A}.root"]
                    x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                                  weights=[1],histname=histname,rebin=rebin,directoryname=directoryname)
                    dx = x_data[1]-x_data[0]

                    def gaus_bdk_exp(x,a0,a1,A):
                        return (a0*10**6*np.exp(x*a1) + A * norm.pdf(x,loc=mu,scale=sigma))

                    def integrated_gaus_bkd_exp(x,a0,a1,A):
                        df=[]
                        for xval in x:
                            df_val = integrate.quad(gaus_bdk_exp, xval - dx / 2, xval + dx / 2, args=(a0, a1, A),epsrel=0.01)
                            df.append(df_val[0])
                        return df
                    # </editor-fold>

                    # <editor-fold desc="Binned Exp">
                    first = int((x_fit_min-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
                    last = int((x_fit_max-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
                    x = x_data[first:last]
                    y = y_data[first:last]
                    yerr = np.sqrt(yerr_data[first:last]**2+1)

                    p0 = [10**2,-6,20]
                    try:
                        popt, pcov = curve_fit(integrated_gaus_bkd_exp,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
                        # print(popt)

                        a0 = popt[0]
                        a1 = popt[1]
                        N = popt[2]

                        N_err = np.sqrt(pcov[2][2])
                        # </editor-fold>

                        # <editor-fold desc="Unbinned Fit">
                        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                                      weights=[1],histname=histname,rebin=1,directoryname=directoryname)
                        x_points=[]
                        w_points=[]
                        x_fit=[]
                        w_fit=[]
                        for i,yval in enumerate(y_data):
                            if yval!=0:
                                x_points=np.append(x_points,x_data[i])
                                w_points=np.append(w_points,yval)
                                if x_data[i]<x_fit_max and x_data[i]>x_fit_min:
                                    x_fit = np.append(x_fit, x_data[i])
                                    w_fit = np.append(w_fit, yval)

                        # Gaussian fit function
                        def gaus_exp_bdk_pdf(x,a0,a1):
                            pdf_val=(1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0 *a1* np.exp(a1*x)/(np.exp(x_fit_max*a1)-np.exp(x_fit_min*a1))) / (1 + a0)
                            return pdf_val

                        def minus_log_likelihood(params):
                            a0,a1,A= params
                            A=abs(A)
                            a0=abs(a0)
                            tmp=w_fit*np.log(A*gaus_exp_bdk_pdf(x_fit,a0,a1))
                            return A-tmp.sum()

                        initial_guess = [popt[0]/(popt[2]*popt[1])*(np.exp(popt[1]*x_fit_max)-np.exp(popt[1]*x_fit_min)),popt[1],20]
                        result = minimize(minus_log_likelihood, initial_guess, method = 'BFGS')#, options=dict(maxiter=10000000)

                        popt = result.x
                        hessian_ = hessian(minus_log_likelihood)
                        pcov = lin.inv(hessian_(popt))


                        N = popt[2]/(1+popt[0])

                        N_err = (np.sqrt(pcov[2][2])/popt[2]+np.sqrt(pcov[0][0])/(1+popt[0]))*N

                        # x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                        #                               weights=[1],histname=histname,rebin=rebin)
                        #
                        # xlin = np.linspace(x_fit[0],x_fit[-1],num=1000)
                        # # plt.subplot(3,1,3)
                        # plt.plot(xlin,(popt[2]*gaus_exp_bdk_pdf(xlin,*popt[0:2],*popt[3:5]))*dx,color='r')
                        # plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0)
                        #
                        # plt.xlim(2.2,3.4)
                        # xmin, xmax, ymin, ymax=plt.axis()
                        # plt.ylim(ymin,ymax)
                        # plt.ylabel("Counts")
                        # plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")
                        #
                        # placeText("No Extra Tracks/Showers" +"\n"+vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
                        # placeText(A, loc=2, yoffset=-30, fontsize=18)
                        #
                        # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                        #           +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
                        # # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
                        # placeText("Unbinned",loc=2)
                        # # </editor-fold>
                        #
                        # # plt.savefig(f"../../files/figs/peakFits/unbinned/AboveThresh/Mee_{A}_AboveThresh_noTrackShower_{vers}_bin{rebin}.pdf")
                        # plt.show()

                        df.loc[len(df.index)] = [preB, sigma_min, sigma_max,E_miss,N,N_err]
                    except:
                        df.loc[len(df.index)] = [preB, sigma_min, sigma_max, E_miss, np.nan, np.nan]

    df.to_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_7_11/peak_params_fixed_{A}_data.csv",index=False)
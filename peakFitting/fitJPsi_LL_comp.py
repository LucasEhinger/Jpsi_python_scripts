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

from pystyle.annotate import placeText

from scipy.optimize import curve_fit, minimize, basinhopping
import os

plt.style.use("SRC_CT_presentation")

A = "D"
vers="v8"
rebin=40
directoryname=".JPsi"
histname="mass_pair_fine"
x_fit_min=2.6
x_fit_max=3.3

# for A in ["D","He","C"]:
#     for vers in ["v5","v7"]:
for A in ["C"]:
    for vers in ["v7"]:
        # <editor-fold desc="Get Data">
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

        # filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/PoverE/m3_p2_sigma/noTracks/preB_03/Emiss_1/pTCuts/0p3_lower/"
        # filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/PoverE/m3_p2_sigma/noTracks/preB_03/Emiss_1/"
        filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/PoverE2/m3_p2_sigma/noTracks/preB_03/Emiss_1/"

        dataFiles=[f"data_hist_{A}.root"]
        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                      weights=[1],histname=histname,rebin=rebin)
        dx = x_data[1]-x_data[0]

        def gaus_bdk_lin(x,a0,a1,A,mu, sigma):
            return (a0 + x*a1 + A * norm.pdf(x,loc=mu,scale=sigma))
        def integrated_gaus_bkd_lin(x,a0,a1,A,mu,sigma):
            df=[]
            for xval in x:
                df_val = integrate.quad(gaus_bdk_lin, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
                df.append(df_val[0])
            return df
        def gaus_bdk_exp(x,a0,a1,A,mu, sigma):
            return (a0*10**6*np.exp(x*a1) + A * norm.pdf(x,loc=mu,scale=sigma))

        def integrated_gaus_bkd_exp(x,a0,a1,A,mu,sigma):
            df=[]
            for xval in x:
                df_val = integrate.quad(gaus_bdk_exp, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
                df.append(df_val[0])
            return df
        # </editor-fold>

        # <editor-fold desc="Binned Linear">
        plt.figure(figsize=(6,10))
        plt.subplot(3,1,1)
        plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0,marker='.')
        first = int((2.8-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        last = int((3.3-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        x = x_data[first:last]
        y = y_data[first:last]
        yerr = np.sqrt(yerr_data[first:last]**2+1)

        p0 = [0,0,40,3.1,0.04]
        popt, pcov = curve_fit(integrated_gaus_bkd_lin,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)

        a0 = popt[0]
        a1 = popt[1]
        N = popt[2]
        mu = popt[3]
        sigma = popt[4]

        N_err = np.sqrt(pcov[2][2])
        mu_err = np.sqrt(pcov[3][3])
        sigma_err = np.sqrt(pcov[4][4])

        xlin = np.linspace(2.8,3.3,num=1000)
        plt.plot(xlin,integrated_gaus_bkd_lin(xlin,*popt),zorder=100,color='r')
        placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
        plt.xlim(2.2,3.4)
        xmin, xmax, ymin, ymax=plt.axis()
        plt.ylim(ymin,1.25*ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"$m(e^+e^-)$ [GeV]")
        placeText("Binned Linear",loc=2)
        placeText("No Extra Tracks"+"\n"+vers,loc=1,yoffset=-60)#+"\n"+"pT<0.3"
        placeText(A,loc=2,yoffset=-30,fontsize=18)
        # </editor-fold>

        # <editor-fold desc="Binned Exp">
        plt.subplot(3,1,2)
        plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0,marker='.')
        first = int((x_fit_min-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        last = int((x_fit_max-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        x = x_data[first:last]
        y = y_data[first:last]
        yerr = np.sqrt(yerr_data[first:last]**2+1)

        p0 = [10**2,-6,20,3.1,0.04]
        popt, pcov = curve_fit(integrated_gaus_bkd_exp,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
        print(popt)

        a0 = popt[0]
        a1 = popt[1]
        N = popt[2]
        mu = popt[3]
        sigma = popt[4]

        N_err = np.sqrt(pcov[2][2])
        mu_err = np.sqrt(pcov[3][3])
        sigma_err = np.sqrt(pcov[4][4])

        xlin = np.linspace(x_fit_min,x_fit_max,num=1000)
        plt.plot(xlin,integrated_gaus_bkd_exp(xlin,*popt),zorder=100,color='r')
        placeText(r"$N_{J/\psi}$" + rf"$={N:.1f}\pm{N_err:.1f}$" + "\n" + rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  + "\n" + rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
        plt.xlim(2.2,3.4)
        xmin, xmax, ymin, ymax=plt.axis()
        plt.ylim(ymin,1.25*ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"$m(e^+e^-)$ [GeV]")
        placeText("Binned Exp",loc=2)
        # </editor-fold>

        # <editor-fold desc="Unbinned Fit">
        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                      weights=[1],histname=histname,rebin=1)
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
        def gaus_exp_bdk_pdf(x,a0,a1,mu,sigma):
            pdf_val=(1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0 *a1* np.exp(a1*x)/(np.exp(x_fit_max*a1)-np.exp(x_fit_min*a1))) / (1 + a0)
            return pdf_val

        def minus_log_likelihood(params):
            a0,a1,A,mu,sigma= params
            A=abs(A)
            a0=abs(a0)
            tmp=w_fit*np.log(A*gaus_exp_bdk_pdf(x_fit,a0,a1,mu,sigma))
            return A-tmp.sum()
            # return -(np.sum(np.log((S*gaus(N,mu,sigma)+B*np.ones(len(N))))))+S+B # Here "A" is the total integral of the distribution funciton, whatever that may be
            #return -np.sum(np.log(gaus(m,A,mu,sigma))) + A
            # return -(np.sum(np.log(gaus(m,A,mu,sigma))) - 0.2*np.sum(np.log(gaus(n,A,mu,sigma)))) + A


        # initial_guess = [1/N,-4,40,3.08,0.04]
        initial_guess = [popt[0]/(popt[2]*popt[1])*(np.exp(popt[1]*x_fit_max)-np.exp(popt[1]*x_fit_min)),popt[1],20,3.1,0.04]
        # initial_guess=[ 9.68114430e-01, -5.06355476e+00,  9.40306056e+01,  3.04613395e+00, 5.00839801e-02]
        result = minimize(minus_log_likelihood, initial_guess, method = 'BFGS')#, options=dict(maxiter=10000000)

        popt = result.x
        # pcov = result.hess_inv
        hessian_ = hessian(minus_log_likelihood)
        pcov = lin.inv(hessian_(popt))


        N = popt[2]/(1+popt[0])
        mu = popt[3]
        sigma = popt[4]

        N_err = np.sqrt(pcov[2][2]/(1+popt[0])**2+pcov[0][0]*N**2/(1+popt[0])**2)
        mu_err = np.sqrt(pcov[3][3])
        sigma_err = np.sqrt(pcov[4][4])

        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                      weights=[1],histname=histname,rebin=rebin)

        xlin = np.linspace(x_fit[0],x_fit[-1],num=1000)
        plt.subplot(3,1,3)
        plt.plot(xlin,(popt[2]*gaus_exp_bdk_pdf(xlin,*popt[0:2],*popt[3:5]))*dx,color='r')
        plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0)

        plt.xlim(2.2,3.4)
        xmin, xmax, ymin, ymax=plt.axis()
        plt.ylim(ymin,1.25*ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"$m(e^+e^-)$ [GeV]")


        placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
        # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
        placeText("Unbinned Exp",loc=2)
        # </editor-fold>

        # plt.savefig(f"figures/p2_preB03_Emiss1/mass_pair_final/mass_pair_comp/Mee_comp_{A}_noExtraTrack_{vers}_bin30.pdf")
        plt.show()




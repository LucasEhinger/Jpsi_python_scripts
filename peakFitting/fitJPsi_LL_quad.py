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


cut ="All"
# cut ="AboveThresh"

A = "D"
rebin=30
# directoryname=".SubThresh.Kin.Jpsi_mass"
directoryname=".All.Kin.Jpsi_mass"
if cut=="ptCut":
    directoryname=".All_pt03_lower.Kin.Jpsi_mass"
if cut=="alphaCut":
    directoryname=".All_alpha1p2_lower.Kin.Jpsi_mass"
if cut=="ptAlphaCut":
    directoryname=".All_pt_alpha.Kin.Jpsi_mass"

histname="mass_pair"
# histname="mass_kin"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3
x_fit_min=2.6
x_fit_max=3.3
# for A in ["D","He","C"]:
#     for vers in ["v5","v7"]:
for A in ["He"]:
    # for vers in ["v5", "v7", "v8"]:
    for vers in ["v8"]:
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

        filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"
        # filepath = f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/cutVary/"
        # directoryname = ".preBmin_sigmamin_min_Emissmin.Jpsi_mass"
        # dataFiles = [f"data_tree_{A}.root"]
        #
        dataFiles=[f"data_hist_{A}.root"]
        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                      weights=[1],histname=histname,rebin=rebin)
        dx = x_data[1]-x_data[0]

        def gaus_bdk_quad(x,a0,a1,a2,A,mu, sigma):
            return (a0 + a1*x + a2*x**2 + A * norm.pdf(x,loc=mu,scale=sigma))

        def integrated_gaus_bkd_quad(x,a0,a1,a2,A,mu,sigma):
            df=[]
            for xval in x:
                df_val = integrate.quad(gaus_bdk_quad, xval - dx / 2, xval + dx / 2, args=(a0, a1, a2, A, mu, sigma),epsrel=0.01)
                df.append(df_val[0])
            return df
        # </editor-fold>

        # <editor-fold desc="Binned Exp">
        first = int((x_fit_min-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        last = int((x_fit_max-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
        x = x_data[first:last]
        y = y_data[first:last]
        yerr = np.sqrt(yerr_data[first:last]**2+1)

        p0 = [10**2,-10,5,40,3.1,0.04]
        popt, pcov = curve_fit(integrated_gaus_bkd_quad,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
        # print(popt)

        a0 = popt[0]
        a1 = popt[1]
        a3 = popt[2]
        N = popt[3]
        mu = popt[4]
        sigma = popt[5]

        N_err = np.sqrt(pcov[3][3])
        mu_err = np.sqrt(pcov[4][4])
        sigma_err = np.sqrt(pcov[5][5])
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
        def gaus_quad_bdk_pdf(x,a0,a1,a2,mu,sigma):
            pdf_val=(1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0*(1+a1*x+a2*x**2)/((x_fit_max-x_fit_min)+a1*(x_fit_max**2-x_fit_min**2)/2+a2*(x_fit_max**3-x_fit_min**3)/3)) / (1 + a0)
            return pdf_val

        def minus_log_likelihood(params):
            norm_val, a0,a1,a2, mu,sigma= params
            norm_val=abs(norm_val)
            likelihood_vals=gaus_quad_bdk_pdf(x_fit,a0,a1,a2,mu,sigma)
            tmp=w_fit*np.log(abs(norm_val*likelihood_vals))*np.sign(likelihood_vals)
            return norm_val-tmp.sum()
            # return -(np.sum(np.log((S*gaus(N,mu,sigma)+B*np.ones(len(N))))))+S+B # Here "A" is the total integral of the distribution funciton, whatever that may be
            #return -np.sum(np.log(gaus(m,A,mu,sigma))) + A
            # return -(np.sum(np.log(gaus(m,A,mu,sigma))) - 0.2*np.sum(np.log(gaus(n,A,mu,sigma)))) + A


        poly_norm=(x_fit_max - x_fit_min) + popt[1] * (x_fit_max ** 2 - x_fit_min ** 2) / 2 + popt[2] * (x_fit_max ** 3 - x_fit_min ** 3) / 3
        initial_guess=[0,0,0,0,0,0]
        initial_guess[0]=0.0001 #norm value
        initial_guess[1]=1/(500/poly_norm/popt[0]-1) #a0
        initial_guess[2] = popt[1] * poly_norm / popt[0] * (1 + initial_guess[1]) #a1
        initial_guess[3] = popt[2] * poly_norm / popt[0] * (1 + initial_guess[1]) #a2
        # initial_guess[4]=popt[3]*(1+initial_guess[1])
        initial_guess[4]=popt[4]
        initial_guess[5]=popt[5]
        initial_guess=[300, 3, -0.25, 0, popt[4], popt[5]]
        initial_guess=[ 1.61358485e+02,  1.65556115e+00, -6.35625515e-01,  1.01115462e-01, 3.04147623e+00,  4.01545206e-02]
        # initial_guess=[160,1.5,-0.5,0.1,3.06,popt[5]]
        # initial_guess = [popt[3]*(1+popt[0]),popt[0]/(1-popt[0])*poly_norm,popt[1]/popt[0]*poly_norm,popt[2]/popt[0]*poly_norm,popt[4],popt[5]]
        # initial_guess = [1/N,-4,40,3.08,0.04]
        # initial_guess = [popt[0]/(popt[2]*popt[1])*(np.exp(popt[1]*x_fit_max)-np.exp(popt[1]*x_fit_min)),popt[1],20,3.1,0.04]
        # initial_guess=[ 9.68114430e-01, -5.06355476e+00,  9.40306056e+01,  3.04613395e+00, 5.00839801e-02]
        result = minimize(minus_log_likelihood, initial_guess, method = 'BFGS')#, options=dict(maxiter=10000000)

        popt = result.x
        # pcov = result.hess_inv
        hessian_ = hessian(minus_log_likelihood)
        pcov = lin.inv(hessian_(popt))


        N = popt[0]/(1+popt[1])
        mu = popt[4]
        sigma = abs(popt[5])

        N_err = np.sqrt(pcov[0][0]/(1+popt[1])**2+pcov[1][1]*N**2/(1+popt[1])**2)
        mu_err = np.sqrt(pcov[4][4])
        sigma_err = np.sqrt(pcov[5][5])

        x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                      weights=[1],histname=histname,rebin=rebin)

        xlin = np.linspace(x_fit[0],x_fit[-1],num=1000)
        # plt.subplot(3,1,3)
        plt.plot(xlin,(popt[0]*gaus_quad_bdk_pdf(xlin,*popt[1:]))*dx,color='r')
        plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0)

        plt.xlim(2.2,3.4)
        xmin, xmax, ymin, ymax=plt.axis()
        plt.ylim(ymin,ymax)
        plt.ylabel("Counts")
        # plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")
        plt.xlabel(r"Measured m($e^+e^-$) [GeV]")

        placeText("No Extra Tracks/Showers" +"\n"+vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
        placeText(A, loc=2, yoffset=-30, fontsize=18)

        placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  +"\n"+rf"$\sigma={sigma:.3f}\pm{sigma_err:.3f}$")
        # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
        placeText("Unbinned",loc=2)
        # </editor-fold>

        # plt.savefig(f"../../files/figs/peakFits/unbinned/{cut}/Mee_{A}_noTrackShower_{vers}_bin{rebin}.pdf")
        plt.savefig(f"../../files/figs/peakFits/quad/{cut}/Mee_{A}_quad_noTrackShower_{vers}_bin{rebin}.pdf")
        plt.show()

        print(popt[1])




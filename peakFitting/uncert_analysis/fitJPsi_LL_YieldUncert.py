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
histname="mass_pair_Egamma_7_8p2"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3
x_fit_min=2.6
x_fit_max=3.3
x_ROI_max=3.2
x_ROI_min=2.9


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
for A in ["C"]:
    filepath = f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/cutVary/noTrackShower/loose_medium/"
    if A=="D":
        x_ROI_min = 3.0
    # <editor-fold desc="Get Data">

    directoryname = f".preB_loc2_sigma_loc2_loc2_Emiss_loc2.Jpsi_mass"
    dataFiles = [f"data_hist_cutVary_{A}.root"]

    x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                  weights=[1],histname=histname,rebin=rebin,directoryname=directoryname)
    dx = x_data[1]-x_data[0]

    def gaus_bdk_exp(x,a0,a1,A,mu, sigma):
        return (a0*10**6*np.exp(x*a1) + A * norm.pdf(x,loc=mu,scale=sigma))

    def integrated_gaus_bkd_exp(x,a0,a1,A,mu,sigma):
        df=[]
        for xval in x:
            df_val = integrate.quad(gaus_bdk_exp, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),epsrel=0.01)
            df.append(df_val[0])
        return df
    # </editor-fold>

    # <editor-fold desc="Binned Exp">
    first = int((x_fit_min-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
    last = int((x_fit_max-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
    x = x_data[first:last]
    y = y_data[first:last]
    yerr = np.sqrt(yerr_data[first:last]**2+1)

    p0 = [10**2,-6,20,3.1,0.04]

    popt, pcov = curve_fit(integrated_gaus_bkd_exp,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
    # print(popt)

    a0 = popt[0]
    a1 = popt[1]
    N = popt[2]
    mu = popt[3]
    sigma = popt[4]

    N_err = np.sqrt(pcov[2][2])
    mu_err = np.sqrt(pcov[3][3])
    sigma_err = np.sqrt(pcov[4][4])
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
    def gaus_exp_bdk_pdf(x,a0,a1,mu,sigma):
        pdf_val=(1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0 *a1* np.exp(a1*x)/(np.exp(x_fit_max*a1)-np.exp(x_fit_min*a1))) / (1 + a0)
        return pdf_val

    def minus_log_likelihood(params):
        a0,a1,A,mu,sigma= params
        A=abs(A)
        a0=abs(a0)
        tmp=w_fit*np.log(A*gaus_exp_bdk_pdf(x_fit,a0,a1,mu,sigma))
        return A-tmp.sum()

    initial_guess = [popt[0]/(popt[2]*popt[1])*(np.exp(popt[1]*x_fit_max)-np.exp(popt[1]*x_fit_min)),popt[1],20,3.1,0.04]
    result = minimize(minus_log_likelihood, initial_guess, method = 'BFGS')#, options=dict(maxiter=10000000)
    popt1 = result.x
    hessian_ = hessian(minus_log_likelihood)
    pcov = lin.inv(hessian_(popt1))


    N1 = popt1[2]/(1+popt1[0])
    mu1 = popt1[3]
    sigma1 = abs(popt1[4])

    N1_err = (np.sqrt(pcov[2][2])/popt1[2]+np.sqrt(pcov[0][0])/(1+popt1[0]))*N
    mu1_err = np.sqrt(pcov[3][3])
    sigma1_err = np.sqrt(pcov[4][4])
    # </editor-fold>

    # <editor-fold desc="Unbinned Fit Bkgd First">
    x_points = []
    w_points = []
    x_fit = []
    w_fit = []
    x_fit_bdk = []
    w_fit_bdk = []
    for i, yval in enumerate(y_data):
        if yval != 0:
            x_points = np.append(x_points, x_data[i])
            w_points = np.append(w_points, yval)
            if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                x_fit = np.append(x_fit, x_data[i])
                w_fit = np.append(w_fit, yval)
                if x_data[i] < x_ROI_min or x_data[i] > x_ROI_max:
                    x_fit_bdk = np.append(x_fit_bdk, x_data[i])
                    w_fit_bdk = np.append(w_fit_bdk, yval)


    # Gaussian fit function

    def exp_bdk_noROI_pdf(x, a1):
        pdf_val = a1 * np.exp(a1 * x) / (
                    np.exp(x_fit_max * a1) - np.exp(x_ROI_max * a1) + np.exp(x_ROI_min * a1) - np.exp(x_fit_min * a1))
        return pdf_val


    def minus_log_likelihood_bkd(params):
        A, a1 = params
        A = abs(A)
        tmp = w_fit_bdk * np.log(A * exp_bdk_noROI_pdf(x_fit_bdk, a1))
        return A - tmp.sum()


    initial_guess_bkd = [popt[0] / (popt[2] * popt[1]) * (np.exp(popt[1] * x_fit_max) - np.exp(popt[1] * x_fit_min)),
                         popt[1]]
    result_bkd = minimize(minus_log_likelihood_bkd, initial_guess_bkd,
                          method='BFGS')  # , options=dict(maxiter=10000000)
    popt_bkd = result_bkd.x
    hessian_bkd = hessian(minus_log_likelihood_bkd)
    pcov_bkd = lin.inv(hessian_bkd(popt_bkd))
    a1_fit = popt_bkd[1]


    def gaus_exp_bdk_pdf(x, a0, mu, sigma):
        a1 = a1_fit
        pdf_val = (1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0 * a1 * np.exp(
            a1 * x) / (np.exp(x_fit_max * a1) - np.exp(x_fit_min * a1))) / (1 + a0)
        return pdf_val


    def minus_log_likelihood(params):
        a0, A, mu, sigma = params
        A = abs(A)
        a0 = abs(a0)
        tmp = w_fit * np.log(A * gaus_exp_bdk_pdf(x_fit, a0, mu, sigma))
        return A - tmp.sum()


    initial_guess = [popt[0] / (popt[2] * popt[1]) * (np.exp(popt[1] * x_fit_max) - np.exp(popt[1] * x_fit_min)), 20,
                     3.1, 0.04]
    result = minimize(minus_log_likelihood, initial_guess, method='BFGS')  # , options=dict(maxiter=10000000)

    popt2 = result.x
    hessian_ = hessian(minus_log_likelihood)
    pcov = lin.inv(hessian_(popt2))

    N2 = popt2[1] / (1 + popt2[0])
    mu2 = popt2[2]
    sigma2 = abs(popt2[3])

    N2_err = (np.sqrt(pcov[1][1]) / popt2[1] + np.sqrt(pcov[0][0]) / (1 + popt2[0])) * N
    mu2_err = np.sqrt(pcov[2][2])
    sigma2_err = np.sqrt(pcov[3][3])
    # </editor-fold>

    # <editor-fold desc="Unbinned Fit Bkgd Sub">
    x_points = []
    w_points = []
    x_fit = []
    w_fit = []
    x_fit_bdk = []
    w_fit_bdk = []
    for i, yval in enumerate(y_data):
        if yval != 0:
            x_points = np.append(x_points, x_data[i])
            w_points = np.append(w_points, yval)
            if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                x_fit = np.append(x_fit, x_data[i])
                w_fit = np.append(w_fit, yval)
                if x_data[i] < x_ROI_min or x_data[i] > x_ROI_max:
                    x_fit_bdk = np.append(x_fit_bdk, x_data[i])
                    w_fit_bdk = np.append(w_fit_bdk, yval)


    # Gaussian fit function

    def exp_bdk_noROI_pdf(x, a1):
        pdf_val = a1 * np.exp(a1 * x) / (
                    np.exp(x_fit_max * a1) - np.exp(x_ROI_max * a1) + np.exp(x_ROI_min * a1) - np.exp(x_fit_min * a1))
        return pdf_val


    def minus_log_likelihood_bkd(params):
        A, a1 = params
        A = abs(A)
        tmp = w_fit_bdk * np.log(A * exp_bdk_noROI_pdf(x_fit_bdk, a1))
        return A - tmp.sum()


    initial_guess_bkd = [popt[0] / (popt[2] * popt[1]) * (np.exp(popt[1] * x_fit_max) - np.exp(popt[1] * x_fit_min)),
                         popt[1]]
    result_bkd = minimize(minus_log_likelihood_bkd, initial_guess_bkd,
                          method='BFGS')  # , options=dict(maxiter=10000000)
    popt_bkd = result_bkd.x
    hessian_bkd = hessian(minus_log_likelihood_bkd)
    pcov_bkd = lin.inv(hessian_bkd(popt_bkd))
    A_fit = popt_bkd[0]
    a1_fit = popt_bkd[1]
    A_fit_err = np.sqrt(pcov_bkd[0][0])
    a1_fit_err = np.sqrt(pcov_bkd[1][1])

    a0_fit = A_fit * a1_fit / (
                np.exp(x_fit_max * a1_fit) - np.exp(x_ROI_max * a1_fit) + np.exp(x_ROI_min * a1_fit) - np.exp(
            x_fit_min * a1_fit))
    a0_fit_err = np.sqrt((A_fit_err * a0_fit / A_fit) ** 2 + ((a1_fit_err) * (a0_fit / a1_fit + a0_fit * (
                x_fit_max * np.exp(x_fit_max * a1_fit) - x_ROI_max * np.exp(x_ROI_max * a1_fit) + x_ROI_min * np.exp(
            x_ROI_min * a1_fit) - x_fit_min * np.exp(x_fit_min * a1_fit)) \
                                                                              / (np.exp(x_fit_max * a1_fit) - np.exp(
                x_ROI_max * a1_fit) + np.exp(x_ROI_min * a1_fit) - np.exp(x_fit_min * a1_fit)))) ** 2)

    ROI_sum = sum(w_fit) - sum(w_fit_bdk)
    ROI_sum_err = np.sqrt(ROI_sum)
    N3 = ROI_sum - (A_fit / a1_fit) * (exp_bdk_noROI_pdf(x_ROI_max, a1_fit) - exp_bdk_noROI_pdf(x_ROI_min, a1_fit))
    dNdA = (1 / a1_fit) * (exp_bdk_noROI_pdf(x_ROI_max, a1_fit) - exp_bdk_noROI_pdf(x_ROI_min, a1_fit))
    dNda1 = (A_fit / a1_fit) * (
                x_ROI_max * exp_bdk_noROI_pdf(x_ROI_max, a1_fit) - x_ROI_min * exp_bdk_noROI_pdf(x_ROI_min, a1_fit)) - (
                        A_fit / a1_fit ** 2) * (
                        exp_bdk_noROI_pdf(x_ROI_max, a1_fit) - exp_bdk_noROI_pdf(x_ROI_min, a1_fit))
    dN = np.sqrt((dNdA * A_fit_err) ** 2 + (dNda1 * a1_fit_err) ** 2 + ROI_sum_err ** 2)

    bkd_sum1 = ((A_fit + A_fit_err) / (a1_fit + a1_fit_err)) * (
            exp_bdk_noROI_pdf(x_ROI_max, (a1_fit + a1_fit_err)) - exp_bdk_noROI_pdf(x_ROI_min,
                                                                                    (a1_fit + a1_fit_err)))
    bkd_sum2 = ((A_fit + A_fit_err) / (a1_fit - a1_fit_err)) * (
            exp_bdk_noROI_pdf(x_ROI_max, (a1_fit - a1_fit_err)) - exp_bdk_noROI_pdf(x_ROI_min,
                                                                                    (a1_fit - a1_fit_err)))
    bkd_sum3 = ((A_fit - A_fit_err) / (a1_fit + a1_fit_err)) * (
            exp_bdk_noROI_pdf(x_ROI_max, (a1_fit + a1_fit_err)) - exp_bdk_noROI_pdf(x_ROI_min,
                                                                                    (a1_fit + a1_fit_err)))
    bkd_sum4 = ((A_fit - A_fit_err) / (a1_fit - a1_fit_err)) * (
            exp_bdk_noROI_pdf(x_ROI_max, (a1_fit - a1_fit_err)) - exp_bdk_noROI_pdf(x_ROI_min,

                                                                                    (a1_fit - a1_fit_err)))
    bkd_sum_err = (max(bkd_sum1, bkd_sum2, bkd_sum3, bkd_sum4) - min(bkd_sum1, bkd_sum2, bkd_sum3, bkd_sum4)) / 2
    dN3 = np.sqrt(ROI_sum_err ** 2 + bkd_sum_err ** 2)
    # </editor-fold>


    print(N1)
    print(N2)
    print(N3)
    print(np.std([N1,N2,N3]))


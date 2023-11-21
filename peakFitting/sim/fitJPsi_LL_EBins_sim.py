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

vers="v8"
rebin=10
rebinSubt=rebin
histname="mass_pair"
x_fit_min=2.6
x_fit_max=3.3
fit_peak=[True, True, True]


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

def double_gaus_pdf(x, B, mu, sigma1, sigma2):
    pdf_val = (1 / np.sqrt(2 * np.pi * sigma1 ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma1 ** 2)) + B / np.sqrt(
        2 * np.pi * sigma2 ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma2 ** 2))) / (1 + B)
    return pdf_val


def minus_log_likelihood(params):
    A, B, mu, sigma1, sigma2 = params
    B = abs(B)
    A = abs(A)
    tmp = w_fit * np.log(A * double_gaus_pdf(x_fit, abs(B), mu, sigma1, sigma2))
    return A - tmp.sum()


# for A in ["D","He","C"]:
#     for vers in ["v5","v7"]:
for A in ["C"]:
    for vers in ["v8"]:


        filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"

        simWeights = [0.242, 0.366, 0.069, 1.13, 0.29]  # Not including 2H (0.242 nb)
        simFiles = ["hist_DSelector_2H_MF_helicity_mixed.root",
                    "hist_DSelector_4He_MF_helicity_mixed.root", "hist_DSelector_4He_SRC_helicity_mixed.root",
                    "hist_DSelector_12C_MF_helicity_mixed.root", "hist_DSelector_12C_SRC_helicity_mixed.root"]
        match A:
            case "D":
                simWeights=[0.242]
                simFiles=["hist_DSelector_2H_MF_helicity_mixed.root"]
            case "He":
                simWeights = simWeights[1:3]
                simFiles = simFiles[1:3]
            case "C":
                simWeights = simWeights[3:5]
                simFiles = simFiles[3:5]


        plt.figure(figsize=(6, 10))
        # <editor-fold desc="Sub-Threshold">

        plt.subplot(3, 1, 1)

        directoryname = ".SubThresh.Kin.Jpsi_mass"
        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=1,directoryname=directoryname)
        x_points = []
        w_points = []
        x_fit = []
        w_fit = []
        for i, yval in enumerate(y_data):
            if yval != 0:
                x_points = np.append(x_points, x_data[i])
                w_points = np.append(w_points, yval)
                if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                    x_fit = np.append(x_fit, x_data[i])
                    w_fit = np.append(w_fit, yval)


        initial_guess = [sum(w_fit), 2, 3.1, 0.02, 0.04]
        result = minimize(minus_log_likelihood, initial_guess, method='BFGS')  # , options=dict(maxiter=10000000)

        popt = abs(result.x)
        # pcov = result.hess_inv
        hessian_ = hessian(minus_log_likelihood)
        pcov = lin.inv(hessian_(popt))

        N1 = popt[0] / (1 + popt[1])
        N2 = popt[0] * popt[1] / (1 + popt[1])
        mu = popt[2]
        sigma1 = abs(popt[3])
        sigma2 = abs(popt[4])

        N1_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / (1 + popt[1])) * N1
        N2_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / popt[1] + np.sqrt(pcov[1][1]) / (
                    1 + popt[1])) * N2
        mu_err = np.sqrt(pcov[2][2])
        sigma1_err = np.sqrt(pcov[3][3])
        sigma2_err = np.sqrt(pcov[4][4])

        N = N1 + N2
        N_err = (N1_err / N1 + N2_err / N2) * N

        sigma = (N1 * sigma1 + N2 * sigma2) / N
        sigma_err = (sigma1_err / sigma1 * N1 + sigma2_err / sigma2 * N2) * sigma / N

        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=rebin,directoryname=directoryname)
        dx = x_data[1] - x_data[0]
        xlin = np.linspace(x_fit[0], x_fit[-1], num=1000)
        # plt.subplot(3,1,3)
        plt.plot(xlin, (popt[0] * double_gaus_pdf(xlin, *popt[1:5])) * dx, color='r')
        plt.plot(xlin, N1 * norm.pdf(xlin, mu, sigma1) * dx, color='b', linestyle='--')
        plt.plot(xlin, N2 * norm.pdf(xlin, mu, sigma2) * dx, color='b', linestyle='--')
        plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='.k', capsize=0)

        plt.xlim(2.2, 3.4)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.ylim(ymin, ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")

        placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
        placeText(A, loc=2, yoffset=-30, fontsize=18)

        N_disp = N1
        N_disp_err = N1_err
        sigma_disp = sigma1
        sigma_disp_err = sigma1_err
        if sigma1 > sigma2:
            N_disp = N2
            N_disp_err = N2_err
            sigma_disp = sigma2
            sigma_disp_err = sigma2_err

        placeText(r"$N_{J/\psi}$" + rf"$={N_disp:.1f}\pm{N_disp:.1f}$" + "\n" + rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  + "\n" + rf"$\sigma={sigma_disp:.3f}\pm{sigma_disp_err:.3f}$", loc=2, yoffset=25)

        placeText(r"$E_\gamma <8.2$ GeV", loc=2)


        # <editor-fold desc="Above Threshold Lower">
        plt.subplot(3, 1, 2)


        # <editor-fold desc="Get Data">
        directoryname = ".AboveThresh_lower.Kin.Jpsi_mass"
        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=1,directoryname=directoryname)
        x_points = []
        w_points = []
        x_fit = []
        w_fit = []
        for i, yval in enumerate(y_data):
            if yval != 0:
                x_points = np.append(x_points, x_data[i])
                w_points = np.append(w_points, yval)
                if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                    x_fit = np.append(x_fit, x_data[i])
                    w_fit = np.append(w_fit, yval)

        initial_guess = [sum(w_fit), 2, 3.1, 0.02, 0.04]
        result = minimize(minus_log_likelihood, initial_guess, method='BFGS')  # , options=dict(maxiter=10000000)

        popt = abs(result.x)
        # pcov = result.hess_inv
        hessian_ = hessian(minus_log_likelihood)
        pcov = lin.inv(hessian_(popt))

        N1 = popt[0] / (1 + popt[1])
        N2 = popt[0] * popt[1] / (1 + popt[1])
        mu = popt[2]
        sigma1 = abs(popt[3])
        sigma2 = abs(popt[4])

        N1_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / (1 + popt[1])) * N1
        N2_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / popt[1] + np.sqrt(pcov[1][1]) / (
                1 + popt[1])) * N2
        mu_err = np.sqrt(pcov[2][2])
        sigma1_err = np.sqrt(pcov[3][3])
        sigma2_err = np.sqrt(pcov[4][4])

        N = N1 + N2
        N_err = (N1_err / N1 + N2_err / N2) * N

        sigma = (N1 * sigma1 + N2 * sigma2) / N
        sigma_err = (sigma1_err / sigma1 * N1 + sigma2_err / sigma2 * N2) * sigma / N

        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=rebin,directoryname=directoryname)
        dx = x_data[1] - x_data[0]
        xlin = np.linspace(x_fit[0], x_fit[-1], num=1000)
        # plt.subplot(3,1,3)
        plt.plot(xlin, (popt[0] * double_gaus_pdf(xlin, *popt[1:5])) * dx, color='r')
        plt.plot(xlin, N1 * norm.pdf(xlin, mu, sigma1) * dx, color='b', linestyle='--')
        plt.plot(xlin, N2 * norm.pdf(xlin, mu, sigma2) * dx, color='b', linestyle='--')
        plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='.k', capsize=0)

        plt.xlim(2.2, 3.4)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.ylim(ymin, ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")

        placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
        placeText(A, loc=2, yoffset=-30, fontsize=18)

        N_disp = N1
        N_disp_err = N1_err
        sigma_disp = sigma1
        sigma_disp_err = sigma1_err
        if sigma1 > sigma2:
            N_disp = N2
            N_disp_err = N2_err
            sigma_disp = sigma2
            sigma_disp_err = sigma2_err

        placeText(r"$N_{J/\psi}$" + rf"$={N_disp:.1f}\pm{N_disp:.1f}$" + "\n" + rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  + "\n" + rf"$\sigma={sigma_disp:.3f}\pm{sigma_disp_err:.3f}$", loc=2, yoffset=25)
        placeText(r"$8.2< E_\gamma <9.5$ GeV", loc=2)
        # </editor-fold>
        # </editor-fold>

        # <editor-fold desc="Above Threshold Upper">
        plt.subplot(3, 1, 3)
        directoryname = ".AboveThresh_upper.Kin.Jpsi_mass"

        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=1,directoryname=directoryname)
        x_points = []
        w_points = []
        x_fit = []
        w_fit = []
        for i, yval in enumerate(y_data):
            if yval != 0:
                x_points = np.append(x_points, x_data[i])
                w_points = np.append(w_points, yval)
                if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                    x_fit = np.append(x_fit, x_data[i])
                    w_fit = np.append(w_fit, yval)


        initial_guess = [sum(w_fit), 2, 3.1, 0.02, 0.04]
        result = minimize(minus_log_likelihood, initial_guess, method='BFGS')  # , options=dict(maxiter=10000000)

        popt = abs(result.x)
        # pcov = result.hess_inv
        hessian_ = hessian(minus_log_likelihood)
        pcov = lin.inv(hessian_(popt))

        N1 = popt[0] / (1 + popt[1])
        N2 = popt[0] * popt[1] / (1 + popt[1])
        mu = popt[2]
        sigma1 = abs(popt[3])
        sigma2 = abs(popt[4])

        N1_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / (1 + popt[1])) * N1
        N2_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / popt[1] + np.sqrt(pcov[1][1]) / (
                    1 + popt[1])) * N2
        mu_err = np.sqrt(pcov[2][2])
        sigma1_err = np.sqrt(pcov[3][3])
        sigma2_err = np.sqrt(pcov[4][4])

        N = N1 + N2
        N_err = (N1_err / N1 + N2_err / N2) * N

        sigma = (N1 * sigma1 + N2 * sigma2) / N
        sigma_err = (sigma1_err / sigma1 * N1 + sigma2_err / sigma2 * N2) * sigma / N

        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in simFiles],
                                          weights=simWeights, histname=histname, rebin=rebin,directoryname=directoryname)
        dx = x_data[1] - x_data[0]
        xlin = np.linspace(x_fit[0], x_fit[-1], num=1000)
        # plt.subplot(3,1,3)
        plt.plot(xlin, (popt[0] * double_gaus_pdf(xlin, *popt[1:5])) * dx, color='r')
        plt.plot(xlin, N1 * norm.pdf(xlin, mu, sigma1) * dx, color='b', linestyle='--')
        plt.plot(xlin, N2 * norm.pdf(xlin, mu, sigma2) * dx, color='b', linestyle='--')
        plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='.k', capsize=0)

        plt.xlim(2.2, 3.4)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.ylim(ymin, ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")

        placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
        placeText(A, loc=2, yoffset=-30, fontsize=18)

        N_disp = N1
        N_disp_err = N1_err
        sigma_disp = sigma1
        sigma_disp_err = sigma1_err
        if sigma1 > sigma2:
            N_disp = N2
            N_disp_err = N2_err
            sigma_disp = sigma2
            sigma_disp_err = sigma2_err

        placeText(r"$N_{J/\psi}$" + rf"$={N_disp:.1f}\pm{N_disp:.1f}$" + "\n" + rf"$\mu={mu:.3f}\pm{mu_err:.3f}$"
                  + "\n" + rf"$\sigma={sigma_disp:.3f}\pm{sigma_disp_err:.3f}$", loc=2, yoffset=25)

        placeText(r"$E_\gamma > 9.5$ GeV", loc=2)
        # </editor-fold>
        # </editor-fold>
        # plt.savefig(f"../../files/figs/peakFits/3panel/Mee_{A}_3panel_noTrackShower_{vers}_bin{rebin}.pdf")
        plt.show()




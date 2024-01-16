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

A = "D"
vers="v8"
rebin=30
histname="mass_pair_Egamma_7_8p2"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3
x_fit_min=2.6
x_fit_max=3.3


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

filepath = f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/cutVary/noTrackShower/loose_medium/"


range_vals=("loc0","loc1","loc2","loc3","loc4")
df = pd.DataFrame(columns=['preB','sigma_min','sigma_max','E_miss','N', 'N_err', 'mu', 'mu_err','sigma','sigma_err'])

for preB in range_vals:
    for sigma_min in range_vals:
        for sigma_max in range_vals:
            for E_miss in range_vals:
                # <editor-fold desc="Get Data">

                directoryname = f".preB_{preB}_sigma_{sigma_min}_{sigma_max}_Emiss_{E_miss}.Jpsi_mass"
                simWeights = [0.242, 0.366, 0.069, 1.13, 0.29]  # Not including 2H (0.242 nb)
                simFiles = ["hist_cutVary_DSelector_2H_MF_helicity_mixed.root",
                            "hist_cutVary_DSelector_4He_MF_helicity_mixed.root", "hist_cutVary_DSelector_4He_SRC_helicity_mixed.root",
                            "hist_cutVary_DSelector_12C_MF_helicity_mixed.root", "hist_cutVary_DSelector_12C_SRC_helicity_mixed.root"]
                match A:
                    case "D":
                        simWeights = [0.242]
                        simFiles = ["hist_cutVary_DSelector_2H_MF_helicity_mixed.root"]
                    case "He":
                        simWeights = simWeights[1:3]
                        simFiles = simFiles[1:3]
                    case "C":
                        simWeights = simWeights[3:5]
                        simFiles = simFiles[3:5]

                # <editor-fold desc="Unbinned Fit">
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


                def double_gaus_pdf(x, B, mu, sigma1, sigma2):
                    pdf_val = (1 / np.sqrt(2 * np.pi * sigma1 ** 2) * np.exp(
                        -(x - mu) ** 2 / (2 * sigma1 ** 2)) + B / np.sqrt(2 * np.pi * sigma2 ** 2) * np.exp(
                        -(x - mu) ** 2 / (2 * sigma2 ** 2))) / (1 + B)
                    return pdf_val


                def minus_log_likelihood(params):
                    A, B, mu, sigma1, sigma2 = params
                    B = abs(B)
                    A = abs(A)
                    tmp = w_fit * np.log(A * double_gaus_pdf(x_fit, abs(B), mu, sigma1, sigma2))
                    return A - tmp.sum()


                initial_guess = [sum(w_fit), 2, 3.1, 0.02, 0.04]
                result = minimize(minus_log_likelihood, initial_guess,
                                  method='BFGS')  # , options=dict(maxiter=10000000)

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
                N2_err = (np.sqrt(pcov[0][0]) / popt[0] + np.sqrt(pcov[1][1]) / popt[1] + np.sqrt(
                    pcov[1][1]) / (1 + popt[1])) * N2
                mu_err = np.sqrt(pcov[2][2])
                sigma1_err = np.sqrt(pcov[3][3])
                sigma2_err = np.sqrt(pcov[4][4])

                N = N1 + N2
                N_err = (N1_err / N1 + N2_err / N2) * N

                sigma = (N1 * sigma1 + N2 * sigma2) / N
                sigma_err = (sigma1_err / sigma1 * N1 + sigma2_err / sigma2 * N2) * sigma / N



                N_disp = N1
                N_disp_err = N1_err
                sigma_disp = sigma1
                sigma_disp_err = sigma1_err
                if sigma1 > sigma2:
                    N_disp = N2
                    N_disp_err = N2_err
                    sigma_disp = sigma2
                    sigma_disp_err = sigma2_err
                # </editor-fold>

                df.loc[len(df.index)] = [preB, sigma_min, sigma_max,E_miss,N1+N2,N_disp_err,mu,mu_err,sigma_disp,sigma_disp_err]



df.to_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_7_8p2/peak_params_{A}_sim.csv",index=False)
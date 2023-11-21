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
vers="v7"
rebin=40
# directoryname=".SubThresh.Kin.Jpsi_mass"
directoryname=".All.Kin.Jpsi_mass"
histname="mass_pair"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3
x_fit_min=2.6
x_fit_max=3.3
x_ROI_max=3.2
x_ROI_min=2.9

for A in ["D","He","C"]:
#     for vers in ["v5","v7"]:
# for A in ["C"]:
    for zVert_num in range(1,9,4):
        for vers in ["v8"]:
            # <editor-fold desc="Get Data">
            def getXY(infiles,weights,histname, rebin,directorynames):
                x=0
                y=0
                yerr=0
                for i, infile in enumerate(infiles):
                    for directoryname in directorynames:
                        f = root2mpl.File(infile,dir=directoryname)
                        h = f.get(histname, rebin=rebin)
                        x = h.x
                        y += h.y*weights[i]
                        yerr = np.sqrt(yerr**2 +(h.yerr*weights[i])**2)
                return x,y,yerr

            filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/"

            dataFiles=[f"data_hist_zComp_{A}.root"]
            directorynames = [f".All_z_vertex_foil_{zVert_num}.Kin.Jpsi_mass",f".All_z_vertex_foil_{zVert_num+1}.Kin.Jpsi_mass",f".All_z_vertex_foil_{zVert_num+2}.Kin.Jpsi_mass",f".All_z_vertex_foil_{zVert_num+3}.Kin.Jpsi_mass"]
            # directoryname = f".All_z_vertex_{zVert_num}.Kin.Jpsi_mass"
            x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                          weights=[1],histname=histname,rebin=rebin,directorynames=directorynames)
            dx = x_data[1]-x_data[0]


            def gaus_bkd_exp(x, a0, a1, A, mu, sigma):
                return (a0 * 10 ** 6 * np.exp(x * a1) + A * norm.pdf(x, loc=mu, scale=sigma))


            def integrated_gaus_bkd_exp(x, a0, a1, A, mu, sigma):
                df = []
                for xval in x:
                    df_val = integrate.quad(gaus_bkd_exp, xval - dx / 2, xval + dx / 2, args=(a0, a1, A, mu, sigma),
                                            epsrel=0.01)
                    df.append(df_val[0])
                return df
            # </editor-fold>

            # <editor-fold desc="Binned Exp">
            first = int((x_fit_min-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
            last = int((x_fit_max-x_data[0])/(x_data[-1]-x_data[0])*len(x_data))
            x = x_data[first:last]
            y = y_data[first:last]
            yerr = np.sqrt(yerr_data[first:last]**2+1)

            p0 = [10**2,-6,5,3.1,0.04]
            # popt, pcov = curve_fit(integrated_gaus_bkd_exp,x,y,sigma=yerr,absolute_sigma=True,p0 = p0)
            # # print(popt)
            #
            # a0 = popt[0]
            # a1 = popt[1]
            # N = popt[2]
            # mu = popt[3]
            # sigma = popt[4]
            #
            # N_err = np.sqrt(pcov[2][2])
            # mu_err = np.sqrt(pcov[3][3])
            # sigma_err = np.sqrt(pcov[4][4])
            # </editor-fold>

            # <editor-fold desc="Unbinned Fit">
            x_data,y_data,yerr_data=getXY(infiles=[filepath+tree for tree in dataFiles],
                                          weights=[1],histname=histname,rebin=1,directorynames=directorynames)
            x_points = []
            w_points = []
            x_fit = []
            w_fit = []
            x_fit_bkd = []
            w_fit_bkd = []
            for i, yval in enumerate(y_data):
                if yval != 0:
                    x_points = np.append(x_points, x_data[i])
                    w_points = np.append(w_points, yval)
                    if x_data[i] < x_fit_max and x_data[i] > x_fit_min:
                        x_fit = np.append(x_fit, x_data[i])
                        w_fit = np.append(w_fit, yval)
                        if x_data[i] < x_ROI_min or x_data[i] > x_ROI_max:
                            x_fit_bkd = np.append(x_fit_bkd, x_data[i])
                            w_fit_bkd = np.append(w_fit_bkd, yval)

            # Gaussian fit function
            def exp_bkd_noROI_pdf(x, a1):
                pdf_val = a1 * np.exp(a1 * x) / (
                            np.exp(x_fit_max * a1) - np.exp(x_ROI_max * a1) + np.exp(x_ROI_min * a1) - np.exp(
                        x_fit_min * a1))
                return pdf_val


            def minus_log_likelihood_bkd(params):
                A, a1 = params
                A = abs(A)
                tmp = w_fit_bkd * np.log(A * exp_bkd_noROI_pdf(x_fit_bkd, a1))
                return A - tmp.sum()

            initial_guess_bkd=[22.28971574, -6.84530912]
            # initial_guess_bkd = [
            #     popt[0] / (popt[2] * popt[1]) * (np.exp(popt[1] * x_fit_max) - np.exp(popt[1] * x_fit_min)), popt[1]]
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
                        x_fit_max * np.exp(x_fit_max * a1_fit) - x_ROI_max * np.exp(
                    x_ROI_max * a1_fit) + x_ROI_min * np.exp(x_ROI_min * a1_fit) - x_fit_min * np.exp(
                    x_fit_min * a1_fit)) \
                                                                                      / (np.exp(
                        x_fit_max * a1_fit) - np.exp(x_ROI_max * a1_fit) + np.exp(x_ROI_min * a1_fit) - np.exp(
                        x_fit_min * a1_fit)))) ** 2)

            ROI_sum = sum(w_fit) - sum(w_fit_bkd)
            ROI_sum_err = np.sqrt(ROI_sum)
            N = ROI_sum - (A_fit / a1_fit) * (
                        exp_bkd_noROI_pdf(x_ROI_max, a1_fit) - exp_bkd_noROI_pdf(x_ROI_min, a1_fit))
            dNdA = (1 / a1_fit) * (exp_bkd_noROI_pdf(x_ROI_max, a1_fit) - exp_bkd_noROI_pdf(x_ROI_min, a1_fit))
            dNda1 = (A_fit / a1_fit) * (
                        x_ROI_max * exp_bkd_noROI_pdf(x_ROI_max, a1_fit) - x_ROI_min * exp_bkd_noROI_pdf(x_ROI_min,
                                                                                                         a1_fit)) - (
                                A_fit / a1_fit ** 2) * (
                                exp_bkd_noROI_pdf(x_ROI_max, a1_fit) - exp_bkd_noROI_pdf(x_ROI_min, a1_fit))
            dN = np.sqrt((dNdA * A_fit_err) ** 2 + (dNda1 * a1_fit_err) ** 2 + ROI_sum_err ** 2)

            bkd_sum1 = ((A_fit + A_fit_err) / (a1_fit + a1_fit_err)) * (
                    exp_bkd_noROI_pdf(x_ROI_max, (a1_fit + a1_fit_err)) - exp_bkd_noROI_pdf(x_ROI_min,
                                                                                            (a1_fit + a1_fit_err)))
            bkd_sum2 = ((A_fit + A_fit_err) / (a1_fit - a1_fit_err)) * (
                    exp_bkd_noROI_pdf(x_ROI_max, (a1_fit - a1_fit_err)) - exp_bkd_noROI_pdf(x_ROI_min,
                                                                                            (a1_fit - a1_fit_err)))
            bkd_sum3 = ((A_fit - A_fit_err) / (a1_fit + a1_fit_err)) * (
                    exp_bkd_noROI_pdf(x_ROI_max, (a1_fit + a1_fit_err)) - exp_bkd_noROI_pdf(x_ROI_min,
                                                                                            (a1_fit + a1_fit_err)))
            bkd_sum4 = ((A_fit - A_fit_err) / (a1_fit - a1_fit_err)) * (
                    exp_bkd_noROI_pdf(x_ROI_max, (a1_fit - a1_fit_err)) - exp_bkd_noROI_pdf(x_ROI_min,

                                                                                            (a1_fit - a1_fit_err)))
            bkd_sum_err = (max(bkd_sum1, bkd_sum2, bkd_sum3, bkd_sum4) - min(bkd_sum1, bkd_sum2, bkd_sum3,
                                                                             bkd_sum4)) / 2
            dN = np.sqrt(ROI_sum_err ** 2 + bkd_sum_err ** 2)

            x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in dataFiles],
                                              weights=[1], histname=histname, rebin=rebin,directorynames=directorynames)

            xlin = np.linspace(x_fit[0], x_fit[-1], num=1000)
            # plt.subplot(3,1,3)
            plt.plot(xlin, (popt_bkd[0] * exp_bkd_noROI_pdf(xlin, popt_bkd[1])) * dx, color='b', linestyle='--')
            # plt.plot(xlin, a0_fit*np.exp(a1_fit*xlin) * dx, color='g', linestyle='--')

            # plt.plot(xlin, (popt[1] * gaus_exp_bkd_pdf(xlin, popt[0], *popt[2:5])) * dx, color='r')

            plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='.k', capsize=0)

            plt.xlim(2.2, 3.4)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.ylim(ymin, ymax)

            plt.plot([x_ROI_min, x_ROI_min], [ymin, ymax * 0.8], 'r--')
            plt.plot([x_ROI_max, x_ROI_max], [ymin, ymax * 0.8], 'r--')

            plt.ylabel("Counts")
            plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")

            placeText("No Extra Tracks/Showers" + "\n" + vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
            placeText(A, loc=2, yoffset=-30, fontsize=18)

            placeText(r"$N_{J/\psi}$" + rf"$={N:.1f}\pm{dN:.1f}$")
            # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
            # placeText("Unbinned", loc=2)
            # </editor-fold>

            # plt.savefig(f"../../files/figs/peakFits/bkd_sub/Mee_{A}_noTrackShower_bkdFirst_sub_{vers}_bin{rebin}.pdf")
            # print(rf"{A} a1= {a1_fit:.3f}   {a1_fit_err:.3f}")


            # placeText(r"$N_{J/\psi}$"+rf"$={N:.1f}\pm{N_err:.1f}$"+"\n"+rf"$\mu={mu:.3f}\pm{mu_err:.3f}$")
            placeText(f"Foil {zVert_num} - {zVert_num+3}",loc=2)
            # placeText(f"z Region {zVert_num}", loc=2)
            # </editor-fold>

            plt.savefig(f"../../files/figs/peakFits/zVertex/foils/bkd_sub/Mee_{A}_zVert_Foil_{zVert_num}_{zVert_num+3}_bkd_sub_noTrackShower_{vers}_bin{rebin}.pdf")
            print(N)
            print(dN)
            plt.show()




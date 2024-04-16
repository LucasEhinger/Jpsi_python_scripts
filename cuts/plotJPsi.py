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

cut ="ptAlphaCut"
cut ="All"

A = "C"
rebin=30
# directoryname=".SubThresh.Kin.Jpsi_mass"
directoryname=".All.Kin.Jpsi_mass"


histname="mass_pair_no_neg"
# histname="mass_kin"
# mass_pair_fine, mass_pair_fine_pt0p3, mass_pair_fine_alpha1p2, mass_pair_fine_alpha1p2_pt0p3

for A in ["D","He","C"]:
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
        dataFiles=[f"data_hist2_{A}.root"]

        x_data, y_data, yerr_data = getXY(infiles=[filepath + tree for tree in dataFiles],
                                          weights=[1], histname=histname, rebin=rebin)

        x_data_accid, y_data_accid, yerr_data_accid = getXY(infiles=[filepath + tree for tree in dataFiles],
                                          weights=[1], histname="mass_pair_neg", rebin=rebin)
        dx = x_data[1]-x_data[0]


        plt.errorbar(x_data,y_data,yerr=yerr_data,fmt='.k',capsize=0,label="On time")
        plt.errorbar(x_data_accid, y_data_accid * (-1), yerr=yerr_data_accid, fmt='.b', capsize=0,label="Accidentals")

        plt.xlim(2.2,3.4)
        xmin, xmax, ymin, ymax=plt.axis()
        plt.ylim(ymin,ymax)
        plt.ylabel("Counts")
        plt.xlabel(r"Light-cone m($e^+e^-$) [GeV]")
        # plt.xlabel(r"True m($e^+e^-$) [GeV]")

        placeText("No Extra Tracks/Showers" +"\n"+vers, loc=1, yoffset=-40)  # +"\n"+"pT<0.3"
        placeText(A, loc=2, yoffset=-30, fontsize=18)


        plt.legend()
        placeText("Unbinned",loc=2)

        plt.savefig(f"../../files/figs/peakFits/unbinned/accidental_comp/Mee_{A}_noTrackShower_{vers}_bin{rebin}_AccidentalComp.pdf")
        plt.show()

        # print(popt[1])




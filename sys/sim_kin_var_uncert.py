import matplotlib.pyplot as plt
import root2mpl
import numpy as np
import pystyle
from pystyle.annotate import placeText
from scipy.optimize import curve_fit

plt.style.use("SRC_CT_presentation")

A="He"
vers="v8"
nbins=10
rebin=1
filepath=f"/Users/lucasehinger/CLionProjects/untitled/Files/ifarmHists/{vers}/filtered/noTrackShower/simHists/resolution/"

directories=[".delta_kmiss",".delta_pT",".delta_alpha_miss",".delta_t",".delta_t_lc"]
sigmas=np.zeros((len(directories),nbins))
mus=np.zeros((len(directories),nbins))

var_range_map = {directory[7:]: np.zeros(2) for directory in directories}
var_range_map["kmiss"] = [0,1]
var_range_map["pT"] = [0,0.6]
var_range_map["alpha_miss"] = [0.8,1.5]
var_range_map["t"] = [-6,0]
var_range_map["t_lc"] = [-6,0]

var_unit_map = {directory[7:]: "" for directory in directories}
var_unit_map["kmiss"] = "[GeV/c]"
var_unit_map["pT"] = "[GeV/c]"
var_unit_map["alpha_miss"] = ""
var_unit_map["t"] = "[GeV^2]"
var_unit_map["t_lc"] = "[GeV^2]"

for directoryname in directories:
    f = root2mpl.File(filepath+f"hist_resolution_DSelector_12C_SRC_helicity_mixed.root",dir=directoryname)
    for bin_num in range(10):
        histname = directoryname[1:]+f"_bin_{bin_num}"
        h = f.get(histname,rebin=rebin)
        x=h.x
        y=h.y
        yerr=h.yerr
        # fit x and y with a gaussian
        # p0 = [50,0,0.1]
        # def fun(x,A,mu,sigma):
        #     return (A * np.exp(-0.5*((x-mu)/sigma)**2))
        # popt, pcov = curve_fit(fun,x,y,p0 = p0)
        # mu = popt[1]
        # sigma = popt[2]

        # Scale factor for y values
        scale_factor = 1000

        y[y<0]=0
        # Flatten the x values according to their weights (y values)
        weighted_data = np.repeat(x, np.round(y * scale_factor).astype(int))

        # Calculate the weighted median
        median = np.median(weighted_data)

        # Calculate the 16th and 84th weighted percentiles (corresponding to +/- 1 sigma in a normal distribution)
        lower_68 = np.percentile(weighted_data, 16)
        upper_68 = np.percentile(weighted_data, 84)
        sigma = (upper_68 - lower_68) / 2
        mu = median

        mus[directories.index(directoryname),bin_num]=mu
        sigmas[directories.index(directoryname),bin_num]=sigma

    plt.figure()
    x_vals = np.linspace(var_range_map[directoryname[7:]][0], var_range_map[directoryname[7:]][1], nbins)
    plt.plot(x_vals, sigmas[directories.index(directoryname)], 'o')
    plt.ylabel(r"$\sigma$" + " " + var_unit_map[directoryname[7:]])
    plt.xlabel(directoryname[7:] + " " + var_unit_map[directoryname[7:]])
    # plt.title(directoryname[7:])
    plt.savefig(f"../../files/figs/sys/{directoryname[7:]}_sigma_{A}.pdf", bbox_inches='tight')
    plt.show()

    # plt.figure()
    # plt.plot(x_vals, mus[directories.index(directoryname)], 'o')
    # plt.ylabel(r"$\mu$" + " " + var_unit_map[directoryname[7:]])
    # plt.xlabel(directoryname[7:] + " " + var_unit_map[directoryname[7:]])
    # # plt.title(directoryname[7:])
    # plt.show()
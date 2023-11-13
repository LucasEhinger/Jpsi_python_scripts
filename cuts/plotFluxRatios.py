#!/usr/bin/env python3.10


import numpy as np
import ROOT
import matplotlib.pyplot as plt
import root2mpl
import pystyle
from scipy.stats import norm

from pystyle.annotate import placeText

from scipy.optimize import curve_fit
import os

plt.figure(figsize=(3,4))
plt.style.use("SRC_CT_presentation")

plt.errorbar(0, 1.49734, yerr=0.091575184, fmt='.k',label='C/He')
plt.errorbar(0, 4.01695, yerr=0.33487554, fmt='.r',label='C/D')
plt.errorbar(0, 2.68272, yerr=0.240366359, fmt='.g',label='He/D')


plt.axhline(y=1.647441,color='k',linestyle='--',label='Flux predictions')
plt.axhline(y=4.102606,color='r',linestyle='--')
plt.axhline(y=2.490290,color='g',linestyle='--')

# xtickvals=['z=32','z=83','z=88']
# plt.xticks([0,1,2],xtickvals)
plt.xticks([])
# plt.ylabel('Peak Height')

# plt.ylim([0.5,8])
# plt.legend()
plt.text(-0.05, 4.15, 'C/D',fontweight='bold',c='r')
plt.text(-0.05, 2.55, 'He/D',fontweight='bold',c='g')
plt.text(-0.05, 1.7, 'C/He',fontweight='bold',c='k')
plt.savefig("figures/"+f"Flux Ratios.pdf")

plt.show()

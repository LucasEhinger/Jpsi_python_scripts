import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystyle
from pystyle.annotate import placeText
plt.style.use("SRC_CT_presentation")

A="C"
Egamma="7_8p2"
# Egamma="8p2_9p5"
df_data =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/peak_params_fixed_{A}_data.csv")
df_sim =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/peak_params_{A}_sim.csv")
mid=(len(df_data["N"])-1)/2
N_ratios=df_data["N"]/df_sim["N"]/(df_data["N"][mid]/df_sim["N"][mid])
N_weights=(df_data["N"]/df_data["N_err"])**2

N_weights=N_weights.loc[(N_ratios<3) & (N_ratios>0)]
N_ratios=N_ratios[(N_ratios<3) & (N_ratios>0)]
N_weights=N_weights[np.isfinite(N_ratios)]
N_ratios=N_ratios[np.isfinite(N_ratios)]

mu=sum(N_ratios)/len(N_ratios)
sigma=np.sqrt(sum((N_ratios-mu)**2)/len(N_ratios))#standard deviation

def getErr(nSigma=1):
    mu=N_ratios.mean()
    lower_index=int(len(N_ratios)*(1-0.68)/2)
    upper_index = int(len(N_ratios) * (1-(1 - 0.68) / 2))
    N_ratios_sorted=sorted(N_ratios)
    return [N_ratios_sorted[lower_index]-mu,N_ratios_sorted[upper_index]-mu]


sigma_minus,sigma_plus=getErr()
dx=0.03
plt.figure(figsize=(6,4))
plt.hist(N_ratios,bins=int((max(N_ratios)-min(N_ratios))/dx),weights=N_weights)
# if A=="D":
#     plt.ylim(0,45)
xmin,xmax,ymin,ymax=plt.axis()
plt.plot([1,1],[ymin,ymax],'r--')
plt.ylabel("Counts")
plt.xlim(0.5,1.5)
plt.xlabel("Y $\epsilon$ / Y$_0$ $\epsilon_0$")
placeText(A,loc=2,yoffset=-25)
placeText(rf'$\sigma$   $={(sigma_plus-sigma_minus)/2:.3f}$'
          +"\n"+rf'$\sigma^+={sigma_plus:.3f}$'
          +"\n"+rf'$\sigma^-={-sigma_minus:.3f}$',loc=1)
plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/yield_comp_uncert_{A}_no_weight.pdf")
plt.show()
# with open(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_{A}.csv",'a') as fd:
#     fd.write(str((sigma_plus-sigma_minus)/2))
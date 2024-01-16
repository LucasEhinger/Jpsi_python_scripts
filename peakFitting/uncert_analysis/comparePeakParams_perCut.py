import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystyle
from pystyle.annotate import placeText
plt.style.use("SRC_CT_presentation")

A="C"
npts=5
Egamma="8p2_9p5"
df_data =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/peak_params_fixed_{A}_data.csv")
df_sim =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/peak_params_{A}_sim.csv")
mid=(len(df_data["N"])-1)/2
N_ratios=df_data["N"]/df_sim["N"]/(df_data["N"][mid]/df_sim["N"][mid])
N_weights=(df_data["N"]/df_data["N_err"])**2

# N_weights=N_weights[N_ratios<3]

mu=sum(N_ratios)/len(N_ratios)
sigma=np.sqrt(sum((N_ratios-mu)**2)/len(N_ratios))


mid=(npts-1)/2
index_list=np.array(range(npts))

def getIndices(indx1,indx2,indx3, cut):
    indices=None
    match cut:
        case 'Emiss':
            indices = (indx1*npts**3)+(indx2*npts**2)+(indx3*npts**1)+(index_list)
        case 'sigma_max':
            indices = (indx1*npts**3)+(indx2*npts**2)+(index_list*npts**1)+(indx3)
        case 'sigma_min':
            indices = (indx1*npts**3)+(index_list*npts**2)+(indx2*npts**1)+(indx3)
        case 'preB':
            indices = (index_list*npts**3)+(indx1*npts**2)+(indx2*npts**1)+(indx3)
    return indices

std_vals=np.zeros((4,npts**3))
for i,cutVal in enumerate(["preB","sigma_min","sigma_max","Emiss"]):
    for j in range(npts):
        for k in range(npts):
            for l in range(npts):
                N_ratios_sub=N_ratios[getIndices(j,k,l,cutVal)]
                N_ratios_sub = N_ratios_sub[N_ratios < 3]
                std_val=np.std(N_ratios_sub)
                std_vals[i,j*npts**2+k*npts+l]=std_val


dx=0.005
plt.figure(figsize=(6,4))
for i,cutVal in enumerate(["preB","sigma_min","sigma_max","Emiss"]):
    std_noNA=std_vals[i,:]
    std_noNA=std_noNA[np.isfinite(std_noNA)]
    plt.hist(std_noNA,bins=int((std_noNA.max()-std_noNA.min())/dx),alpha=0.8,label=cutVal)

plt.legend()
if A=="C":
    plt.xlim(0,0.3)
if A=="He":
    plt.xlim(0,0.2)
# xmin,xmax,ymin,ymax=plt.axis()
# plt.plot([1,1],[ymin,ymax],'r--')
plt.ylabel("Counts")
plt.xlabel("5 point stdev")
placeText(A,loc=2,yoffset=-25)
# placeText(rf'$\sigma={sigma:.3f}$',loc=1)
# plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/yield_comp_uncert_byCut_{A}_no_weight.pdf")
plt.show()

df = pd.DataFrame(columns=["std"])
for i in range(4):
    std_noNA = std_vals[i, :]
    std_noNA=std_noNA[np.isfinite(std_noNA)]
    df.loc[i] = sum(std_noNA)/len(std_noNA)

# df.to_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_{A}.csv",header=False,index=False)


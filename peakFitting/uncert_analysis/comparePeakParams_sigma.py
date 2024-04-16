import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystyle
from pystyle.annotate import placeText
plt.style.use("SRC_CT_presentation")

cut="All"
all_cuts=["All","pt03","alpha1p2","pt_alpha"]
multifiles=True

for sim in [False]:
    for weighted in [True]:
        # for A in ["D","He","C"]:
        for A in ["C"]:
            df_data =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/{cut}/peak_params_{A}_data.csv")
            df_sim =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/{cut}/peak_params_{A}_sim.csv")
            if sim:
                df_data=df_sim
            if multifiles:
                df_data = pd.concat((pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/{cut}/peak_params_{A}_data.csv") for cut in all_cuts), ignore_index=True)
                if sim:
                    df_data = pd.concat(
                        (pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/{cut}/peak_params_{A}_sim.csv") for
                         cut in all_cuts), ignore_index=True)

            mid=(len(df_data["sigma"])-1)/2
            sigma_data=df_data["sigma"]
            sigma_weights=(1/df_data["sigma_err"])**2

            ix=np.argsort(sigma_data)
            ix = np.delete(ix, np.where(ix == -1))
            sigma_data = sigma_data[ix]
            sigma_weights = sigma_weights[ix]

            sigma_weights=sigma_weights.loc[(sigma_data<1) & (sigma_data>0) & (np.invert(sigma_weights.isna()))]
            sigma_data=sigma_data.loc[(sigma_data<1) & (sigma_data>0) & (np.invert(sigma_weights.isna()))]
            sigma_weights=sigma_weights[np.isfinite(sigma_data)]
            sigma_data=sigma_data[np.isfinite(sigma_data)]

            sigma_center=sum(sigma_data)/len(sigma_data)
            sigma=np.sqrt(sum((sigma_data-sigma_center)**2)/len(sigma_data))#standard deviation

            def getErr(sigma_data,sigma_weights,weighted=False):
                if not weighted:
                    sigma_center=sigma_data.mean()
                    return [np.percentile(sigma_data, (100-68)/2)-sigma_center,np.percentile(sigma_data, 100-(100-68)/2)-sigma_center]
                else:
                    cdf = (np.cumsum(sigma_weights) - 0.5 * sigma_weights) / np.sum(sigma_weights)
                    sigma_center=np.interp(0.5, cdf, sigma_data)
                    return [np.interp((1-0.68)/2, cdf, sigma_data)-sigma_center,np.interp(1-(1-0.68)/2, cdf, sigma_data)-sigma_center]

            sigma_minus,sigma_plus=getErr(sigma_data,sigma_weights,weighted)
            dx=0.03
            plt.figure(figsize=(6,4))
            # plt.hist(sigma_data,bins=int((max(sigma_data)-min(sigma_data))/dx),weights=sigma_weights)
            if weighted:
                plt.hist(sigma_data,bins=10,weights=sigma_weights)
                placeText("Weighted",loc=1,yoffset=-25)
            else:
                plt.hist(sigma_data, bins=10)
            # if A=="D":
            #     plt.ylim(0,45)
            xmin,xmax,ymin,ymax=plt.axis()
            # plt.plot([1,1],[ymin,ymax],'r--')
            plt.ylabel("Counts")
            plt.xlim(xmin,xmax+(xmax-xmin)/3)

            plt.xlabel("Peak Width [GeV]")
            placeText(A+f"{' Sim' if sim else ''}",loc=2,yoffset=-30,xoffset=-40,fontsize=20)
            placeText(rf'$\mu$   $={sigma_center:.3f}$'
                      +"\n"+rf'$\sigma$   $={(sigma_plus-sigma_minus)/2:.3f}$'
                      +"\n"+rf'$\sigma^+={sigma_plus:.3f}$'
                      +"\n"+rf'$\sigma^-={-sigma_minus:.3f}$',loc=1)
            if weighted:
                plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/{cut}/sigma_comp_uncert_{A}_weighted{'_sim' if sim else ''}.pdf")
            else:
                plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/{cut}/sigma_comp_uncert_{A}_no_weight{'_sim' if sim else ''}.pdf")
            plt.show()
            # with open(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_{A}.csv",'a') as fd:
            #     fd.write(str((sigma_plus-sigma_minus)/2))
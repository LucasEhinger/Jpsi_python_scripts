import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystyle
from pystyle.annotate import placeText
plt.style.use("SRC_CT_presentation")

cut="All"
all_cuts=["All","pt03","alpha1p2","pt_alpha"]
multifiles=True

for sim in [True, False]:
    for weighted in [True, False]:
        for A in ["D","He","C"]:
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

            mid=(len(df_data["mu"])-1)/2
            mu_data=df_data["mu"]
            mu_weights=(1/df_data["mu_err"])**2

            ix=np.argsort(mu_data)
            ix=np.delete(ix, np.where(ix == -1))
            mu_data = mu_data[ix]
            mu_weights = mu_weights[ix]

            mu_weights=mu_weights.loc[(mu_data<3.4) & (mu_data>3)]
            mu_data=mu_data.loc[(mu_data<3.4) & (mu_data>3)]
            mu_weights=mu_weights[np.isfinite(mu_data)]
            mu_data=mu_data[np.isfinite(mu_data)]

            mu_center=sum(mu_data)/len(mu_data)
            sigma=np.sqrt(sum((mu_data-mu_center)**2)/len(mu_data))#standard deviation

            def getErr(mu_data,mu_weights,weighted=False):
                if not weighted:
                    mu_center=mu_data.mean()
                    return [np.percentile(mu_data, (100-68)/2)-mu_center,np.percentile(mu_data, 100-(100-68)/2)-mu_center]
                else:
                    cdf = (np.cumsum(mu_weights) - 0.5 * mu_weights) / np.sum(mu_weights)
                    mu_center=np.interp(0.5, cdf, mu_data)
                    return [np.interp((1-0.68)/2, cdf, mu_data)-mu_center,np.interp(1-(1-0.68)/2, cdf, mu_data)-mu_center]

            sigma_minus,sigma_plus=getErr(mu_data,mu_weights,weighted)
            dx=0.03
            plt.figure(figsize=(6,4))
            # plt.hist(mu_data,bins=int((max(mu_data)-min(mu_data))/dx),weights=mu_weights)
            if weighted:
                plt.hist(mu_data,bins=10,weights=mu_weights)
                placeText("Weighted",loc=1,yoffset=-25)
            else:
                plt.hist(mu_data, bins=10)
            # if A=="D":
            #     plt.ylim(0,45)
            xmin,xmax,ymin,ymax=plt.axis()
            # plt.plot([1,1],[ymin,ymax],'r--')
            plt.ylabel("Counts")
            plt.xlim(xmin,xmax+(xmax-xmin)/3)

            plt.xlabel("Peak Mean [GeV]")
            placeText(A+f"{' Sim' if sim else ''}",loc=2,yoffset=-30,xoffset=-40,fontsize=20)
            placeText(rf'$\mu$   $={mu_center:.3f}$'
                      +"\n"+rf'$\sigma$   $={(sigma_plus-sigma_minus)/2:.3f}$'
                      +"\n"+rf'$\sigma^+={sigma_plus:.3f}$'
                      +"\n"+rf'$\sigma^-={-sigma_minus:.3f}$',loc=1)
            if weighted:
                plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/{cut}/mu_comp_uncert_{A}_weighted{'_sim' if sim else ''}{'_combo' if multifiles else ''}.pdf")
            else:
                plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/{cut}/mu_comp_uncert_{A}_no_weight{'_sim' if sim else ''}{'_combo' if multifiles else ''}.pdf")
            plt.show()
            # with open(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_{A}.csv",'a') as fd:
            #     fd.write(str((sigma_plus-sigma_minus)/2))
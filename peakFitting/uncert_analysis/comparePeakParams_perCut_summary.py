import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystyle
from pystyle.annotate import placeText
plt.style.use("SRC_CT_presentation")
Egamma="7_11"

df_stdev_D =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_D.csv",header=None)
df_stdev_He =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_He.csv",header=None)
df_stdev_C =pd.read_csv(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/stdev_means_C.csv",header=None)

df_uncert_D=df_stdev_D
df_uncert_He=df_stdev_He
df_uncert_C=df_stdev_C

# df_uncert_D=df_stdev_D[0:4]/df_stdev_D[0:4].sum()*df_stdev_D[0][4]
# df_uncert_He=df_stdev_He[0:4]/df_stdev_He[0:4].sum()*df_stdev_He[0][4]
# df_uncert_C=df_stdev_C[0:4]/df_stdev_C[0:4].sum()*df_stdev_C[0][4]

y0=np.array([df_uncert_D[0][0],df_uncert_He[0][0],df_uncert_C[0][0]])*100
y1=np.array([df_uncert_D[0][1],df_uncert_He[0][1],df_uncert_C[0][1]])*100
y2=np.array([df_uncert_D[0][2],df_uncert_He[0][2],df_uncert_C[0][2]])*100
y3=np.array([df_uncert_D[0][3],df_uncert_He[0][3],df_uncert_C[0][3]])*100

x=[0,1,2]
xerr=[0.5,0.5,0.5]
# plt.bar(x,y0,color='b',label=r'$E_{preshower}$')
# plt.bar(x,y1,bottom=y0,color='orange',label='p/E lower')
# plt.bar(x,y2,bottom=y0+y1,color='g',label='p/E upper')
# plt.bar(x,y3,bottom=y0+y1+y2,color='r',label=r'$E_{miss}$')
plt.errorbar(x,y0,xerr=xerr,fmt='none',color='b',capsize=0,label=r'$E_{preshower}$')
plt.errorbar(x,y1,xerr=xerr,fmt='none',color='orange',capsize=0,label='p/E lower')
plt.errorbar(x,y2,xerr=xerr,fmt='none',color='g',capsize=0,label='p/E upper')
plt.errorbar(x,y3,xerr=xerr,fmt='none',color='r',capsize=0,label=r'$E_{miss}$')
plt.ylim(0,25)
plt.xticks(x,["D","He","C"])
plt.legend(fontsize=10)
plt.ylabel('Uncertainty (%)')
plt.title('Total Cut Uncertainty by Cut')
plt.savefig(f"../../../files/figs/peakFits/uncert/5 params/Egamma_{Egamma}/Cut_Uncertainty_Breakdown_v2.pdf")
plt.show()
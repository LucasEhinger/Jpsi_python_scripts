import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("SRC_CT_presentation")
from pystyle.annotate import placeText

df_sim_peaks = pd.read_csv ('../../../files/figs/peakFits/zVertex/foils/sim/sim_peak_yields.csv')
df_sim_peaks_noTrigger = pd.read_csv ('../../../files/figs/peakFits/zVertex/foils/sim/sim_peak_yields_noTrigger.csv')
df_data_peaks= pd.read_csv ('../../../files/figs/peakFits/zVertex/foils/bkd_sub/data_peak_yields_2bins.csv')

x=range(0,len(df_sim_peaks))
xerr=np.ones(len(df_sim_peaks))*0.5
plt.figure(figsize=(7,7))
plt.subplot(3,1,1)
plt.errorbar(x,df_sim_peaks["D"]/sum(df_sim_peaks["D"]),xerr=xerr,fmt='ro--',label="D")
plt.errorbar(x,df_sim_peaks["He"]/sum(df_sim_peaks["He"]),xerr=xerr,fmt='go--',label="He")
plt.errorbar(x,df_sim_peaks["C"]/sum(df_sim_peaks["C"]),xerr=xerr,fmt='bo--',label="C")
# plt.ylabel("Normalized yield (Simulataion)")
plt.legend(loc=2)
plt.ylabel("Normalized yield")
placeText("Simulation",loc=1,yoffset=-22)
plt.xticks([0,1,2,3,4,5,6,7],["1","2","3", "4","5","6","7","8"])

plt.subplot(3,1,2)
plt.errorbar(x,df_sim_peaks_noTrigger["D"]/sum(df_sim_peaks_noTrigger["D"]),xerr=xerr,fmt='ro--',label="D")
plt.errorbar(x,df_sim_peaks_noTrigger["He"]/sum(df_sim_peaks_noTrigger["He"]),xerr=xerr,fmt='go--',label="He")
plt.errorbar(x,df_sim_peaks_noTrigger["C"]/sum(df_sim_peaks_noTrigger["C"]),xerr=xerr,fmt='bo--',label="C")

plt.xticks([])
plt.xticks([0,1,2,3,4,5,6,7],["1","2","3", "4","5","6","7","8"])
plt.ylabel("Normalized yield")
placeText("Simulation (No Trigger)",loc=1,yoffset=-22)

plt.subplot(3,1,3)
x=[1.5,5.5]
dx=np.ones(len(x))*0.05
plt.errorbar(x,df_data_peaks["D"]/sum(df_data_peaks["D"]),yerr=df_data_peaks["D Err"]/sum(df_data_peaks["D"]),xerr=[2,2],fmt='ro--',label="D")
plt.errorbar(x+dx,df_data_peaks["He"]/sum(df_data_peaks["He"]),yerr=df_data_peaks["He Err"]/sum(df_data_peaks["He"]),xerr=[2,2],fmt='go--',label="D")
plt.errorbar(x+dx*2,df_data_peaks["C"]/sum(df_data_peaks["C"]),yerr=df_data_peaks["C Err"]/sum(df_data_peaks["C"]),xerr=[2,2],fmt='bo--',label="D")
# plt.plot(x,df_sim_peaks_noTrigger["He"]/sum(df_sim_peaks_noTrigger["He"]),'go--',label="He")
# plt.plot(x,df_sim_peaks_noTrigger["C"]/sum(df_sim_peaks_noTrigger["C"]),'bo--',label="C")
plt.xticks([1.5,5.5],["1 - 4","5 -8"])
plt.xlabel('z-location ("foil")')
placeText("Data",loc=1,yoffset=-22)
plt.ylabel("Normalized yield")
plt.savefig(f"../../../files/figs/peakFits/zVertex/foils/sim/sim_yield_comparison_zVertex_2peaks.pdf")
plt.show()

#
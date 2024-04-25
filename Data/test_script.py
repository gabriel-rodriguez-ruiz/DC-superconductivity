import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('resonance_shift_new.dat',skip_header=1, skip_footer=0)
B=data[:,0]
shift_trans_3_7GHz=data[:,1]/1e6
shift_long_3_7GHz=data[:,2]/1e6
shift_trans_4_4GHz=data[:,3]/1e6
shift_long_4_4GHz=data[:,4]/1e6

fig=plt.figure()
plt.scatter(B, shift_trans_3_7GHz, label='delta f, 3.7 GHz, transversal')
plt.scatter(B, shift_long_3_7GHz, label='delta f, 3.7 GHz, longitudinal')
plt.scatter(B, shift_trans_4_4GHz, label='delta f, 4.4 GHz, transversal')
plt.scatter(B, shift_long_4_4GHz, label='delta f, 4.4 GHz, longitudinal')
size=36
plt.rcParams['figure.figsize'] = [16,12] 
plt.rcParams['axes.linewidth'] = 2
plt.minorticks_on()
plt.tick_params(axis='both', which="major", length=20, width=2, top=True, right=True, direction="in", pad=10)
plt.tick_params(axis="both", which="minor", length=10, width=2, top=True, right=True, direction="in", pad=10)
plt.legend(loc='lower left', prop={'size':size-10}, ncol = 1, fontsize=30, markerscale=2, edgecolor='black', fancybox='false') 
plt.xlabel("magnetic field [T]",fontsize=size)
plt.ylabel("resonance shift [MHz]",fontsize=size)
plt.yticks(fontsize=size)
plt.xticks(fontsize=size)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=5)
plt.tight_layout()
plt.show()
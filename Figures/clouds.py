import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import *
import numpy as np
import math
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

def weightedmean(data, err):            #calculates the weighted mean for data points data with std devs. err
        weights = 1.0/err**2
        mu = np.sum(data*weights)/np.sum(weights)
        var = 1.0/np.sum(weights)
        return [mu, np.sqrt(var)]                #returns weighted mean and standard dev.


def get_offset(model, data, err):
	bchi = 1.0e10
	for offset in np.arange(0, 20, 0.5):
		chi = np.sum((data - model - offset/err)**2)
		if chi < bchi:		
			boffset = offset
	return boffset
		

def boxcar_smooth(x, nsmooth):
	n = len(x)
	for i in arange(0, n):
		lower = np.max(np.array([0, i - int(nsmooth/2)]))
		upper = np.min(np.array([n-1, i + int(nsmooth/2)]))
		x[i] = 	np.mean(x[lower:upper])
	return x

def bin_at_obs_res(waves, model_waves, model):
	delta_lambda = (waves[1] - waves[0])/2.0
	binned_model = np.zeros_like(waves)
	for i in range(0, len(waves)):
		binned_model[i] = np.average(model[(model_waves>waves[i]-delta_lambda)&(model_waves<waves[i]+delta_lambda)])
	return binned_model	
			


matplotlib.rcParams.update({'font.size':10})
plt.rc('legend', **{'fontsize':10})

fillcolor1 = "#81f7f3"
fillcolor2 = "#01a9db"

rcParams['axes.linewidth'] = 0.8
mew=0.0
ew = 0.7
ms=3.0
marker="o"

fig = plt.figure(figsize=(6,3))



#########################################################
## PANEL 2 ######################################

#ax3 = plt.subplot(212)

d = np.genfromtxt("transmission_5pix.dat")
x = d[:,0]
y = d[:,1]*1.0e6
offset = np.mean(y)
y -= offset 
e = d[:,2]*1.0e6

deltawave = 0.0115
for i in range(len(x)): print '{0:0.3f}'.format(x[i]-deltawave), '{0:0.3f}'.format(x[i]+deltawave), y[i], e[i]

m = np.genfromtxt("WASP43b_TRAN_bin_5pix_spectrum_range.txt", skip_header=1)
m[:,1::] *= 1.0e6
m[:,1::] -= offset

#ax3.fill_between(m[:,0], m[:,5], m[:,6],linewidth=0, color=fillcolor1)
#ax3.fill_between(m[:,0], m[:,3], m[:,4],linewidth=0, color=fillcolor2)
plt.plot(m[:,0], m[:,1], color="blue", linewidth=0.6)

b = np.genfromtxt("WASP43b_TRAN_bin_5pix_bestfit_spectrum_binned.txt", skip_header=1)
b[:,1] *= 1.0e6
b[:,1] -= offset
#plt.plot(b[:,0], b[:,1], marker='o', color='b', linestyle="none", ms=ms+1.5)

ms = 3.0
#ax3.errorbar(x, y, yerr=e, linestyle="None", ms = ms+1.2, marker='o', markeredgewidth=0.8, color = "w", markeredgecolor='k',linewidth=0.8, ecolor='k', capsize=2.0,zorder=100)

#plt.xlim((1.1, 1.7))

plt.gca().set_axis_off()

plt.tight_layout()
plt.savefig("model.png", dpi=300)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from circles import circles
from matplotlib import cm

figure(figsize=(8,8))

ax=subplot(aspect='equal')
start = 0.0
stop = 1.0
n = 50
cm_subsection = linspace(start, stop, n) 
cmap = matplotlib.cm.get_cmap('viridis')

colors = [ cmap(x) for x in cm_subsection ]

s = np.linspace(2., 0.1, n)
for i in range(n):
	circles(0, 0, s=s[i], c=colors[i],  ec='none')

circles(-1.8, 0, s=1.0, c="0.8", alpha=0.7, ec='none')
circles(-1.8, 0, s=0.8, c="#c1c6fc",  ec='none')


plt.xlim(-4, 4)
plt.ylim(-3,3)
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)



plt.savefig("fig1.pdf")



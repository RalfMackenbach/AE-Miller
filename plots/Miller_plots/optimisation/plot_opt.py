import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
import scipy
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# import the data from optimized-tokamak-par.h5
f = h5py.File('optimized-tokamak-par.h5', 'r')
AE_opt = f['AE_opt'][:]
kappa_opt = f['kappa_opt'][:]
delta_opt = f['delta_opt'][:]
sv = f['sv'][:]
alphav = f['alphav'][:]
f.close()


# find minimal and maximal values of kappa and delta
kappa_min = 1/2
kappa_max = 2.0
delta_min = -0.5
delta_max = 0.5



def fmt(x, pos):
    a = round(x,pos)
    return r'${}$'.format(a)

# remap AE_opt to log10(AE_opt)
AE_opt = np.log10(AE_opt)

# plot the data
AE_min = np.amin(AE_opt)
AE_max = np.amax(AE_opt)

fig, axs = plt.subplots(1,3, figsize=(6.850394, 5.0/1.7),tight_layout=True) #figsize=(6.850394, 3.0)
pAE         = axs[0].pcolor(alphav,sv,AE_opt,cmap='plasma',vmin=AE_min,vmax=AE_max)
pAE.set_edgecolor('face')
cbarae   = fig.colorbar(pAE, ticks=[AE_min,AE_max], ax=axs[0],orientation="horizontal",pad=0.3,label=r'$\log_{10} \widehat{A}$')
cbarae.set_ticks(ticks=[AE_min,AE_max])
cbarae.set_ticklabels([fmt(AE_min,1),fmt(AE_max,1)])
pkappa      = axs[1].pcolor(alphav,sv,kappa_opt,cmap='bwr',vmin=kappa_min,vmax=kappa_max)
pkappa.set_edgecolor('face')
cbarkappa   = fig.colorbar(pkappa, ticks=[kappa_min,kappa_max], ax=axs[1],orientation="horizontal",pad=0.3,label=r'$\kappa$')
cbarkappa.set_ticks(ticks=[kappa_min,kappa_max])
cbarkappa.set_ticklabels([r'$1/2$',r'$2$'])
pdelta      = axs[2].pcolor(alphav,sv,delta_opt,cmap='bwr')
pdelta.set_edgecolor('face')
cbardelta   = fig.colorbar(pdelta, ticks=[delta_min,delta_max], ax=axs[2],orientation="horizontal",pad=0.3,label=r'$\delta$')
cbardelta.set_ticklabels([r'$-1/2$',r'$+1/2$'])
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
# levels =    MaxNLocator(nbins=4).tick_values(0, 4)
# cmap = plt.colormaps['jet']
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# cat         = 2+2*np.arctan2(delta_opt,kappa_opt-1)/np.pi
# pcat        = axs[3].pcolor(alphav,sv,cat,cmap=cmap,norm=norm)
# pcat.set_edgecolor('face')
# cbarcat     = fig.colorbar(pcat, ax=axs[3],orientation="horizontal",pad=0.3)
# cbarcat.set_ticks(ticks=[0.5,1.5,2.5,3.5],labels=[r'$\mathrm{NC}$',r'$\mathrm{NT}$',r'$\mathrm{PT}$',r'$\mathrm{PC}$'])

axs[0].set_ylabel(r'$s$')
axs[1].set_ylabel(r'$s$')
axs[2].set_ylabel(r'$s$')
# axs[3].set_ylabel(r'$s$')
axs[0].set_xlabel(r'$\alpha$')
axs[1].set_xlabel(r'$\alpha$')
axs[2].set_xlabel(r'$\alpha$')
# axs[3].set_xlabel(r'$\alpha$')
axs[0].set_yticks([-1,0,1,2,3,4])
axs[1].set_yticks([-1,0,1,2,3,4])
axs[2].set_yticks([-1,0,1,2,3,4])
# axs[3].set_yticks([-1,0,1,2,3,4])
axs[0].set_yticklabels([r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
axs[1].set_yticklabels([None,None,None,None,None,None])
axs[2].set_yticklabels([None,None,None,None,None,None])
# axs[3].set_yticklabels([None,None,None,None,None,None])
axs[0].text(0.8,-0.5,r'$(a)$',color='white')
axs[1].text(0.8,-0.5,r'$(b)$')
axs[2].text(0.8,-0.5,r'$(c)$')
# axs[3].text(0.8,-0.5,r'$(d)$')
plt.savefig('optimized-salpha.png', format='png',
            #This is recommendation for publication plots
            dpi=1000,
            # Plot will be occupy a maximum of available space
            bbox_inches='tight')
plt.show()
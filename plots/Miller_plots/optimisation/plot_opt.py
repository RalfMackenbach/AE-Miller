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
kappa_min = np.amin(kappa_opt)
kappa_max = np.amax(kappa_opt)
delta_min = np.amin(delta_opt)
delta_max = np.amax(delta_opt)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



fig, axs = plt.subplots(1,4, figsize=(6.850394, 5.0/2.5)) #figsize=(6.850394, 3.0)
pAE         = axs[0].pcolor(alphav,sv,AE_opt,cmap='plasma')
pAE.set_edgecolor('face')
cbarae   = fig.colorbar(pAE, ticks=[np.amin(AE_opt),np.amax(AE_opt)], ax=axs[0],orientation="horizontal",pad=0.3,label=r'$\widehat{A}$')
cbarae.set_ticks(ticks=[np.amin(AE_opt),np.amax(AE_opt)])
cbarae.set_ticklabels([fmt(np.amin(AE_opt),1),fmt(np.amax(AE_opt),1)])
pkappa      = axs[1].pcolor(alphav,sv,kappa_opt,cmap='viridis')
pkappa.set_edgecolor('face')
cbarkappa   = fig.colorbar(pkappa, ticks=[kappa_min,kappa_max], ax=axs[1],orientation="horizontal",pad=0.3,label=r'$\kappa$')
pdelta      = axs[2].pcolor(alphav,sv,delta_opt,cmap='viridis')
pdelta.set_edgecolor('face')
cbardelta   = fig.colorbar(pdelta, ticks=[delta_min,delta_max], ax=axs[2],orientation="horizontal",pad=0.3,label=r'$\delta$')
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
levels =    MaxNLocator(nbins=4).tick_values(0, 4)
cmap = plt.colormaps['jet']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cat         = 2+2*np.arctan2(delta_opt,kappa_opt-1)/np.pi
pcat        = axs[3].pcolor(alphav,sv,cat,cmap=cmap,norm=norm)
pcat.set_edgecolor('face')
cbarcat     = fig.colorbar(pcat, ax=axs[3],orientation="horizontal",pad=0.3)
cbarcat.set_ticks(ticks=[0.5,1.5,2.5,3.5],labels=[r'$\mathrm{NC}$',r'$\mathrm{NT}$',r'$\mathrm{PT}$',r'$\mathrm{PC}$'])

axs[0].set_ylabel(r'$s$')
axs[1].set_ylabel(r'$s$')
axs[2].set_ylabel(r'$s$')
axs[3].set_ylabel(r'$s$')
axs[0].set_xlabel(r'$\alpha$')
axs[1].set_xlabel(r'$\alpha$')
axs[2].set_xlabel(r'$\alpha$')
axs[3].set_xlabel(r'$\alpha$')
axs[0].set_yticks([-1,0,1,2,3,4])
axs[1].set_yticks([-1,0,1,2,3,4])
axs[2].set_yticks([-1,0,1,2,3,4])
axs[3].set_yticks([-1,0,1,2,3,4])
axs[0].set_yticklabels([r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
axs[1].set_yticklabels([None,None,None,None,None,None])
axs[2].set_yticklabels([None,None,None,None,None,None])
axs[3].set_yticklabels([None,None,None,None,None,None])
axs[0].text(0.8,-0.5,r'$(a)$',color='white')
axs[1].text(0.8,-0.5,r'$(b)$')
axs[2].text(0.8,-0.5,r'$(c)$')
axs[3].text(0.8,-0.5,r'$(d)$')
plt.tight_layout()
plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/optimisation/optimized-triptych.eps', format='eps',
            #This is recommendation for publication plots
            dpi=1000,
            # Plot will be occupy a maximum of available space
            bbox_inches='tight')
plt.show()
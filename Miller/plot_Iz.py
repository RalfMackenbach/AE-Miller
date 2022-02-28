import AE_tokamak_functions as AEtf
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



vint = np.vectorize(AEtf.integral_over_z, otypes=[np.float64])

c0 = np.linspace(-1.0, 1.0,    1000,   endpoint=True)
c1 = np.linspace(-1.0, 1.0,   1000,   endpoint=True)

c0v, c1v = np.meshgrid(c0, c1, indexing='ij')


fig = plt.figure(figsize=(3.375, 2.3))
ax  = fig.gca()
levels = np.linspace(0.0,7.0,25)
cnt = plt.contourf(c0v,c1v,vint(c0v,c1v),levels=levels,cmap='plasma')
plt.xlabel(r'$c_0$')
plt.ylabel(r'$c_1$')
cbar = plt.colorbar(ticks=[0.0,7.0])
cbar.set_label(r'$I_z$')
cbar.solids.set_edgecolor("face")
ax.xaxis.set_tick_params(which='major', direction='in', top='on')
ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
ax.yaxis.set_tick_params(which='major', direction='in', top='on')
ax.set_yticks([-0.5,0.0,0.5,1.0])
ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
for c in cnt.collections:
    c.set_edgecolor("face")
plt.tight_layout()
plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/plot_Iz.eps', format='eps',
            #This is recommendation for publication plots
            dpi=1000,
            # Plot will be occupy a maximum of available space
            bbox_inches='tight',pad_inches = 0.01)

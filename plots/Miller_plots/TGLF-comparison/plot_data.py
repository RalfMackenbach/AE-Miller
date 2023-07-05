import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.close('all')

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if b != 0:
        return r'${} \cdot 10^{{{}}}$'.format(a, b)
    if b == 0:
        return r'${}$'.format(a)



AEf     = h5py.File('AE_data.hdf5', 'r')
AEkd    = AEf['AEkd']
AEsa    = AEf['AEsa']
AEsa_2    = AEf['AEsa_2']
kappa   = AEf['kappa']
delta   = AEf['delta']
alpha   = AEf['alpha']
shear   = AEf['shear']



TGLFf       = h5py.File('AE_data.hdf5', 'r')
Qkd         = AEf['AEkd']
Qsa         = AEf['AEsa']
kappa_TGLF  = AEf['kappa']
delta_TGLF  = AEf['delta']
alpha_TGLF  = AEf['alpha']
shear_TGLF  = AEf['shear']


QAEkd    = np.asarray(AEkd)**(3/2)
QAEsa   = np.asarray(AEsa)**(3/2)
QAEsa2  = np.asarray(AEsa_2)**(3/2)


lvls0   = np.linspace(0.0,np.asarray(QAEkd).max(),21)
lvls1   = np.linspace(0.0,np.asarray(QAEsa).max(),21)
lvls2   = np.linspace(0.0,np.asarray(QAEsa2).max(),21)
lvls3   = np.linspace(0.0,np.asarray(QAEkd).max(),21)
lvls4   = np.linspace(0.0,np.asarray(QAEsa).max(),21)
lvls5   = np.linspace(0.0,np.asarray(QAEsa2).max(),21)


fig, axs = plt.subplots(2,3, figsize=(6.850394, 5.0),constrained_layout=True)




cnt0 = axs[0,0].contourf(kappa,delta,QAEkd,levels=lvls0,cmap='plasma')
axs[0,0].set_xlabel(r'$\kappa$')
axs[0,0].set_ylabel(r'$\delta$')
axs[0,0].set_xticks([0.5,1.0,1.5,2.0])
axs[0,0].set_yticks([-0.5,0.0,0.5])
cbar0 = fig.colorbar(cnt0,ticks=[0.0, np.asarray(QAEkd).max()],ax=axs[0,0],label=r'$\widehat{Q}_\mathrm{\AE{}}$',location='top')
cbar0.set_ticklabels([r'$0$', fmt(np.asarray(QAEkd).max(),1)])


cnt1 = axs[0,1].contourf(alpha,shear,QAEsa,levels=lvls1,cmap='plasma')
axs[0,1].set_xlabel(r'$\alpha$')
axs[0,1].set_ylabel(r'$s$')
cbar1 = fig.colorbar(cnt1,ticks=[0.0, np.asarray(QAEsa).max()],ax=axs[0,1],label=r'$\widehat{Q}_\mathrm{\AE{}}$',location='top')
cbar1.set_ticklabels([r'$0$', fmt(np.asarray(QAEsa).max(),1)])


cnt2 = axs[0,2].contourf(alpha,shear,QAEsa2,levels=lvls2,cmap='plasma')
axs[0,2].set_xlabel(r'$\alpha$')
axs[0,2].set_ylabel(r'$s$')
cbar2 = fig.colorbar(cnt2,ticks=[0.0, np.asarray(QAEsa2).max()],ax=axs[0,2],label=r'$\widehat{Q}_\mathrm{\AE{}}$',location='top')
cbar2.set_ticklabels([r'$0$', fmt(np.asarray(QAEsa2).max(),1)])



cnt3 = axs[1,0].contourf(kappa,delta,QAEkd,levels=lvls3,cmap='plasma')
axs[1,0].set_ylabel(r'$\delta$')
axs[1,0].xaxis.tick_top()
axs[1,0].set_xticks([0.5,1.0,1.5,2.0])
axs[1,0].set_yticks([-0.5,0.0,0.5])
cbar3 = fig.colorbar(cnt3,ticks=[0.0, np.asarray(QAEkd).max()],ax=axs[1,0],label=r'$\widehat{Q}_\mathrm{TGLF}$',location='bottom')
cbar3.set_ticklabels([r'$0$', fmt(np.asarray(QAEkd).max(),1)])


cnt4 = axs[1,1].contourf(alpha,shear,QAEsa,levels=lvls4,cmap='plasma')
axs[1,1].xaxis.tick_top()
axs[1,1].set_ylabel(r'$s$')
cbar4 = fig.colorbar(cnt4,ticks=[0.0, np.asarray(QAEsa).max()],ax=axs[1,1],label=r'$\widehat{Q}_\mathrm{TGLF}$',location='bottom')
cbar4.set_ticklabels([r'$0$', fmt(np.asarray(QAEsa).max(),1)])


cnt5 = axs[1,2].contourf(alpha,shear,QAEsa2,levels=lvls5,cmap='plasma')
axs[1,2].xaxis.tick_top()
axs[1,2].set_ylabel(r'$s$')
cbar5 = fig.colorbar(cnt5,ticks=[0.0, np.asarray(QAEsa2).max()],ax=axs[1,2],label=r'$\widehat{Q}_\mathrm{TGLF}$',location='bottom')
cbar5.set_ticklabels([r'$0$', fmt(np.asarray(QAEsa2).max(),1)])


cbar0.solids.set_edgecolor("face")
cbar1.solids.set_edgecolor("face")
cbar2.solids.set_edgecolor("face")
cbar3.solids.set_edgecolor("face")
cbar4.solids.set_edgecolor("face")
cbar5.solids.set_edgecolor("face")

for c in cnt0.collections:
    c.set_edgecolor("face")
for c in cnt1.collections:
    c.set_edgecolor("face")
for c in cnt2.collections:
    c.set_edgecolor("face")
for c in cnt3.collections:
    c.set_edgecolor("face")
for c in cnt4.collections:
    c.set_edgecolor("face")
for c in cnt5.collections:
    c.set_edgecolor("face")



plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/TGLF-comparison/TGLF_plot.eps', format='eps',
                #This is recommendation for publication plots
                dpi=2000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', pad_inches = 0.1)




plt.show()
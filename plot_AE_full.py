import h5py
import matplotlib.pyplot as plt
import numpy



# import  matplotlib  as mpl
# from matplotlib import rc
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
# # rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern']})
# ## for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
# rc('text', usetex=True)
plt.rcParams["font.family"] = "sans-serif"


f = h5py.File('data_full.h5', 'r')
shear_arr = f['shear_array']
omn_arr = f['omn_array']
eta_arr = f['eta_array']
AE_err_arr = f['AE_error_array']
AE_arr = f['AE_array']



cmapstring='plasma_r'

fig, axs = plt.subplots(2, 2, figsize=(3.375, 3.0))
c1 =axs[0,0].contourf(shear_arr[:,:,0],omn_arr[:,:,0],AE_arr[:,:,0],12,cmap=cmapstring)
c2 =axs[0,1].contourf(shear_arr[:,:,1],omn_arr[:,:,1],AE_arr[:,:,1],12,cmap=cmapstring)
c3 =axs[1,0].contourf(shear_arr[:,:,2],omn_arr[:,:,2],AE_arr[:,:,2],12,cmap=cmapstring)
c4 =axs[1,1].contourf(shear_arr[:,:,3],omn_arr[:,:,3],AE_arr[:,:,3],12,cmap=cmapstring)
axs[0,0].text(0.7, 0.35, '(a)', color='0.0')
axs[0,1].text(0.7, 0.35, '(b)', color='0.0')
axs[1,0].text(0.7, 0.35, '(c)', color='0.0')
axs[1,1].text(0.7, 0.35, '(d)', color='0.0')

for ax in axs.flat:
    ax.set(xlabel=r'$s$', ylabel=r'$\omega_n$')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on', color='0.0')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on', color='0.0')
    ax.spines['bottom'].set_color('0.0')
    ax.spines['top'].set_color('0.0')
    ax.spines['left'].set_color('0.0')
    ax.spines['right'].set_color('0.0')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

for c in c1.collections:
    c.set_edgecolor("face")
for c in c2.collections:
    c.set_edgecolor("face")
for c in c3.collections:
    c.set_edgecolor("face")
for c in c4.collections:
    c.set_edgecolor("face")



cb1=fig.colorbar(c1,ax=axs[0,0])
cb2=fig.colorbar(c2,ax=axs[0,1])
cb3=fig.colorbar(c3,ax=axs[1,0])
cb4=fig.colorbar(c4,ax=axs[1,1])
cb1.solids.set_edgecolor("face")
cb2.solids.set_edgecolor("face")
cb3.solids.set_edgecolor("face")
cb4.solids.set_edgecolor("face")
cb1.ax.tick_params(size=0)
cb2.ax.tick_params(size=0)
cb3.ax.tick_params(size=0)
cb4.ax.tick_params(size=0)

# cb2.set_label(r'$\hat{A}$',rotation=0,loc='bottom')
# cb4.set_label(r'$\hat{A}$',rotation=0)

fig.tight_layout()
plt.subplots_adjust(left=0.11, right=0.95, top=0.95, bottom=0.15)
plt.savefig('AE_plots.png', format='png',
            #This is recommendation for publication plots
            dpi=1000)
plt.show()

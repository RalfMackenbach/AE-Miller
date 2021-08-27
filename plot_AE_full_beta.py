import h5py
import matplotlib.pyplot as plt
import numpy             as np
import matplotlib.ticker as ticker
import matplotlib        as mpl
from   matplotlib        import rc


mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
# rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern']})
## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return '{} \cdot 10^{{{}}}'.format(a, b)




f = h5py.File('data_full_beta_withprefac.h5', 'r')
wn_arr      = f['wn_array']
q_arr       = f['q_array']
shear_arr   = f['shear_array']
beta_arr    = f['beta_array']
eta_arr     = f['eta_array']
AE_err_arr  = f['AE_error_array']
AE_arr      = f['AE_array']


# wn_idx = 3



cmapstring='plasma'
v       = np.linspace(0, 1.0, 100, endpoint=True)
v_ticks = np.linspace(0, 1.0, 6, endpoint=True)
v_label = [r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$']


q_idx   = np.array([0,1,2,3])
eta_idx = np.array([0,1,2,3])



q_arr, eta_arr = np.meshgrid(q_idx, eta_idx, indexing='ij')

for idx, val in np.ndenumerate(q_arr):

    q_p   = q_arr[idx]
    eta_p = eta_arr[idx]


    fig, axs = plt.subplots(2, 2, figsize=(4.72440945, 4.0*0.7))
    c1 =axs[0,0].contourf(shear_arr[0,q_p,:,:,eta_p],beta_arr[0,q_p,:,:,eta_p],AE_arr[0,q_p,:,:,eta_p]/np.amax(AE_arr[0,q_p,:,:,eta_p]),v,cmap=cmapstring)
    c2 =axs[0,1].contourf(shear_arr[1,q_p,:,:,eta_p],beta_arr[1,q_p,:,:,eta_p],AE_arr[1,q_p,:,:,eta_p]/np.amax(AE_arr[1,q_p,:,:,eta_p]),v,cmap=cmapstring)
    c3 =axs[1,0].contourf(shear_arr[2,q_p,:,:,eta_p],beta_arr[2,q_p,:,:,eta_p],AE_arr[2,q_p,:,:,eta_p]/np.amax(AE_arr[2,q_p,:,:,eta_p]),v,cmap=cmapstring)
    c4 =axs[1,1].contourf(shear_arr[3,q_p,:,:,eta_p],beta_arr[3,q_p,:,:,eta_p],AE_arr[3,q_p,:,:,eta_p]/np.amax(AE_arr[3,q_p,:,:,eta_p]),v,cmap=cmapstring)
    axs[0,0].text(-3.6, 0.3*0.5, '(a)', color='1.0')
    axs[0,1].text(-3.6, 0.3*0.5, '(b)', color='1.0')
    axs[1,0].text(-3.6, 0.3*0.5, '(c)', color='1.0')
    axs[1,1].text(-3.6, 0.3*0.5, '(d)', color='1.0')
    axs[0,0].text(-3.4, 1.65*0.5, r'${}$'.format(fmt(np.amax(AE_arr[0,q_p,:,:,eta_p]),2)), color='1.0')
    axs[0,1].text(-3.4, 1.65*0.5, r'${}$'.format(fmt(np.amax(AE_arr[1,q_p,:,:,eta_p]),2)), color='1.0')
    axs[1,0].text(-3.4, 1.65*0.5, r'${}$'.format(fmt(np.amax(AE_arr[2,q_p,:,:,eta_p]),2)), color='1.0')
    axs[1,1].text(-3.4, 1.65*0.5, r'${}$'.format(fmt(np.amax(AE_arr[3,q_p,:,:,eta_p]),2)), color='1.0')
    axs[0,0].set_yticks([0,1])
    axs[0,1].set_yticks([0,1])
    axs[1,0].set_yticks([0,1])
    axs[1,1].set_yticks([0,1])
    axs[0,0].set_xticks([-4,0,4])
    axs[0,1].set_xticks([-4,0,4])
    axs[1,0].set_xticks([-4,0,4])
    axs[1,1].set_xticks([-4,0,4])


    for ax in axs.flat:
        ax.set(xlabel=r'$s$', ylabel=r'$\beta$')
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



    cb1=fig.colorbar(c1,ax=axs[0,0],ticks=v_ticks)
    cb2=fig.colorbar(c2,ax=axs[0,1],ticks=v_ticks)
    cb3=fig.colorbar(c3,ax=axs[1,0],ticks=v_ticks)
    cb4=fig.colorbar(c4,ax=axs[1,1],ticks=v_ticks)
    cb1.ax.set_yticklabels(v_label)
    cb2.ax.set_yticklabels(v_label)
    cb3.ax.set_yticklabels(v_label)
    cb4.ax.set_yticklabels(v_label)
    cb1.solids.set_edgecolor("face")
    cb2.solids.set_edgecolor("face")
    cb3.solids.set_edgecolor("face")
    cb4.solids.set_edgecolor("face")
    # cb1.ax.tick_params(size=0.5)
    # cb2.ax.tick_params(size=0.5)
    # cb3.ax.tick_params(size=0.5)
    # cb4.ax.tick_params(size=0.5)



    # cb2.set_label(r'$\hat{A}$',rotation=0,loc='bottom')
    # cb4.set_label(r'$\hat{A}$',rotation=0)

    fig.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12)
    plt.savefig('AE_plots_beta_q{}_eta{}.eps'.format(q_p/2+0.5, eta_p), format='eps',
                #This is recommendation for publication plots
                dpi=1200)

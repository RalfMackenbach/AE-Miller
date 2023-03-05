import AEtok.AE_tokamak_calculation as AEtok
import AEtok.Miller_functions as Mf
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
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

omn     = 1.0
eta     = 0
epsilon = 1/3
q       = 2.0
kappa   = 1.5
delta   = 0.0
dR0dr   =-0.5
s_q     = 'scan'
s_kappa = 0.0
s_delta = 0.0
alpha   = 'scan'
theta_res   = int(1e2 +1)
lam_res     = int(1e2)
L_ref       = 'major'
plot_steepest_descent = True




resolution  = 11
scan_arr    = np.linspace(-0.5,0.5,resolution)
s_arr       = np.empty_like(scan_arr)
alpha_arr   = np.empty_like(scan_arr)
AE_arr      = np.empty_like(scan_arr)
marker_vert = []

vals=[0.5,0]
for idx, val in enumerate(scan_arr):
    print('at step', idx+1, 'out of', len(scan_arr))
    #                               (omn,eta,epsilon,  q,kappa,delta,dR0dr, s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref,rho)
    fun = lambda x: -1*AEtok.calc_AE(omn,eta,epsilon,  q,kappa,  val,dR0dr,x[0],s_kappa,s_delta, x[1],theta_res,lam_res,L_ref)
    res = scipy.optimize.minimize(fun, vals)
    vals=res.x
    s_arr[idx] = vals[0]
    alpha_arr[idx] = vals[1]
    AE_arr[idx] = -1*fun(vals)
    # make marker
    MC       = Mf.MC(0.9,  q,kappa,  val,dR0dr,vals[0],s_kappa,s_delta, vals[1])
    theta_arr= np.linspace(-np.pi,+np.pi,100,endpoint=True)
    Rs       = list(Mf.R_s(theta_arr,MC)-1.0)
    Zs       = list(Mf.Z_s(theta_arr,MC))
    verts    = list(zip(Rs,Zs))
    marker_vert.append(verts)
    # plt.scatter(vals[1],vals[0],marker=verts,s=1000)



fig = plt.figure(figsize=(3.375, 2.3))
ax  = fig.gca()
# plt.scatter(alpha_arr,s_arr,c=AE_arr,cmap='plasma',marker=marker_vert)
ax.plot(alpha_arr,s_arr,c='black')

# make colormap stuff
plasma = mpl.colormaps['plasma']




# plot tokamak marker, and eigenvectors of Hessian if requested
for idx, val in enumerate(AE_arr):
    if idx%(int(resolution/4))==0:
        if plot_steepest_descent==True:
            ds  = 1e-2
            da  = 1e-2
            scan_val = scan_arr[idx]
            AE0 = AE_arr[idx]
            s0  = s_arr[idx]
            a0  = alpha_arr[idx]
            fun = lambda theta: AEtok.calc_AE(omn,eta,epsilon,  q,kappa,scan_val,dR0dr,s0+ds*np.sin(theta),s_kappa,s_delta,a0+da*np.cos(theta),theta_res,lam_res,L_ref)
            res = scipy.optimize.minimize(fun, 0)
            theta_min=res.x
            headscale=0.8
            headaxislenght=headscale*5
            headlength=headscale*5
            headwidth=headscale*3
            ax.quiver(a0,s0,np.cos(theta_min),np.sin(theta_min),angles='uv',zorder=3.5,
                      headaxislength=headaxislenght,headlength=headlength,headwidth=headwidth,color='g',scale=12)
        

        ax.scatter(alpha_arr[idx],s_arr[idx],color=plasma((AE_arr[idx]-AE_arr.min())/(AE_arr.max()-AE_arr.min())),marker=marker_vert[idx],s=150,zorder=2.5)


norm = mpl.colors.Normalize(vmin=AE_arr.min(), vmax=AE_arr.max())
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plasma),label=r'$\widehat{A}$')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$s$')
alpha_range=alpha_arr.max()-alpha_arr.min()
s_range    =s_arr.max()-s_arr.min()
p_fac = 0.1
ax.set_xlim((alpha_arr.min()-p_fac*alpha_range,alpha_arr.max()+p_fac*alpha_range))
ax.set_ylim((s_arr.min()-p_fac*s_range,s_arr.max()+p_fac*s_range))
ax.xaxis.set_tick_params(which='major', direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
ax.yaxis.set_tick_params(which='major', direction='in', top='on')
ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
#plt.axis('scaled')
plt.tight_layout()
plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/maximal-points.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
plt.show()
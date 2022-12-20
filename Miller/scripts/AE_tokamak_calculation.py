import numpy                as np
import Miller_functions     as Mf
import AE_tokamak_functions as AEtf
import matplotlib.pyplot    as plt
from    matplotlib          import  cm
import  matplotlib.colors   as      mplc
from numba import vectorize, njit





def calc_AE(omn,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res=100,lam_res=1000,del_sign=0.0,L_ref='major',rho=1.0,plot=False):
    """
    omn         -   number density gradient
    eta         -   ratio between temperature and density gradient omt/omn
    epsilon     -   ratio between radial coordinate r and major radius
    q           -   safety factor, inverse rotational transform
    kappa       -   elongation, kappa = major radius / minor radius (of ellipse)
    delta       -   triangularity as defined by Miller
    dR0dr       -   How quickly R0 changes with flux surfaces
    s_q         -   Magnetic shear
    s_kappa     -   Radial derivative of kappa (r/kappa * dkappa/dr)
    s_delta     -   Radial derivative of delta (r * d arcsin(delta)/dr)
    alpha       -   Dimensionless pressure gradient, i.e. Shafranov shifts
    theta_res   -   Resolution of theta array
    lam_res     -   Resolution for the lambda (pitch angle) integral
    del_sign    -   Padding around singularity
    L_ref       -   Decides the normalisation. If 'major', omn = R0/n dn/dr and
                    rho* = rho/R_0. If 'minor', omn = a_minor/n dn/dr and rho* = rho/a_minor.
    rho         -   ratio between radial coordinate r and a_minor
    plot        -   Make plot showing AE per bounce well
    """

    # create Miller class
    MC       = Mf.MC(epsilon, q, kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha)

    vint = np.vectorize(AEtf.integral_over_z, otypes=[np.float64])
    eps  = MC.epsilon
    xi   = MC.xi
    #############################################################################
    #
    #
    #
    # Calculate!
    #
    #
    #
    #############################################################################


    # various arrays needed for bounce precession frequency
    theta_arr   = np.linspace(-np.pi,+np.pi,theta_res,endpoint=True)
    b_arr       = Mf.bs(theta_arr, MC)
    sinu        = Mf.sinu(theta_arr, MC)
    rdbdrho     = Mf.rdbdrho(theta_arr,MC)
    rdbpdrho    = Mf.rdbpdrho(theta_arr,MC)
    Rc          = Mf.R_c(theta_arr,MC)
    bps         = Mf.bps(theta_arr,MC)
    Rs          = Mf.R_s(theta_arr,MC)
    l_theta     = Mf.ltheta(theta_arr,MC)

    lam_arr     = np.delete(np.linspace(1/np.amax(b_arr),1/np.amin(b_arr),lam_res+1,endpoint=False),0)
    AE_arr      = np.empty_like(lam_arr)
    ae_list     = []
    oml_list    = []

    if L_ref == 'minor':
        omn = rho*omn/epsilon

    for idx, lam_val in enumerate(lam_arr):
        averaging_arr   = ( 2. * ( 1. - lam_val*b_arr ) * ( rdbdrho - rdbpdrho - 1./Rc ) - lam_val * b_arr * rdbdrho ) / ( MC.epsilon * bps * Rs ) 
        f_arr_numer     = averaging_arr * l_theta * b_arr / bps
        f_arr_denom     = l_theta * b_arr/ bps
        num,den         = AEtf.omega_lam(theta_arr,b_arr,f_arr_numer,f_arr_denom,lam_val)
        oml             = np.asarray(num)/np.asarray(den)
        g_hat_eps       = np.asarray(den)*np.sqrt(eps)/xi
        int_z           = vint(omn / (oml) * (1. - 3./2. * eta),1. - omn / (oml) * eta)
        ae_list.append(int_z * oml**2.0 * g_hat_eps / eps)
        AE_arr[idx]     = np.sum(int_z * oml**2.0 * g_hat_eps / eps)
        oml_list.append(list(oml))

    if plot == True:
        bw_list = []
        k2_arr   = ((1 - lam_arr*np.amin(b_arr))*np.amax(b_arr)/(np.amax(b_arr)-np.amin(b_arr)) )
        for lam_val in lam_arr:
            t_wells, _ = AEtf.bounce_wells(theta_arr, b_arr, lam_val)
            bw_list.append(t_wells)
        length = max(map(len, oml_list))
        oml_arr=np.array([xi+[None]*(length-len(xi)) for xi in oml_list])
        
        fig ,axs = plt.subplots(2,2)
        fig.set_size_inches(2*8,8)
        fig.suptitle(r'Miller parameters: $\epsilon=${}, $q=${}, $\kappa=${}, $\delta=${}, $\partial_r R_0 =${}, $s_q=${}, $s_\kappa =${}, $s_\delta =${} , $\alpha=${}'.format(epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha))
        axs[0,0].plot(k2_arr,oml_arr)
        axs[0,0].set_xlabel(r'$k^2$')
        axs[0,0].set_xlim((0,1))
        axs[0,0].set_ylabel(r'$\omega_\lambda$')
        axs[0,0].set_title('precession')
        axs[0,1].plot(Rs,Mf.Z_s(theta_arr,MC))
        axs[0,1].set_aspect('equal')
        axs[0,1].set_title('cross section')
        axs[0,1].set_xlabel(r'$R/R_0$')
        axs[0,1].set_ylabel(r'$Z/R_0$')
        axs102 = axs[1,0].twinx()
        axs[1,0].plot(theta_arr,b_arr,color='green')
        axs[1,0].set_xlabel(r'$\theta$')
        axs[1,0].set_ylabel(r'$B/B_0$')
        axs[1,0].set_title('grad B drive')
        axs[1,0].set_xlim((-np.pi,np.pi))
        axs102.set_ylabel(r'$\frac{r}{B} \frac{\partial B}{\partial\rho}$')
        axs102.plot(theta_arr,rdbdrho,color='red')
        axs102.plot(theta_arr,theta_arr*0,color='red',linestyle='dashed')
        axs112 = axs[1,1].twinx()
        axs[1,1].plot(theta_arr,b_arr,color='green')
        axs[1,1].set_xlabel(r'$\theta$')
        axs[1,1].set_ylabel(r'$B/B_0$')
        axs[1,1].set_title('curvature drive')
        axs[1,1].set_xlim((-np.pi,np.pi))
        axs112.set_ylabel(r'$\frac{r}{B} \frac{\partial B}{\partial\rho} - \frac{r}{B_p} \frac{\partial B_p}{\partial\rho} - r/R_c $')
        axs112.plot(theta_arr, -1.0* (rdbdrho - rdbpdrho - 1/Rc),color='red')
        axs112.plot(theta_arr,theta_arr*0,color='red',linestyle='dashed')
        plt.tight_layout()
        plot_AE_per_bouncewell(theta_arr,b_arr,  rdbdrho  ,lam_arr,bw_list,ae_list,1.0)

    fluxtube_vol = MC.xi_2 / MC.xi

    if L_ref == 'major':
        return np.sqrt(epsilon) * q**2.0 * np.trapz(AE_arr,lam_arr) / fluxtube_vol
    if L_ref == 'minor':
        return (epsilon/rho)**2.0 * np.sqrt(epsilon) * q**2.0 * np.trapz(AE_arr,lam_arr) / fluxtube_vol 



def plot_AE_per_bouncewell(theta_arr,b_arr,dbdx_arr,lam_arr,bw,ae_list,n_pol):
    c = 0.5
    # shift by pi
    fig ,ax = plt.subplots()
    fig.set_size_inches(8*5, 3.75)
    ax.set_xlim(min(theta_arr),max(theta_arr))

    list_flat = []
    for val in ae_list:
        list_flat.extend(val)

    max_val = max(list_flat)
    cm_scale = lambda x: x
    colors_plot = [cm.plasma(cm_scale(np.asarray(x) * 1.0/max_val)) for x in ae_list]

    # iterate over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        b_val = 1/lam

        # iterate over all bounce wells
        for idx_bw, _ in enumerate(ae_list[idx_lam]):
            # check if well crosses boundary
            if(bw[idx_lam][idx_bw][0] > bw[idx_lam][idx_bw][1]):
                ax.plot([bw[idx_lam][idx_bw][0], max(theta_arr)], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                ax.plot([min(theta_arr), bw[idx_lam][idx_bw][1]], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
            # if not normal plot
            else:
                ax.plot([bw[idx_lam][idx_bw][0], bw[idx_lam][idx_bw][1]], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

    ax.plot(theta_arr,b_arr,color='black',linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(theta_arr, dbdx_arr, 'red')
    ax.set_ylabel(r'$B$')
    ax2.set_ylabel(r'$r \partial_\rho b$',color='red')
    ax2.plot(theta_arr,theta_arr*0.0,linestyle='dashed',color='red')
    ax2.tick_params(axis='y', colors='black',labelcolor='red',direction='in')
    ax.set_xlabel(r'$\theta/n_{pol}$')
    ax.set_xticks([n_pol*-np.pi,n_pol*-np.pi/2,0,n_pol*np.pi/2,n_pol*np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$',r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='both',direction='in')
    ax.set_title(r'AE distribution')
    cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_val, clip=False), cmap=cm.plasma), ticks=[0, max_val], ax=ax,location='bottom',label=r'$\widehat{A}_\lambda$') #'%.3f'
    cbar.ax.set_xticklabels([0, round(max_val, 1)])
    plt.show()

#       omn, eta, eps,  q,   kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha
# calc_AE(3.0, 0.0, 0.18, 1.4, 1.0,   0.0,   0.0,   0.78, 0.0,     0.0,     0.0,theta_res=1000,lam_res=1000,del_sign=0.0,L_ref='major',rho=1.0,plot=True)
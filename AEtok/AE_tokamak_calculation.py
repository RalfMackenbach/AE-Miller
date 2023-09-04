import  numpy                   as      np
from    BAD                     import  bounce_int 
import  AEtok.Miller_functions  as      Mf
import  matplotlib.pyplot       as      plt
from    matplotlib              import  cm
import  matplotlib.colors       as      mplc
from    scipy.special           import  erf, ellipe, ellipk
from    scipy.integrate         import  quad



def AE_per_lam(c0,c1,tau_b,wlam):
    r"""
    function containing the integral over z for exactly omnigenous systems.
    This is the available energy per lambda.
    """
    condition1 = np.logical_and((c0>=0),(c1<=0))
    condition2 = np.logical_and((c0>=0),(c1>0))
    condition3 = np.logical_and((c0<0),(c1<0))
    ans = np.zeros(len(c1))
    ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
    ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
    ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )
    return ans*tau_b*wlam**2





def all_drifts(f,h0,h1,x):
    r"""
    ``all_drifts`` does the bounce integral
    and wraps the root finding routine into one function.
    Does the bounce int for three functions
    I0 = ∫h0(x)/sqrt(f(x))  dx
    I1 = ∫h1(x)/sqrt(f(x)) dx
    and returns all of these values.
    Uses gtrapz, also returns roots
     Args:
        f: function or array containing f
        h0:function or array containing h0
        h1:function of array containing h1
        x :x values of h
    """
    # if false use array for root finding
    index,root = bounce_int._find_zeros(f,x,is_func=False)
    # check if first well is edge, if so roll
    first_well = bounce_int._check_first_well(f,x,index,is_func=False)
    if first_well==False:
        index = np.roll(index,1)
        root  = np.roll(root,1)
    # do bounce integral
    I0 = bounce_int._bounce_integral(f,h0,x,index,root,is_func=False,sinhtanh=False)
    I1 = bounce_int._bounce_integral(f,h1,x,index,root,is_func=False,sinhtanh=False)
    return [I0,I1], root





def calc_AE(omn,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res=1001,L_ref='major',A=3.0,rho=1.0,int_meth='quad',lam_res=1000,plot_precs=False,plot_dist=False,Cr_model='r_eff',output='AE'):
    """
    ``calc_AE`` calculates the AE of a Miller tokamak.
    Takes as input the following set of parameters
     Args:
        omn:        -R0 * d ln(n) / dr
        eta:        ratio of (dln(T)/dr) / (dln(n)/dr)
        epsilon:    r/R0, where r is the location of the flux surface. Ignored if L_ref is 'minor'
        q:          safety factor
        kappa:      elongation
        delta:      triangularity
        dR0dr:      how quickly central location R0 changes with flux surface label r
        s_q:        magnetic shear, r/q * dq/dr
        s_kappa:    variation in kappa, r/kappa * d kappa/dr
        s_delta:    variation in delta, r * d arcsin(delta)/dr
        alpha:      alpha_MHD, -epsilon * r dp/dr / (poloidal magnetic pressure)**2
        theta_res:  number of theta nodes used for gtrapz
        L_ref:      reference lengtscale, either 'major', or 'minor'.
                    If chosen 'minor', one has to pass the kwargs for the device
                    aspect ratio, and rho as well. In 'minor' omn is assumed to be
                    omn = -a_minor/n * dn/dr, and epsilon is ignored in favour of
                    rho and aspect ratio.
        A:          Device aspect ratio (R0/a_minor), only needed if L_ref='minor'
        rho:        Normalized radial location r/a, only needed if L_ref='minor'
        int_meth:   Integration method over lambda. 'trapz' uses trapezoidal rule,
                    'quad' uses quadrature. If 'trapz', one can set the resolution
                    via lam_res. If one wants to use plot_precs, trapz needs to
                    be used.
        lam_res:    lambda resolution for trapezoidal integral over lambda.
        plot_precs: Boolean. If set to True, one plots the AE per lambda.
                    Can be useful for debugging and physics insights. Needs
                    'trapz' for int_meth.
    """

    # Do conversions between minor and major here
    # For minor, we convert everything back to 
    # equivalent major
    prefac = 1.0
    if L_ref=='minor':
        omn = A * omn       # R0 * df/dr = R0/a * a df/dr = A * omn
        epsilon = rho / A   # epsilon = r/R0 = (r/a) / (R0/a) = rho / A
        prefac  = A**(-2)   # rho_g/R_0 = rho_g/a * a/R0 = rho_* / A 
    
    MC   = Mf.MC(epsilon, q, kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha)
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
    rdbdrho     = Mf.rdbdrho(theta_arr,MC)
    rdbpdrho    = Mf.rdbpdrho(theta_arr,MC)
    Rc          = Mf.R_c(theta_arr,MC)
    bps         = Mf.bps(theta_arr,MC)
    Rs          = Mf.R_s(theta_arr,MC)
    l_theta     = Mf.ltheta(theta_arr,MC)

    
    if int_meth=='trapz':
        lam_arr     = np.delete(np.linspace(1/np.amax(b_arr),1/np.amin(b_arr),lam_res+1,endpoint=False),0)
        AE_arr      = np.empty_like(lam_arr)
        ae_list     = []
        oml_list    = []
        root_list   = []
        tau_list    = []

        for idx, lam_val in enumerate(lam_arr):
            averaging_arr   = ( 2. * ( 1. - lam_val*b_arr ) * ( rdbdrho - rdbpdrho - 1./Rc ) - lam_val * b_arr * rdbdrho ) / ( MC.epsilon * bps * Rs )
            dldtheta        = l_theta * b_arr / bps
            f_arr_numer     = averaging_arr * dldtheta
            f_arr_denom     = l_theta * b_arr/ bps
            f               = 1 - lam_val * b_arr
            ave,roots       = all_drifts(f,f_arr_denom,f_arr_numer,theta_arr)
            den,num         = ave[0], ave[1]
            oml             = np.asarray(num)/np.asarray(den)
            g_hat_eps       = np.asarray(den)*np.sqrt(eps)/xi
            c0              = np.asarray( omn / (oml) * (1. - 3./2. * eta) )
            c1              = np.asarray(1. - omn / (oml) * eta)
            int_z           = AE_per_lam(c0,c1,g_hat_eps,oml)/MC.epsilon
            ae_list.append(int_z)
            AE_arr[idx]     = np.sum(int_z)
            oml_list.append(list(oml))
            root_list.append(list(roots))
            tau_list.append(list(g_hat_eps))

        fluxtube_vol = MC.xi_2 / MC.xi

        int_res = np.trapz(AE_arr,lam_arr)

        if plot_precs==True:
            plot_precession(oml_list,root_list,theta_arr,b_arr,lam_arr,ae_list)

        # plot integrand
        if plot_dist==True:
            plot_integrand(oml_list,tau_list,lam_arr,omn,eta,b_arr)

    if int_meth=='quad':
        def quad_int(lam_val):
            averaging_arr   = ( 2. * ( 1. - lam_val*b_arr ) * ( rdbdrho - rdbpdrho - 1./Rc ) - lam_val * b_arr * rdbdrho ) / ( MC.epsilon * bps * Rs )
            dldtheta        = l_theta * b_arr / bps
            f_arr_numer     = averaging_arr * dldtheta
            f_arr_denom     = dldtheta
            f               = 1 - lam_val * b_arr
            ave,roots       = all_drifts(f,f_arr_denom,f_arr_numer,theta_arr)
            den,num         = ave[0], ave[1]
            oml             = np.asarray(num)/np.asarray(den)
            g_hat_eps       = np.asarray(den)*np.sqrt(eps)/xi
            c0              = np.asarray( omn / (oml) * (1. - 3./2. * eta) )
            c1              = np.asarray(1. - omn / (oml) * eta)
            int_z           = AE_per_lam(c0,c1,g_hat_eps,oml)/MC.epsilon
            return np.sum(int_z)

        lam_min     = 1./b_arr.max()
        lam_max     = 1./b_arr.min()

        int_res, err = quad(quad_int,lam_min,lam_max,epsrel=1e-3,epsabs=1e-20,limit=1000)

        fluxtube_vol = MC.xi_2 / MC.xi

    # set C_r model
    if Cr_model=='r_miller':
        Cr = 1.0
    if Cr_model=='r_eff':
        Cr = 1.0/np.sqrt(MC.psi_tor)

    AE = Cr**2 * prefac * np.sqrt(epsilon) * int_res / fluxtube_vol * 1/8

    if output=='AE':
        return AE
    
    if output=='Qe':
        # PRL constant of proportionality
        C = 1072.6914449397775
        return C * AE**(3/2)
    
    



def plot_precession(walpha,roots,theta,b_arr,lam_arr,ae_per_lam):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    import matplotlib.pyplot as plt
    import matplotlib        as mpl
    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)

    k2 = (b_arr.max() - lam_arr * b_arr.min() * b_arr.max())/(b_arr.max() - b_arr.min())

    # reshape for plotting
    walp_arr = np.nan*np.zeros([len(walpha),len(max(walpha,key = lambda x: len(x)))])
    for i,j in enumerate(walpha):
        walp_arr[i][0:len(j)] = j
    wpsi_arr = np.nan*np.zeros([len(walpha),len(max(walpha,key = lambda x: len(x)))])
    for i,j in enumerate(walpha):
        wpsi_arr[i][0:len(j)] = j
    alp_l  = np.shape(walp_arr)[1]
    k2_arr = np.repeat(k2,alp_l)
    fig, ax = plt.subplots(2, 1, figsize=(2*3.5, 5.0),gridspec_kw={'height_ratios': [1.2, 2]})
    ax[0].scatter(k2_arr,walp_arr,s=0.2,marker='.',color='black',facecolors='black')
    ax[0].plot(k2,0.0*k2,color='red',linestyle='dashed')
    ax[0].set_xlim(0,1)
    ax[0].set_xlabel(r'$k^2$')
    ax[0].set_ylabel(r'$\omega_\lambda$',color='black')


    # now do plot as a function of bounce-angle
    walpha_bounceplot = []
    roots_bounceplot  = []
    for lam_idx, lam_val in enumerate(lam_arr):
        root_at_lam = roots[lam_idx]
        walpha_at_lam= walpha[lam_idx]
        roots_bounceplot.extend(root_at_lam)
        for idx in range(len(walpha_at_lam)):
            walpha_bounceplot.extend([walpha_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])

    roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))

    lam_arr   = np.asarray(lam_arr).flatten()
    ae_per_lam = ae_per_lam
    list_flat = []
    for val in ae_per_lam:
        list_flat.extend(val)
    max_ae_per_lam = max(list_flat)

    cm_scale = lambda x: x
    colors_plot = [cm.plasma(cm_scale(np.asarray(x) * 1.0/max_ae_per_lam)) for x in ae_per_lam]

    # iterate over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        b_val = 1/lam

        # iterate over all bounce wells
        for idx_bw, _ in enumerate(ae_per_lam[idx_lam]):
            bws = roots[idx_lam]
            # check if well crosses boundary
            if(bws[2*idx_bw] > bws[2*idx_bw+1]):
                ax[1].plot([bws[2*idx_bw], max(theta)], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                ax[1].plot([min(theta), bws[2*idx_bw+1]], [b_val, b_val],color=colors_plot[idx_lam][idx_bw])
            # if not normal plot
            else:
                ax[1].plot([bws[2*idx_bw], bws[2*idx_bw+1]], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

    ax[1].plot(theta,b_arr,color='black',linewidth=2)
    ax2 = ax[1].twinx()
    ax2.plot(roots_ordered, walpha_bounceplot, 'red')
    ax[1].set_ylabel(r'$B$')
    ax2.set_ylabel(r'$\omega_\lambda$',color='red')
    ax2.plot(theta,theta*0.0,linestyle='dashed',color='red')
    ax2.tick_params(axis='y', colors='black',labelcolor='red',direction='in')
    ax[1].set_xlabel(r'$\theta$')
    ax[1].set_xlim(theta.min(),theta.max())
    ax[1].tick_params(axis='both',direction='in')
    plt.subplots_adjust(left=0.1, right=0.88, top=0.99, bottom=0.08)
    cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_ae_per_lam, clip=False), cmap=cm.plasma), ticks=[0, max_ae_per_lam], ax=ax[1],location='bottom',label=r'$\widehat{A}_\lambda/\widehat{A}_{\lambda,\mathrm{max}}$') #'%.3f'
    cbar.ax.set_xticklabels([0, 1])
    plt.savefig('ae_per_lam.png', format='png',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()





def plot_integrand(wlam,tau_b,lam_arr,omn,eta,b_arr):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    import matplotlib.pyplot as plt
    import matplotlib        as mpl
    from matplotlib.colors import LogNorm
    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)

    # first construct the integrand function   
    def integrand(wlam,tau_b,omn,eta,z):
        omega_drift = omn * (1 + eta * (z - 3/2))
        ramp_fun = np.heaviside(omega_drift/(z*wlam) - 1, 0) *  (omega_drift/(z*wlam) - 1)
        return ramp_fun * np.exp(-z) * z**(5/2) * wlam**2 # * tau_b
    
    # now construct the integrand array
    z = np.logspace(-2,2,1000)
    integrand_arr = np.zeros((len(lam_arr),len(z)))
    # loop over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        # loop over all values of z
        for idx_z, z_val in enumerate(z):
            integrand_arr[idx_lam,idx_z] = np.sum(integrand(np.asarray(wlam[idx_lam]),np.asarray(tau_b[idx_lam]),omn,eta,z_val))

    # do integral over z
    integrand_arr_lam = np.trapz(integrand_arr,z,axis=1)

    # make two plot: one showing the full integrand, and one showing the integral over z
    fig, ax = plt.subplots(1,2,sharey=True,gridspec_kw={'width_ratios': [2.5, 1]})
    fig.set_size_inches(5/6*6, 5/6*3.5)
    ax[0].set_xscale('log')

    # find minimal value of b_arr
    b_min = b_arr.min()

    # make contour of integrand
    cmap = 'gist_heat_r'
    log = False
    lam_plot = np.arcsin(lam_arr*b_min)
    # convert to degrees
    lam_plot = lam_plot * 180/np.pi
    if log==True:
        # make levels logarithmic
        levels = np.logspace(np.log10(integrand_arr.max()/1000),np.log10(integrand_arr.max()),100)
        cnt = ax[0].contourf(z,lam_plot,integrand_arr,levels=levels,cmap=cmap,norm = LogNorm())
    else:
        cnt = ax[0].contourf(z,lam_plot,integrand_arr,cmap=cmap,levels=100)
    ax[0].set_xlabel(r'$z=\epsilon/T_0$')
    ax[0].set_ylabel(r'$\theta_{\rm pitch}$')
    ax[0].set_title(r'$\int \epsilon (f_i - f_g) \mathrm{d}\psi \mathrm{d}\alpha \Big/ \int \mathrm{d}\psi \mathrm{d}\alpha $')
    ax[0].set_ylim(lam_plot.min(),lam_plot.max())
    ax[0].set_xlim(z.min(),z.max())

    # add colorbar, extend bottom
    # cbar = fig.colorbar(cnt, ax=ax[0],location='bottom')

    # make line plot of integral over z
    ax[1].plot(integrand_arr_lam,lam_plot,color='red')
    # have small buffer on left and right
    dif = integrand_arr_lam.max() - integrand_arr_lam.min()
    ax[1].set_xlim(0.0,integrand_arr_lam.max() + 0.05*dif)
    ax[1].set_ylim(lam_plot.min(),lam_plot.max())
    # remove right spline 
    ax[1].spines['right'].set_visible(False)



    # show plot
    plt.tight_layout()
    plt.show()




    





def CHM(k2,s,alpha,q):
    '''
    ``CHM`` Contains the analytical bounce-averaged drift
    of Connor, Hastie, and Martin, for a s-alpha tokamak.
    '''
    E   = ellipe(k2, out=np.zeros_like(k2))
    K   = ellipk(k2, out=np.zeros_like(k2))
    EK  = E/K
    G1 = EK - 1/2
    G2 = EK + k2 - 1
    G3 = EK * (2 * k2 - 1) +1 - k2 
    wlam = 2 * ( G1 - alpha/(4*q**2) + 2 * s * G2 - 2 * alpha / 3 * G3 )
    return np.asarray(wlam), np.asarray(K)



def calc_AE_salpha(omn,eta,epsilon,q,s_q,alpha,L_ref='major',A=3.0,rho=1.0,int_meth='quad',lam_res=1000,output='AE'):
    '''
    ``calc_AE`` calculates the AE of a Miller tokamak.
    Takes as input the following set of parameters
     Args:
        omn:        R0 * d ln(n) / dr
        eta:        ratio of (dln(T)/dr) / (dln(n)/dr)
        epsilon:    r/R0, where r is the location of the flux surface.
        q:          safety factor
        s_q:        magnetic shear, r/q*dq/dr
        alpha:      alpha_MHD, epsilon * r dp/dr / (poloidal magnetic pressure)**2
        lam_res:    number of lambda values used for integral over lambda
        L_ref:      reference lengtscale, either 'major', or 'minor'.
                    If chosen 'minor', one has to pass the kwargs for the device
                    aspect ratio, and rho as well. In 'minor' omn is assumed to be
                    omn = a_minor/n * dn/dr, and epsilon is ignored in favour of
                    rho and aspect ratio.
        A:          Device aspect ratio (R0/a_minor), only needed if L_ref='minor'
        rho:        Normalized radial location r/a, only needed if L_ref='minor'
        int_meth:   Integration method over lambda. 'trapz' uses trapezoidal rule,
                    'quad' uses quadrature. If 'trapz', one can set the resolution
                    via lam_res. If one wants to plot AE per lambda, trapz needs to
                    be used.
        lam_res:    lambda resolution for trapezoidal integral over lambda.
    '''
    prefac = 1.0
    if L_ref=='minor':
        omn = A * omn       # R0 * df/dr = R0/a * a df/dr = A * omn
        epsilon = rho / A   # epsilon = r/R0 = (r/a) / (R0/a) = rho / A
        prefac  = A**(-2)   # rho_g/R_0 = rho_g/a * a/R0 = rho_* / A 


    if int_meth=='trapz':
        k2_arr  = np.linspace(0,1,lam_res+2)
        k2_arr  = k2_arr[1:-1]
        wlam, K = CHM(k2_arr,s_q,alpha,q)
        c0              = np.asarray( omn / (wlam) * (1. - 3./2. * eta) )
        c1              = np.asarray(1. - omn / (wlam) * eta)
        ae_per_lam      = AE_per_lam(c0,c1,K,wlam)
        int_res         = np.trapz(ae_per_lam,k2_arr)

    if int_meth=='quad':
        def quad_int(k2_val):
            wlam, K = CHM(k2_val,s_q,alpha,q)
            c0              = np.asarray([ omn / (wlam) * (1. - 3./2. * eta) ])
            c1              = np.asarray([1. - omn / (wlam) * eta])
            ae_per_lam      = AE_per_lam(c0,c1,K,wlam)
            return ae_per_lam
        int_res, err = quad(quad_int,0,1,epsrel=1e-4)

    # create AE
    AE = prefac * 2 * np.sqrt(2)/np.pi * np.sqrt(epsilon) * int_res  * 1/8

    if output=='AE':
        return AE

    if output=='Qe':
        # PRL constant of proportionality
        C = 1072.6914449397775
        return C * AE**(3/2)
    

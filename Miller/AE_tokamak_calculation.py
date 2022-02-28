import numpy                as np
import Miller_functions     as Mf
import AE_tokamak_functions as AEtf
import matplotlib.pyplot    as plt
from numba import vectorize, njit





def calc_AE(omn,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign):



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

    for idx, lam_val in enumerate(lam_arr):
        f_arr_numer     = (( 2 * ( 1 - lam_val*b_arr ) * ( rdbdrho - rdbpdrho - 1/Rc ) - lam_val * b_arr * rdbdrho ) / MC.epsilon) * l_theta * b_arr/ bps
        f_arr_denom     = l_theta * b_arr/ bps
        num,den         = AEtf.omega_lam(theta_arr,b_arr,f_arr_numer,f_arr_denom,lam_val)
        oml             = np.asarray(num)/np.asarray(den)
        g_hat_eps       = np.asarray(den)*np.sqrt(eps)/xi
        int_z           = vint(omn / (oml) * (1. - 3./2. * eta),1. - omn / (oml) * eta)
        AE_arr[idx]     = np.sum(int_z * oml**2.0 * g_hat_eps / eps)

    # length = max(map(len, oml_list))
    # oml_arr = np.array([xi+[None]*(length-len(xi)) for xi in oml_list])
    # den_arr = np.array([xi+[None]*(length-len(xi)) for xi in den_list])
    # c0 = omn / (oml_arr) * (1. - 3./2. * eta)
    # c1 = 1. - omn / (oml_arr) * eta
    # g_hat_epsilon   = den_arr*np.sqrt(MC.epsilon)/MC.xi
    # print(oml_arr)
    # AE_integrand    = AEtf.integrand_full(c0,c1,oml_arr,g_hat_epsilon,MC)
    return np.trapz(AE_arr,lam_arr)

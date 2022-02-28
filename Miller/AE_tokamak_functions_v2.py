import  numpy               as      np
import  Miller_functions    as      Mf
from    scipy.integrate     import  dblquad, quad
from    scipy.optimize      import  fsolve
from    scipy               import  special
from    numba               import  vectorize, njit
from    numba               import  int32, float32    # import the types


#############################################################################
#
#
#
# Here we define functions relevant for the AE calculation
#
#
#
#############################################################################


# ramp function for convenience
def ramp(x):
    return (x + np.abs(x))/2


# integrand for bounce time
def integrand_bounce_time(theta,lam,MC,del_sing):
    return np.real(Mf.ltheta(theta,MC)*Mf.bs(theta,MC)/Mf.bps(theta,MC) / np.sqrt( 1 - lam * Mf.bs(theta,MC) + 0j) * np.heaviside(1 - lam * Mf.bs(theta,MC) - del_sing, 0))


# integrand for bounce drifts
def integrand_bounce_drift(theta,lam,MC,del_sing):
    return np.real((1/MC.epsilon) * (Mf.ltheta(theta,MC)*Mf.bs(theta,MC)/Mf.bps(theta,MC) * (
    2 * (1 - lam*Mf.bs(theta,MC) ) * ( Mf.rdbdrho(theta,MC) - Mf.rdbpdrho(theta,MC) - Mf.R_c(theta,MC) ) - lam * Mf.bs(theta,MC) * Mf.rdbdrho(theta,MC)
    ) / np.sqrt( 1 - lam * Mf.bs(theta,MC) + 0j) * np.heaviside(1 - lam * Mf.bs(theta,MC) - del_sing, 0)) / (Mf.bps(theta,MC) * Mf.R_s(theta,MC)))



def bounce_time(lam,MC,del_sing):
    return quad(lambda theta: integrand_bounce_time(theta,lam,MC,del_sing),  0, np.pi,limit=int(1e6),epsabs=1.49e-08)



def bounce_drift(lam,MC,del_sing):
    return quad(lambda theta: integrand_bounce_drift(theta,lam,MC,del_sing),  0, np.pi,limit=int(1e6),epsabs=1.49e-08)


def integral_over_z(c0,c1):
    if (c0>=0) and (c1<=0):
        return 2 * c0 - 5 * c1
    if (c0>=0) and (c1>0):
        return (2 * c0 - 5 * c1) * special.erf(np.sqrt(c0/c1)) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0 + 15 * c1 ) * np.sqrt(c0/c1) * np.exp( - c0/c1 )
    if (c0<0)  and (c1<0):
        return ( (2 * c0 - 5 * c1) * (1 - special.erf(np.sqrt(c0/c1))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0 + 15 * c1 ) * np.sqrt(c0/c1) * np.exp( - c0/c1 ) )
    else:
        return 0.



def integrand_full(c0,c1,om_lam,g_hat_epsilon,MC):
    vint = np.vectorize(integral_over_z, otypes=[np.float64])

    int_z = vint(c0,c1)

    return int_z * om_lam**2.0 * g_hat_epsilon / MC.epsilon

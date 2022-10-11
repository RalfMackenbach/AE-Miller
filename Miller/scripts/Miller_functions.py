import scipy
import scipy.special
from   scipy                import integrate
import numpy                as np
from   scipy.integrate      import dblquad, quad
from   numba                import int32, float32    # import the types
from   numba.experimental   import jitclass


#############################################################################
#
#
#
# We will store all information in a "Miller" class. This class has all
# free parameters stored as properties
#
#
#
# The free parameters in a Miller class are the following
# epsilon:  inverse aspect ratio
# q:        safety factor
# kappa:    elongation (kappa=1 is circular)
# x:        arcsin(triangularity)
# dR0dr:    displacement of flux-surfaces, related to Shafranov shift
# s_q:      shear of safety factor
# s_kappa:  variation of elongation with radial coordinate
# s_delta:  variation of triangularity with radial coordinate
# alpha:    dimensionless pressure gradient
#
#
#
#############################################################################


#############################################################################
#
#
#
# In the section below we define all the relevant functions needed to define
# the Miller class geometry (so that includes derived quantities such as
# gamma and f'(psi)*R0)
#
#
#
#############################################################################


# integrand for gamma.
def gamma_integrand(theta,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
  return (kappa*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta))))/(1 + epsilon*np.cos(theta + x*np.sin(theta)))


# integrand first term of eq (21)
def C1_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return  (kappa**2*(kappa*np.cos(theta)*(1 + x*np.cos(theta))**2*np.cos(theta + x*np.sin(theta)) + kappa*np.sin(theta)*np.sin(theta + x*np.sin(theta)))*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))**2)/((1 + epsilon*np.cos(theta + x*np.sin(theta)))*(kappa**2*np.cos(theta)**2 + (1 + x*np.cos(theta))**2*np.sin(theta + x*np.sin(theta))**2)**2)


# integrand second term of eq (21)
def C2_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return -((kappa**3*np.cos(theta)*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))**2)/((1 + epsilon*np.cos(theta + x*np.sin(theta)))**2*(kappa**2*np.cos(theta)**2 + (1 + x*np.cos(theta))**2*np.sin(theta + x*np.sin(theta))**2)))


# integrand third term of eq (21)
def C3_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return (kappa**3*(1 + epsilon*np.cos(theta + x*np.sin(theta)))*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))**3)/(kappa**2*np.cos(theta)**2 + (1 + x*np.cos(theta))**2*np.sin(theta + x*np.sin(theta))**2)


# integrand fourth term of eq (21)
def C4_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return (kappa**4*(1 + epsilon*np.cos(theta + x*np.sin(theta)))*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))**4)/(kappa**2*np.cos(theta)**2 + (1 + x*np.cos(theta))**2*np.sin(theta + x*np.sin(theta))**2)**1.5


# function for fpR0
def fpR0_func(gamma,C1,C2,C3,C4,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return (q*gamma*(alpha*C3 + 2*epsilon*(2*C1 + 2*C2*epsilon + s_q*gamma)))/(2.*(C4*epsilon*q**2 + epsilon**3*gamma**3))


# function for xi
def xi_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha):
    return kappa*(1 + epsilon*np.cos(theta + x*np.sin(theta)))*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))*np.sqrt((1 + (epsilon**2*gamma**2*(kappa**2*np.cos(theta)**2 + (1 + x*np.cos(theta))**2*np.sin(theta + x*np.sin(theta))**2))/(kappa**2*q**2*(np.cos(theta)*(dR0dr + np.cos(theta + x*np.sin(theta))) + (1 + s_kappa + (-s_delta + x + s_kappa*x)*np.cos(theta))*np.sin(theta)*np.sin(theta + x*np.sin(theta)))**2))/(1 + epsilon*np.cos(theta + x*np.sin(theta)))**2)


# spec = [
#     ('epsilon', int32),             # a simple scalar field
#     ('q', int32),                   # a simple scalar field
#     ('kappa', int32),               # a simple scalar field
#     ('delta', int32),               # a simple scalar field
#     ('dR0dr', int32),               # a simple scalar field
#     ('s_q', int32),                 # a simple scalar field
#     ('s_kappa', int32),             # a simple scalar field
#     ('s_delta', int32),             # a simple scalar field
#     ('alpha', int32),               # a simple scalar field
#     ('x', int32),                   # a simple scalar field
#     ('gamma', int32),               # a simple scalar field
#     ('fpR0', int32),                # a simple scalar field
#     ('sigma', int32),               # a simple scalar field
# ]
#
# @jitclass(spec)
class MC:
    """
    Miller class (MC) that stores all free Miller parameters in a pythonic way.
    The Miller geometry has the following parameters:
    epsilon:    inverse aspect ratio
    q:          safety factor
    kappa:      elongation (kappa=1 is circular)
    delta:      triangularity (delta=0 is circular)
    dR0dr:      displacement of flux-surfaces, related to Shafranov shift
    s_q:        shear of safety factor
    s_kappa:    variation of elongation with radial coordinate
    s_delta:    variation of triangularity with radial coordinate
    alpha:      dimensionless pressure gradient

    It automatically generates other relevant dimensionless variables derived
    from these quantities.
    """
    def __init__(self, epsilon, q, kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha):

        # Let's set all the properties
        self.epsilon= epsilon
        self.q      = q
        self.kappa  = kappa
        self.delta  = delta
        self.dR0dr  = dR0dr
        self.s_q    = s_q
        self.s_kappa= s_kappa
        self.s_delta= s_delta
        self.alpha  = alpha

        # We now calculate the various dimensionless properties derived from
        # this set of parameters
        x                   = np.arcsin(delta)
        gamma,  gamma_err   = quad(lambda theta: gamma_integrand(theta,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)/(2*np.pi),  -np.pi, np.pi)
        C1,     C1_err      = quad(lambda theta: C1_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)/(2*np.pi),  -np.pi, np.pi)
        C2,     C2_err      = quad(lambda theta: C2_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)/(2*np.pi),  -np.pi, np.pi)
        C3,     C3_err      = quad(lambda theta: C3_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)/(2*np.pi),  -np.pi, np.pi)
        C4,     C4_err      = quad(lambda theta: C4_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)/(2*np.pi),  -np.pi, np.pi)
        xi,     xi_err      = quad(lambda theta: xi_integrand(theta,gamma,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha),  -np.pi, np.pi)
        fpR0                = fpR0_func(gamma,C1,C2,C3,C4,epsilon,q,kappa,x,dR0dr,s_q,s_kappa,s_delta,alpha)

        # and assign to self
        self.x              = x
        self.gamma          = gamma
        self.fpR0           = fpR0
        self.sigma          = q/gamma * fpR0 - alpha/(2*epsilon)
        self.xi             = xi


#############################################################################
#
#
#
# Here we define functions relevant for miller geometry
#
#
#
#############################################################################

# poloidal field normalized by B0
def bps_0(theta,MC):
    return MC.gamma * MC.epsilon / MC.q * np.sqrt(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)/(MC.kappa*(1 + MC.epsilon*np.cos(theta + MC.x*np.sin(theta)))*(np.cos(theta)*(MC.dR0dr + np.cos(theta + MC.x*np.sin(theta))) + (1 + MC.s_kappa + (-MC.s_delta + MC.x + MC.s_kappa*MC.x)*np.cos(theta))*np.sin(theta)*np.sin(theta + MC.x*np.sin(theta))))

# poloidal field normalized by Bp,0
def bps(theta,MC):
    return np.sqrt(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)/(MC.kappa*(1 + MC.epsilon*np.cos(theta + MC.x*np.sin(theta)))*(np.cos(theta)*(MC.dR0dr + np.cos(theta + MC.x*np.sin(theta))) + (1 + MC.s_kappa + (-MC.s_delta + MC.x + MC.s_kappa*MC.x)*np.cos(theta))*np.sin(theta)*np.sin(theta + MC.x*np.sin(theta))))

# toroidal field normalized by B0
def bts(theta,MC):
    return 1/(1 + MC.epsilon*np.cos(theta + MC.x*np.sin(theta)))

# radius of curvature normalized by minor radius
def R_c(theta,MC):
    return (MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)**1.5/(MC.kappa*np.cos(theta)*(1 + MC.x*np.cos(theta))**2*np.cos(theta + MC.x*np.sin(theta)) + MC.kappa*np.sin(theta)*np.sin(theta + MC.x*np.sin(theta)))

# major radial coordinate, normalized by major radius
def R_s(theta,MC):
    return 1 + MC.epsilon*np.cos(theta + MC.x*np.sin(theta))

# major radial coordinate, normalized by major radius
def Z_s(theta,MC):
    return MC.epsilon*MC.kappa*np.sin(theta)

# sin of u
def sinu(theta,MC):
    return ((MC.kappa*np.cos(theta))/np.sqrt(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2))

# poloidal field radial derivative r*db_p/drho
def rdbpdrho(theta,MC):
    return 1/R_c(theta,MC) - MC.alpha/(2*MC.epsilon*bps(theta,MC))*(1/R_s(theta,MC) - R_s(theta,MC)) - MC.sigma/(R_s(theta,MC)*bps(theta,MC))

# toroidal field radial derivative r*db_t/drho
def rdbtdrho(theta,MC):
    return MC.epsilon * ( (MC.sigma + MC.alpha/(2* MC.epsilon))* MC.gamma**2 * MC.epsilon / MC.q**2 * bps(theta,MC) * R_s(theta,MC) - sinu(theta,MC)/R_s(theta,MC) )

# total field strength
def bs(theta,MC):
    return np.sqrt(bps_0(theta,MC)**2.0 + bts(theta,MC)**2.0)

# total field radial derivative r*db/drho
def rdbdrho(theta,MC):
    return (bts(theta,MC)**2.0 * rdbtdrho(theta,MC) + bps_0(theta,MC)**2.0 * rdbpdrho(theta,MC))/(bs(theta,MC)**2.0)

# derivative of poloidal arclength w.r.t theta, normalized by r
def ltheta(theta,MC):
    return np.sqrt(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)

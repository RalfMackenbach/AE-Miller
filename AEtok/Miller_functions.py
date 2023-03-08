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




class MC:
    """
    Miller class (MC) that stores all free Miller parameters in a pythonic way.
    The Miller geometry has the following parameters
    Args:
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
        self.x      = np.arcsin(delta)

        # We now calculate the various dimensionless properties derived from
        # this set of parameters
        gamma,  gamma_err   = quad(lambda theta: 1.0/(2.0*np.pi) * ltheta(theta,self)/(R_s(theta,self)**2 * bps(theta,self)),                       -np.pi, np.pi)
        C1,     C1_err      = quad(lambda theta: 1.0/(2.0*np.pi) * ltheta(theta,self) /(R_c(theta,self)* R_s(theta,self)**3 * bps(theta,self)**2),  -np.pi, np.pi)
        C2,     C2_err      = quad(lambda theta: 1.0/(2.0*np.pi) * ltheta(theta,self) * sinu(theta,self) /(R_s(theta,self)**4 * bps(theta,self)**2),-np.pi, np.pi)
        C3,     C3_err      = quad(lambda theta: 1.0/(2.0*np.pi) * ltheta(theta,self) /(R_s(theta,self)**2 * bps(theta,self)**3),                   -np.pi, np.pi)
        C4,     C4_err      = quad(lambda theta: 1.0/(2.0*np.pi) * ltheta(theta,self) /(R_s(theta,self)**4 * bps(theta,self)**3),                   -np.pi, np.pi)

        # and assign to self
        self.gamma          = gamma
        self.C1             = C1
        self.C2             = C2
        self.C3             = C3
        self.C4             = C4
        fpR0                = fpR0_func(self)
        self.fpR0           = fpR0
        self.sigma          = q/gamma * fpR0 - alpha/(2*epsilon)
        # calculate xi
        xi,     xi_err      = quad(lambda theta: ltheta(theta,self) * bs(theta,self) / bps(theta,self),  -np.pi, np.pi)
        xi_2,   xi_2_err    = quad(lambda theta: ltheta(theta,self) / bps(theta,self),  -np.pi, np.pi)
        xi_3,   xi_3_err    = quad(lambda theta: ltheta(theta,self) * bs(theta,self)**2 /(bs(theta,self)),  -np.pi, np.pi)
        self.xi             = xi
        self.xi_2           = xi_2
        self.xi_3           = xi_3


#############################################################################
#
#
#
# Here we define functions relevant for miller geometry
#
#
#
#############################################################################

# radius of curvature normalized by minor radius
def R_c(theta,MC):
    return -(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)**1.5/(MC.kappa*(np.cos(MC.x*np.sin(theta)) + MC.x*np.cos(theta)**2*(2 + MC.x*np.cos(theta))*np.cos(theta + MC.x*np.sin(theta))))

# major radial coordinate, normalized by major radius
def R_s(theta,MC):
    return 1 + MC.epsilon*np.cos(theta + MC.x*np.sin(theta))

# derivative of poloidal arclength w.r.t theta, normalized by r
def ltheta(theta,MC):
    return np.sqrt(MC.kappa**2*np.cos(theta)**2 + (1 + MC.x*np.cos(theta))**2*np.sin(theta + MC.x*np.sin(theta))**2)

# sin of u
def sinu(theta,MC):
    return (MC.kappa*np.cos(theta))/ltheta(theta,MC)

# poloidal field normalized by Bp,0
def bps(theta,MC):
    return ltheta(theta,MC) / (MC.kappa * R_s(theta,MC) * (np.cos(theta)*(MC.dR0dr + np.cos(theta + MC.x*np.sin(theta))) + (1 + MC.s_kappa + (-MC.s_delta + MC.x + MC.s_kappa*MC.x)*np.cos(theta))*np.sin(theta)*np.sin(theta + MC.x*np.sin(theta))))

# toroidal field normalized
def bts(theta,MC):
    return 1/(R_s(theta,MC)) # by B0
    #return 1/(MC.gamma * R_s(theta,MC)) # by Bunit

# total field strength normalized
def bs(theta,MC):
    return np.sqrt((MC.gamma * MC.epsilon / MC.q * bps(theta,MC))**2 + bts(theta,MC)**2.0) # by B0
    #return np.sqrt((MC.epsilon / MC.q * bps(theta,MC))**2 + bts(theta,MC)**2.0) # by Bunit

# function for R0*f'
def fpR0_func(MC):
    return (MC.gamma*MC.q*(MC.alpha*MC.C3 + 2*MC.epsilon*(2*MC.C1 + 2*MC.C2*MC.epsilon + MC.gamma*MC.s_q)))/(2.*(MC.epsilon**3*MC.gamma**3 + MC.C4*MC.epsilon*MC.q**2))

# radial derivative of poloidal magnetic field
def rdbpdrho(theta,MC):
    return 1/R_c(theta,MC) - MC.alpha/(2 * MC.epsilon * bps(theta,MC)) * (1/R_s(theta,MC) - R_s(theta,MC)) - MC.sigma/(R_s(theta,MC) * bps(theta,MC))

# radial derivative of toroidal magnetic field
def rdbtdrho(theta,MC):
    return MC.epsilon * (  MC.gamma**2 *MC.epsilon / MC.q**2.0 * ( MC.sigma + MC.alpha/(2*MC.epsilon) ) * R_s(theta,MC) * bps(theta,MC) - sinu(theta,MC)/R_s(theta,MC) )

def rdbdrho(theta,MC):
    return ( bts(theta,MC)**2 * rdbtdrho(theta,MC) + (MC.gamma * MC.epsilon / MC.q * bps(theta,MC))**2 * rdbpdrho(theta,MC) ) / bs(theta,MC)**2 # by B0
    #return ( bts(theta,MC)**2 * rdbtdrho(theta,MC) + (MC.epsilon / MC.q * bps(theta,MC))**2 * rdbpdrho(theta,MC) ) / bs(theta,MC)**2 # by Bunit

def Z_s(theta,MC):
    return MC.epsilon*MC.kappa*np.sin(theta)
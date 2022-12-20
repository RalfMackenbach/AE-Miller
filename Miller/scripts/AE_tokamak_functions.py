import numpy                as np
from   scipy.signal         import  find_peaks
from    scipy               import  special


#############################################################################
#
#
#
# Here we define functions relevant for the AE calculation
#
#
#
#############################################################################


# return zero crossings
def zerocross1d(x, y, getIndices=False):
  """
    !!! This beautiful little gem is adapted from PyAstronomy !!!
    Find the zero crossing events in a discrete data set.
    Linear interpolation is used to determine the actual
    locations of the zero crossing between two data points
    showing a change in sign. Data point which are zero
    are counted in as zero crossings. Also returns the
    indices left of the zero crossing (even if it hits
    zero exactly).
    *x*
        Array containing the abscissa
    *y*
        Array containing the ordinate
    *getIndices*
        Boolean, if True, also the indicies of the points preceding
        the zero crossing event will be returned. Default is
        True.

    Returns
    ------------------------------------------------------------------
    xvals : array
        The locations of the zero crossing events determined
        by linear interpolation on the data.
    indices : array, optional
        The indices of the points preceding the zero crossing
        events.
  """

  # Check sorting of x-values
  if np.any((x[1:] - x[0:-1]) <= 0.0):
    raise(PE.PyAValError("The x-values must be sorted in ascending order!", \
                         where="zerocross1d", \
                         solution="Sort the data prior to calling zerocross1d."))

  # Indices of points *before* zero-crossing.
  indi = np.where(y[1:]*y[0:-1] < 0.0)[0]

  # Find the zero crossing by linear interpolation
  dx = x[indi+1] - x[indi]
  dy = y[indi+1] - y[indi]
  zc = -y[indi] * (dx/dy) + x[indi]

  # What about the points, which are actually zero?
  # We don't count the first element.
  # Because we are going over a smaller array, the
  # indices are shifted by -1, ensuring we end up
  # left of the crossing in y[zi].
  zi = (np.where(y[1:len(y)] == 0.0)[0])

  # Concatenate indices
  zzindi = np.concatenate((indi, zi))
  # Concatenate zc and locations corresponding to zi
  zz = np.concatenate((zc, x[zi+1]))

  # Sort by x-value
  sind = np.argsort(zz)
  zz, zzindi = zz[sind], zzindi[sind]

  if not getIndices:
    return zz
  else:
    return zz, zzindi


# returns the bounce wells for a given lambda
def bounce_wells(theta_arr, b_arr, lam):
    """
        Returns a list of bounce points and indices
        t       = [[-0.1,0.05],[0.31,0.36]]
        t_idx   = [[20,54],[80,94]]
        where each row corresponds to a bounce well
        and the left and right column corresponds to
        the left and right bounce points respectively.
        The bounce wells are ordered so that each
        *theta_arr*
            Array of theta coordinate
        *b_arr*
            Array of magnetic field strength values
        *lam*
            pitch-angle
    """
    # Rewrite array in a form suitable for zerocross1d
    zeros_arr   = 1 - lam * b_arr

    # Find zero crossings
    t_cross, t_cross_idx = zerocross1d(theta_arr, zeros_arr, True)
    if np.mod(len(t_cross),2)==1:
        raise(PE.PyAValError("Uneven number of bounce wells", \
                           where="bounce_wells", \
                           solution="Choose another lamba, or increase resolution"))

    t_wells     = []
    idx_wells   = []
    # Check if zero crossing defines the end or start of a well.
    # If it is a start, it and the next coordinate define a well.
    for (idx, t_idx) in enumerate(t_cross_idx):
        if zeros_arr[t_idx+1]>zeros_arr[t_idx]:
            t_wells.append([t_cross[idx],t_cross[(idx+1) % len(t_cross)]])
            idx_wells.append([t_cross_idx[idx],t_cross_idx[(idx+1) % len(t_cross)]])

    return t_wells, idx_wells


# integral for f/np.sqrt(1-lam*b) for linear approximation of f and b between two nodes
def int_approx(thi,thj,fi,fj,bi,bj,lam):
     """
       Gives the exact integral of f/np.sqrt(1 - lam * b), where f and b
       are replaced by their linear approximation in theta. Integral
       taken from thetai to thetaj.
     """
     blami = 1.0 - lam * bi
     blamj = 1.0 - lam * bj
     return (-2*(np.sqrt(blamj)*(2*fi + fj) + np.sqrt(blami)*(fi + 2*fj))*(thi - thj))/(3.*(np.sqrt(blami) + np.sqrt(blamj))**2)


# integral for f/np.sqrt(1-lam*b) for linear approximation of f and b between two nodes, where denominator vanishes on the left
def int_approx_left(thi,thj,fi,fj,bi,bj,lam):
     """
       Gives the exact integral of f/np.sqrt(1 - lam * b), where b vanishes
       at thi
     """
     blami = 1.0 - lam * bi
     blamj = 1.0 - lam * bj
     return (2*(2*fi + fj)*(-thi + thj))/(3.*np.sqrt(blamj))


# integral for f/np.sqrt(1-lam*b) for linear approximation of f and b between two nodes, where denominator vanishes on the right
def int_approx_right(thi,thj,fi,fj,bi,bj,lam):
     """
       Gives the exact integral of f/np.sqrt(1 - lam * b), where b vanishes
       at thj
     """
     blami = 1.0 - lam * bi
     blamj = 1.0 - lam * bj
     return (-2*np.sqrt(1/blami)*(fi + 2*fj)*(thi - thj))/3.


# given lamdba and and theta, b, and f arrays, calculate full integral
def int_full(theta_arr,b_arr,f_arr,lam):
    """
      Computes the full integral over all bounce wells for a
      given lamdba.
    """
    # find bounce points
    t_wells, idx_wells = bounce_wells(theta_arr, b_arr, lam)
    int_wells = []
    for idx in range(0,len(t_wells)):
        wells_loc = t_wells[idx]
        wells_idx = idx_wells[idx]
        l_idx = wells_idx[0]
        r_idx = wells_idx[1]
        l_val = wells_loc[0]
        r_val = wells_loc[1]
        # check if integral goes over periodicity boundary,
        # if so split inner integral into two parts
        if l_val>r_val:
            # create arrays for inner integral and do integral
            thetai      = theta_arr[(l_idx+1):(-1)]
            thetaj      = theta_arr[(l_idx+2)::]
            bi          = b_arr[(l_idx+1):(-1)]
            bj          = b_arr[(l_idx+2)::]
            fi          = f_arr[(l_idx+1):(-1)]
            fj          = f_arr[(l_idx+2)::]
            inner_int_1 = np.sum(int_approx(thetai,thetaj,fi,fj,bi,bj,lam))
            # create arrays for inner integral and do integral
            thetai      = theta_arr[0:(r_idx)]
            thetaj      = theta_arr[1:(r_idx+1)]
            bi          = b_arr[0:(r_idx)]
            bj          = b_arr[1:(r_idx+1)]
            fi          = f_arr[0:(r_idx)]
            fj          = f_arr[1:(r_idx+1)]
            inner_int_2 = np.sum(int_approx(thetai,thetaj,fi,fj,bi,bj,lam))
            inner_int   = inner_int_1 + inner_int_2
        # otherwise, straightforward implementation
        if l_val<r_val:
            # create arrays for inner integral and do integral
            thetai    = theta_arr[(l_idx+1):(r_idx)]
            thetaj    = theta_arr[(l_idx+2):(r_idx+1)]
            bi        = b_arr[(l_idx+1):(r_idx)]
            bj        = b_arr[(l_idx+2):(r_idx+1)]
            fi        = f_arr[(l_idx+1):(r_idx)]
            fj        = f_arr[(l_idx+2):(r_idx+1)]
            inner_int = np.sum(int_approx(thetai,thetaj,fi,fj,bi,bj,lam))
        # now the integrals at the edge
        # linear interpolation of f_i
        fleft       = f_arr[l_idx] + (l_val - theta_arr[l_idx])/(theta_arr[l_idx+1] - theta_arr[l_idx]) * (f_arr[l_idx+1] - f_arr[l_idx])
        left_int    = int_approx_left(l_val,theta_arr[l_idx+1],fleft,f_arr[l_idx+1],0.0,b_arr[l_idx+1],lam)
        # linear interpolation of f_j
        fright      = f_arr[r_idx] + (r_val - theta_arr[r_idx])/(theta_arr[r_idx+1] - theta_arr[r_idx]) * (f_arr[r_idx+1] - f_arr[r_idx])
        right_int   = int_approx_right(theta_arr[r_idx],r_val,f_arr[r_idx],fright,b_arr[r_idx],0.0,lam)
        # construct total in and append
        int_wells.append(left_int+inner_int+right_int)

    return int_wells


# given lamdba and and theta, b, and f arrays, calculate bounce frequencies
def omega_lam(theta_arr,b_arr,f_arr_numer,f_arr_denom,lam):
    numer = int_full(theta_arr,b_arr,f_arr_numer,lam)
    denom = int_full(theta_arr,b_arr,f_arr_denom,lam)

    return numer,denom



# integral over z
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

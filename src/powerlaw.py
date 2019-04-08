import numpy as np
import mpmath as mpm
from scipy.special import zeta
import time
## power law
#disc

def zeta_minmax(gamma,kmin,kmax):
    '''kmax == None means kmax == infty
    '''
    if gamma <= 1.0:
        if isinstance(kmax,(list,np.ndarray)):
            kmax_max = np.max(kmax)
            x = (1.0*np.arange(kmin,kmax_max+1,1))**(-gamma)
            Fx = np.cumsum(x)
            C = []
            # C0 = mpm.zeta(gamma,kmin)

            for k_ in kmax:
                # C += [ float(C0-mpm.zeta(gamma,float(k_)))]
                C += [ Fx[ int(k_- kmin) ] ]
            C = np.array(C)
        elif kmax == None:
            print('ERROR: Series does not converge!!!')
            C = 0
        else:
            mpm.dps=25
            # C = (float(mpm.sumem(lambda k: k**(-gamma),[kmin,kmax])))
            C = float(mpm.zeta(gamma,float(kmin)) - mpm.zeta(gamma,float(kmax)))
    else:
        if isinstance(kmax,(list,np.ndarray)):
            C = zeta(gamma,kmin)-zeta(gamma,kmax)
        elif kmax == None:
            C = zeta(gamma,kmin)
        else:
            C = zeta(gamma,kmin)-zeta(gamma,kmax)
    return C

def pdf_power_disc(x,xmin,xmax,gamma):
    '''returns discrete power law with cutoff xmin,xmax...including xmax as last element
       therefore xmax+1 in argument. 
    '''
    if xmax == None:
        C = (x**(-gamma))/zeta_minmax(gamma,xmin,xmax)
    else:
        C = (x**(-gamma))/zeta_minmax(gamma,xmin,xmax+1)
    return C

def cdf_power_disc(x,xmin,xmax,gamma):
    '''returns discrete power law with cutoff xmin,xmax...including xmax as last element
       therefore xmax+1 in argument. 
    '''
    if xmax == None:
        C = zeta_minmax(gamma,xmin,x+1)/zeta_minmax(gamma,xmin,xmax)
    else:
        C = zeta_minmax(gamma,xmin,x+1)/zeta_minmax(gamma,xmin,xmax+1)
    return C

def logL_power_disc(x,nx,xmin,xmax,gamma):
    ## neg normalized log-Likelihood
    N = sum(nx)
    C= zeta_minmax(gamma,xmin,xmax)
    logL = np.log(C) + gamma*np.sum(nx/float(N)*np.log(x))
    return logL

## Random number generator for discrete powerlaw
def draw_power_binary(N,xmin,xmax,gamma):
    '''
    Draw random variables from a discrete power law with exponent gamma
    p(x) ~ x^{-gamma} for x = xmin,...,xmax

    for xmax == infty, put xmax=None.

    Returns an array of size N
    '''
    # np.random.seed()

    ## enforce upper limit for random variables (64-but integer)
    if xmax == None:
        xmax_nan = np.nan
    else:
        xmax_nan = xmax
    cdf_max = cdf_power_disc( np.nanmin([xmax_nan,2**63-1]),xmin,xmax,gamma)

    x_uni = np.sort(np.random.rand(N) * cdf_max)
    x_uni_tmp = 1.0*x_uni
    x1 = xmin
    x2 = xmin
    ind_done = ([])
    x1_array = ([])
    x2_array = ([])

    while len(x_uni_tmp)>0:
        cdf_x2 = cdf_power_disc(x2,xmin,xmax,gamma)
        x_uni_tmp = np.delete(x_uni_tmp,ind_done)
        ind_done = np.where(cdf_x2>x_uni_tmp)[0]
        x1_array = np.append(x1_array,x1*np.ones(len(ind_done)))
        x2_array = np.append(x2_array,x2*np.ones(len(ind_done)))
        x1 = x2
        x2 = 2.0*x1
        
    x_uni_tmp = 1.0*x_uni
    ind_done = ([])
    x_random = ([])
    ind_done = np.where(x1_array==x2_array)[0]
    x_random = np.append(x_random,x2_array[ind_done])
    x1_array = np.delete(x1_array,ind_done)
    x2_array = np.delete(x2_array,ind_done)
    x_uni_tmp = np.delete(x_uni_tmp,ind_done)   

    while len(x_uni_tmp)>0:


        ind_done = np.where(x1_array==(x2_array-1))[0]
        x_random = np.append(x_random,x2_array[ind_done])
        x1_array = np.delete(x1_array,ind_done)
        x2_array = np.delete(x2_array,ind_done)
        x_uni_tmp = np.delete(x_uni_tmp,ind_done)
        if len(x1_array)>0:
        
            x_delta = x2_array - x1_array
            x_mid = (x1_array + (0.5*x_delta).astype(int))
            # x_mid = (0.5*(x2_array+x1_array)).astype(int)


            cdf_xmid = cdf_power_disc(x_mid,xmin,xmax,gamma)
            x1_array =  ((cdf_xmid > x_uni_tmp)*x1_array + (cdf_xmid <= x_uni_tmp)*(x_mid)).astype(np.int)
            x2_array = ((cdf_xmid > x_uni_tmp)*x_mid + (cdf_xmid <= x_uni_tmp)*x2_array).astype(int)


        # print(x_mid,xmin,xmax,gamma)

    np.random.shuffle(x_random)
    return np.array(x_random).astype('int')


### CONTINUOUS
def pdf_power_cont(x,xmin,xmax,gamma):
    '''returns cont power law 
    '''
    if xmax == None:
        xmax=np.inf
    C = (1.0-gamma)/(xmax**(1.-gamma) - xmin**(1.-gamma))
    pdf_x = C*x**(-gamma)
    return pdf_x

def cdf_power_cont(x,xmin,xmax,gamma):
    '''returns discrete power law with cutoff xmin,xmax...including xmax as last element
       therefore xmax+1 in argument. 
    '''
    if xmax == None:
        xmax=np.inf
    cdf_x = ( x**(1.-gamma) - xmin**(1.0-gamma))/( xmax**(1.-gamma) - xmin**(1.0-gamma))
    return cdf_x

# def logL_power_cont(sample_x,xmin,xmax,gamma):
#     ## neg normalized log-Likelihood
#     if xmax==None:
#         xmax=np.inf
#     logL = 0.0
#     logL += np.log(gamma-1.) 
#     logL += -1.*np.log( xmin**(1.0-gamma)-xmax**(1.-gamma)) 
#     logL += -1.*gamma*np.mean(np.log(sample_x))
#     return -logL

def logL_power_cont(x,nx,xmin,xmax,gamma):
    ## neg normalized log-Likelihood
    N = np.sum(nx)
    if xmax==None:
        xmax=np.inf
    logL = 0.0
    logL += np.log(gamma-1.) 
    logL += -1.*np.log( xmin**(1.0-gamma)-xmax**(1.-gamma)) 
    logL += -1.*gamma*np.sum( nx*np.log(x) )/N
    return -logL

## Random number generator for cont powerlaw via inverse
def draw_power_cont(N,xmin,xmax,gamma):
    x_random = np.random.random(size=N)

    if xmax==None:
        xmax=np.inf
    return ( x_random*( xmax**(1.-gamma)-xmin**(1.-gamma) )+xmin**(1.-gamma) )**(1./(1.-gamma))
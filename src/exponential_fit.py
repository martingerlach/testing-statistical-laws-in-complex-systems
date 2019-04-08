import numpy as np

from collections import Counter
from scipy.special import gamma
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.special import logsumexp

from stats import pval_from_score


from stats import KS_stat, KL_stat, xdata_to_xnx

def pdf_exponential_cont(x,a,xmin=0):
    '''
    '''
    return a*np.exp(-a*(x-xmin))



def cdf_exponential_cont(x,a,xmin=0):
    '''
    '''
    return 1.-np.exp(-a*(x-xmin))



def draw_exponential_cont(a,N,xmin=0,cont=True):
    '''
    Use inverse of cumulative to draw from continuous stretched exponential.
    
    When setting cont=True we map the continuous random numbers to discrete ones. 
    '''
    u_random = np.random.random(N)
    x_random = ( -1./a*np.log(u_random) + xmin   )

    if cont == False:
        x_random = x_random.astype('int')
    return x_random


def KS_exponential(x_emp,px_emp,a,cont=True,xmin=0):

    Fx_emp = np.cumsum(px_emp)

    if cont==True:
        ## continuous exp
        Fx_true = cdf_exponential_cont(x_emp,a,xmin=xmin)
        KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true[1:] ))
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
    else:
        ## discrete exp
        # D_plus
        if len(Fx_emp)>1:
            Fx_true = cdf_e_disc(x_emp[1:]-1,a,xmin=xmin)
            KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true ))
        else:
            KS_plus = 0.0
        # D_minus
        Fx_true = cdf_e_disc(x_emp,a,xmin=xmin)
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
    KS = np.max([KS_plus,KS_minus])
    return KS

def logL_exponential_cont(x,nx,a,xmin=0):
    N = np.sum(nx)
    logL = np.log(a) -a*np.sum( nx/N*(x-xmin) )
    return -logL

def fit_exponential_cont(x,nx,xmin=0):
    N = np.sum(nx)
    result = []
    a_fit = 1./np.sum( nx/N*(x-xmin) )
    par = [a_fit]
    result+=[par]
    return result
    
def fit_exponential_cont_wrapper(x,nx,xmin=0.):
    V = len(x)
    N = sum(nx)
    
    px = nx/float(N)

    ## fit
    result_fit = fit_exponential_cont(x,nx,xmin=xmin)
    a_fit = result_fit[0][0]
    KS = KS_exponential(x,px,a_fit,xmin=xmin)
    L = logL_exponential_cont(x,nx,a_fit,xmin=xmin)
    result = {}
    result['a'] = a_fit
    result['L'] = L
    result['KS'] = KS
    return result

def fit_exponential_cont_sign(x,nx,nrep_synth=0,xmin=0,tau=1):
    '''
    decrease the size of the synthetic data from N --> N/tau
    '''

    ## fit real
    result = fit_exponential_cont_wrapper(x,nx,xmin=xmin)
    a_fit = result['a']
    ## fit sign
    a_synth = []
    L_synth = []
    KS_synth = []

    ## surrogate data
    N = sum(nx)
    n = int(N/tau)
    for i_nrep in range(nrep_synth):
        x_data_synth = draw_exponential_cont(a_fit,n,xmin=xmin)
        x_synth,nx_synth = xdata_to_xnx(x_data_synth)
        result_synth = fit_exponential_cont_wrapper(x_synth,nx_synth,xmin=xmin)

        a_synth+=[result_synth['a']]
        L_synth+=[result_synth['L']]
        KS_synth+=[result_synth['KS']]

    result['a_synth'] = a_synth
    result['L_synth'] = L_synth
    result['KS_synth'] = KS_synth

    pval = pval_from_score(KS_synth,result['KS'],kind='right')
    result['pval'] = pval

    return result
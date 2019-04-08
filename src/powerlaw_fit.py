import numpy as np
import mpmath as mpm
from scipy.special import zeta
from scipy.optimize import fsolve
from collections import Counter
from scipy.stats import zipf
from stats import pval_from_score

import time

from powerlaw import *
from stats import KS_stat, KL_stat,xdata_to_xnx


### DISCRETE

def fit_power_disc_sign(x,nx,xmin=1,xmax=None,a0=1.5,nrep_synth=0,tau=1):
    '''
    decrease the size of the synthetic data from N --> N/tau
    '''

    ## fit real
    result = fit_power_disc_wrapper(x,nx,xmin,xmax,a0=a0)
    alpha_fit = result['alpha']
    ## fit sign
    alpha_synth = []
    L_synth = []
    KS_synth = []
    KL_synth = []

    ## surrogate data
    N = sum(nx)
    for i_nrep in range(nrep_synth):
        x_data_synth = draw_power_binary(int(N/tau),xmin,xmax,alpha_fit)
        x_synth,nx_synth = xdata_to_xnx(x_data_synth)
        result_synth = fit_power_disc_wrapper(x_synth,nx_synth,xmin,xmax,a0=a0)
        alpha_synth+=[result_synth['alpha']]
        L_synth+=[result_synth['L']]
        KS_synth+=[result_synth['KS']]

    result['alpha_synth'] = alpha_synth
    result['L_synth'] = L_synth
    result['KS_synth'] = KS_synth

    pval = pval_from_score(KS_synth,result['KS'],kind='right')
    result['pval'] = pval

    return result

def fit_power_disc_wrapper(x,nx,xmin,xmax,a0=1.5):
    V = len(x)
    N = sum(nx)
    
    px = nx/float(N)
    ###
    ## REAL DATA
    ###

    ## fit
    result_fit = fit_power_disc(x,nx,xmin,xmax,a0=a0)
    alpha_fit = result_fit[0][0]
    ## KS - test
    KS = KS_pow(x,px,alpha_fit,xmin,xmax,cont=False)
    L = logL_power_disc(x,nx,xmin,xmax,alpha_fit)
    result = {}
    result['alpha'] = alpha_fit
    result['L'] = L
    result['KS'] = KS
    return result


def zeta_minmax_n(a,xmin,xmax,n):
    z = 0.0
    if xmax == np.inf or xmax==None:
        z = mpm.zeta(a,float(xmin),n)
    else:
        z = mpm.zeta(a,float(xmin),n) - mpm.zeta(a,float(xmax+1),n)
    return float(z)
    
def fit_power_disc(x,nx,xmin,xmax,a0=1.5):
    a0=float(a0)
    N = sum(nx)
    D = np.sum( nx/float(N)*np.log(x) )
    alpha_fit = -1
    ## if alpha is smaller 0, try a different initial condition
    while alpha_fit<0:
        a0 = 1.0+np.random.random()+0.001
        result = fsolve(fit_power_disc_func,a0,args = (xmin,xmax,D),full_output=True  )
        alpha_fit = result[0][0]
    return result
    
def fit_power_disc_func(a,xmin,xmax,D):
    # print(a,xmin,xmax)
    if a[0]<0:
        a[0]=0
    return D + zeta_minmax_n(a[0],xmin,xmax,1)/zeta_minmax_n(a[0],xmin,xmax,0)


### CONTINUOUS
def fit_power_cont_sign(x,nx,xmin=1,xmax=None,a0=1.5,nrep_synth=0):

    if xmax==None:
        xmax=np.inf
    ## fit real
    result = fit_power_cont_wrapper(x,nx,xmin,xmax,a0=a0)
    alpha_fit = result['alpha']
    ## fit sign
    alpha_synth = []
    L_synth = []
    KS_synth = []
    N=sum(nx)
    ## surrogate data
    for i_nrep in range(nrep_synth):
        x_data_synth = draw_power_cont(int(N),xmin,xmax,alpha_fit)
        x_synth,nx_synth = xdata_to_xnx(x_data_synth)
        result_synth = fit_power_cont_wrapper(x_synth,nx_synth,xmin,xmax,a0=a0)
        alpha_synth+=[result_synth['alpha']]
        L_synth+=[result_synth['L']]
        KS_synth+=[result_synth['KS']]


    result['alpha_synth'] = alpha_synth
    result['L_synth'] = L_synth
    result['KS_synth'] = KS_synth

    pval = pval_from_score(KS_synth,result['KS'],kind='right')
    result['pval'] = pval

    return result

def fit_power_cont_wrapper(x,nx,xmin,xmax,a0=1.5):

    ## fit
    result_fit = fit_power_cont(x,nx,xmin,xmax,a0=a0)
    alpha_fit = result_fit[0][0]
    
    ## KS - test -- perhaps improve this one?
    px = nx/float(np.sum(nx))
    KS = KS_pow(x,px,alpha_fit,xmin,xmax,cont=True)

    L = logL_power_cont(x,nx,xmin,xmax,alpha_fit)

    result = {}
    result['alpha'] = alpha_fit
    result['L'] = L
    result['KS'] = KS
    # result['cdf'] = (arr_x_emp,arr_cdfx_emp,arr_cdfx_mod)
    return result

def fit_power_cont(x,nx,xmin,xmax,a0=1.5):
    if xmax==None:
        xmax=np.inf
    N = np.sum(nx)
    a0=float(a0)
    # D = np.mean(np.log(sample_x))
    D = np.sum(nx*np.log(x))/N
    result = fsolve(fit_power_cont_func,a0,args = (xmin,xmax,D),full_output=True)
    return result
    
def fit_power_cont_func(a,xmin,xmax,D):
    gamma = a[0]
    if xmax==np.inf:
        return D + 1./(1.-gamma) - np.log(xmin)
    else:
        return D + 1./(1.-gamma) - (np.log(xmax)*xmax**(1.-gamma)-np.log(xmin)*xmin**(1.-gamma))/(xmax**(1.-gamma)-xmin**(1.-gamma))


def KS_pow(x_emp,px_emp,a,xmin,xmax,cont = True):

    Fx_emp = np.cumsum(px_emp)

    if cont==True:
        ## continuous powerlaw
        Fx_true = cdf_power_cont(x_emp,xmin,xmax,a)
        KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true[1:] ))
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
        
    else:
        ## discrete powerlaw
        # D_plus
        if len(Fx_emp)>1:
            Fx_true = cdf_power_disc(x_emp[1:]-1,xmin,xmax,a)
            KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true ))
        else:
            KS_plus = 0.0
        # D_minus
        Fx_true = cdf_power_disc(x_emp,xmin,xmax,a)
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
    KS = np.max([KS_plus,KS_minus])
    return KS
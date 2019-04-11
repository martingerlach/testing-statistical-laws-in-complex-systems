import numpy as np

from collections import Counter
from scipy.special import gamma
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.special import logsumexp

from stats import pval_from_score


from stats import KS_stat, KL_stat, xdata_to_xnx

def pdf_stretched_exponential_cont(x,a,b,xmin=0):
    '''
    The 2-parameter continuous stretched exponential pdf for x>0.
    a > 0, b>0. (b=1 is the exponential)
    '''
    return a*b*x**(b-1.)*np.exp(-a*(x**b-xmin**b)  )
    # return a*b*(x-xmin)**(b-1.)*np.exp(-a*(x-xmin)**(b))
    # return a*b*x**(b-1.)*np.exp(-a*x**(b))
    # return np.exp( np.log(a) + np.log(b) + (b-1)*np.log(x) - a*x**b ) 


def cdf_stretched_exponential_cont(x,a,b,xmin=0):
    '''
    The 2-parameter continuous stretched exponential cdf for x>0.
    cdf is \int_0^{t} dt' pdf(t')
    a > 0, b > 0.
    '''
    return 1.-np.exp(-a*(x**b-xmin**b))
    # return 1.-np.exp(-a*(x-xmin)**(b))
    # return 1.-np.exp(-a*x**(b))



def draw_stretched_exponential_cont(a,b,N,xmin=0,cont=True):
    '''
    Use inverse of cumulative to draw from continuous stretched exponential.
    
    When setting cont=True we map the continuous random numbers to discrete ones. 
    '''
    u_random = np.random.random(N)
    x_random = ( -1./a*np.log(u_random) + xmin**b   )**(1./b)

    if cont == False:
        x_random = x_random.astype('int')
    return x_random



## function for discrete
def Fx_se_disc(x,a,b,xmin=0):
    '''
    sum_{x'=x}^{infty} P(x') = exp{-a*( x^b - xmin^b )}
    --> F(x=xmin)= 1 and F(x-->infty) = 0

    '''
    return (x>=xmin)*np.exp(-a*(x**b-xmin**b))
def cdf_se_disc(x,a,b,xmin=0):
    '''
    sum_{x'=xmin}^{x} P(x')
    '''
    return (x>=xmin)*(1.-np.exp(-a*((x+1)**b-xmin**b)))

def log_Fx_se_disc(x,a,b,xmin=0):
    return (x>=xmin)*(-a*(x**b-xmin**b))

def Px_se_disc(x,a,b,xmin=0):
    return Fx_se_disc(x,a,b,xmin=xmin)-Fx_se_disc(x+1,a,b,xmin=xmin)

def log_Px_se_disc(x,a,b,xmin=0,eps_beta=10.0**(-6)):
    '''
    Using trick:
    log P = log ( F(x)-F(x+1) )
    log (b-a) = log[ a*( exp(log(b/a))-1 )]
              = log(F(x)) + log( exp[log(F(x)) - log(F(x+1)) ] - 1 )

    eps_beta is the minium value for beta. the distribution is only defined for beta > 0.
    '''
    if b <= 0.: ## enforce positive value for b to avoid invalid value of log
        b = eps_beta
    P = log_Fx_se_disc(x+1,a,b,xmin=xmin)
    P += np.log( np.exp( log_Fx_se_disc(x,a,b,xmin=xmin)-log_Fx_se_disc(x+1,a,b,xmin=xmin)   )  -1.0)
    return P

def logL_se_disc(x,nx,a,b,xmin=0):
    N = np.sum(nx)
    logL = 0.
#     logL += np.sum(nx/N*np.log( Px_se_disc(x,a,b,xmin=xmin)))
    logL = np.sum(nx/N*log_Px_se_disc(x,a,b,xmin=xmin))
    return -logL

def fit_se_disc(x,nx,a0=1.,b0=1.0,xmin=0):
    N = np.sum(nx)
    a0=float(a0)
    b0=float(b0)
    par0 = (a0,b0)
    result = fmin(fit_se_disc_func,(par0),args = (x,nx,xmin),full_output=True,disp=0)
    return result
    
def fit_se_disc_func(par,x,nx,xmin):
    a = par[0]
    b = par[1]
    nlogL =  logL_se_disc(x,nx,a,b,xmin=xmin)
    return nlogL


def fit_se_disc_wrapper(x,nx,a0=1.0,b0=1.0,xmin=0):
    V = len(x)
    N = sum(nx)
    
    px = nx/float(N)

    ## fit
    result_fit = fit_se_disc(x,nx,a0=a0,b0=b0,xmin=xmin)
    a_fit = result_fit[0][0]
    b_fit = result_fit[0][1]
    ## KS - test
    KS = KS_se(x,px,a_fit,b_fit,xmin=xmin,cont = False)
    L = logL_se_disc(x,nx,a_fit,b_fit,xmin=xmin)
    result = {}
    result['a'] = a_fit
    result['b'] = b_fit
    result['L'] = L
    result['KS'] = KS
    return result

def fit_se_disc_sign(x,nx,a0=1.0,b0=1.0,xmin=0,nrep_synth=0,tau=1):
    '''
    decrease the size of the synthetic data from N --> N/tau
    '''

    ## fit real
    result = fit_se_disc_wrapper(x,nx,a0=a0,b0=b0,xmin=xmin)
    a_fit = result['a']
    b_fit = result['b']
    ## fit sign
    a_synth = []
    b_synth = []
    L_synth = []
    KS_synth = []

    ## surrogate data
    N = sum(nx)
    n = int(N/tau)
    for i_nrep in range(nrep_synth):
        x_data_synth = draw_stretched_exponential_cont(a_fit,b_fit,n,xmin=xmin,cont=False)
        # x_data_synth = draw_se_binary(a_fit,b_fit,n,xmin=xmin)

        x_synth,nx_synth = xdata_to_xnx(x_data_synth)
        result_synth = fit_se_disc_wrapper(x_synth,nx_synth,a0=a0,b0=b0,xmin=xmin)

        a_synth+=[result_synth['a']]
        b_synth+=[result_synth['b']]
        L_synth+=[result_synth['L']]
        KS_synth+=[result_synth['KS']]

    result['a_synth'] = a_synth
    result['b_synth'] = b_synth
    result['L_synth'] = L_synth
    result['KS_synth'] = KS_synth

    pval = pval_from_score(KS_synth,result['KS'],kind='right'   )
    result['pval'] = pval

    return result



def KS_se(x_emp,px_emp,a,b,cont=True,xmin=0):

    Fx_emp = np.cumsum(px_emp)

    if cont==True:
        ## continuous se
        Fx_true = cdf_stretched_exponential_cont(x_emp,a,b,xmin=xmin)
        KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true[1:] ))
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
        
    else:
        ## discrete powerlaw
        # D_plus
        if len(Fx_emp)>1:
            Fx_true = cdf_se_disc(x_emp[1:]-1,a,b,xmin=xmin)
            KS_plus = np.max(np.abs( Fx_emp[:-1] - Fx_true ))
        else:
            KS_plus = 0.0
        # D_minus
        Fx_true = cdf_se_disc(x_emp,a,b,xmin=xmin)
        KS_minus = np.max(np.abs( Fx_emp - Fx_true ))
    KS = np.max([KS_plus,KS_minus])
    return KS

def logL_se_cont(x,nx,a,b,xmin=0):
    N = np.sum(nx)
    logL = 0.
    logL += np.log(a)
    logL += np.log(b)
    logL += (b-1.)*np.sum( nx/N*np.log(x) )
    logL += -a*np.sum( nx/N*(x**b-xmin**b) )
    # print(a,b,np.sum( nx/N*(x**b-xmin**b) ))
    return -logL

def fit_se_cont(x,nx,a0=1.,b0=1.0,xmin=0):
    N = np.sum(nx)
    a0=float(a0)
    b0=float(b0)
    par0 = (a0,b0)
    result = fmin(fit_se_cont_func,(par0),args = (x,nx,xmin),full_output=True,disp=0)
    return result
    
def fit_se_cont_func(par,x,nx,xmin):
    a = par[0]
    b = par[1]
    return logL_se_cont(x,nx,a,b,xmin=xmin)


## the 1 parameter version where we fix the average (==fixing a) and only fitting b
## only works for xmin=0
def fit_se_1paravg_cont(x,nx,b0=1.0):
    N = np.sum(nx)
    b0=float(b0)
    
    result = fmin(fit_se_1paravg_cont_func,b0,args = (x,nx),full_output=True,disp=0)
    return result
    
def fit_se_1paravg_cont_func(a,x,nx):
    b = a[0]
    N = np.sum(nx)
    nu = 1./np.sum(x*nx/N)
    a = (nu*gamma((1.+b)/(b)) )**b
    return logL_se_cont(x,nx,a,b)


def fit_se_1paravg_cont_wrapper(x,nx,b0=1.0):
    V = len(x)
    N = sum(nx)
    
    px = nx/float(N)

    ## fit
    result_fit = fit_se_1paravg_cont(x,nx,b0=b0)
    b_fit = result_fit[0][0]

    ## KS - test
    nu = 1./np.sum(x*nx/N)
    a_fit = (nu*gamma((1.+b_fit)/(b_fit)) )**b_fit

    KS = KS_se(x,px,a_fit,b_fit,cont = False)
    L = logL_se_cont(x,nx,a_fit,b_fit)
    result = {}
    result['a'] = a_fit
    result['b'] = b_fit
    result['L'] = L
    result['KS'] = KS
    return result

def fit_se_1paravg_cont_sign(x,nx,b0=1.0,nrep_synth=0,tau=1):
    '''
    decrease the size of the synthetic data from N --> N/tau
    '''

    ## fit real
    result = fit_se_1paravg_cont_wrapper(x,nx,b0=b0)
    a_fit = result['a']
    b_fit = result['b']
    ## fit sign
    a_synth = []
    b_synth = []
    L_synth = []
    KS_synth = []
    KL_synth = []

    ## surrogate data
    N = sum(nx)
    n = int(N/tau)
    for i_nrep in range(nrep_synth):
        x_data_synth = draw_stretched_exponential_cont(a_fit,b_fit,n,cont=False)
        x_synth,nx_synth = xdata_to_xnx(x_data_synth)
        result_synth = fit_se_1paravg_cont_wrapper(x_synth,nx_synth,b0=b0)

        a_synth+=[result_synth['a']]
        b_synth+=[result_synth['b']]
        L_synth+=[result_synth['L']]
        KS_synth+=[result_synth['KS']]

    result['a_synth'] = a_synth
    result['b_synth'] = b_synth
    result['L_synth'] = L_synth
    result['KS_synth'] = KS_synth

    pval = pval_from_score(KS_synth,result['KS'],kind='right')
    result['pval'] = pval

    return result



# ## Random number generator for discrete stretched exponeital --> not tested
def draw_se_binary(a,b,N,xmin=0):
    '''
    Draw random variables from a discrete stretched exponential

    Returns an array of size N
    '''
    # np.random.seed()

    ## enforce upper limit for random variables (64-but integer)

    cdf_max = cdf_se_disc(2**63-1,a,b,xmin=xmin)

    x_uni = np.sort(np.random.rand(N) * cdf_max)
    x_uni_tmp = 1.0*x_uni
    x1 = xmin
    x2 = xmin
    ind_done = ([])
    x1_array = ([])
    x2_array = ([])

    while len(x_uni_tmp)>0:
        cdf_x2 = cdf_se_disc(x2,a,b,xmin=xmin)
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


            cdf_xmid = cdf_se_disc(x_mid,a,b,xmin=xmin)
            x1_array =  ((cdf_xmid > x_uni_tmp)*x1_array + (cdf_xmid <= x_uni_tmp)*(x_mid)).astype(np.int)
            x2_array = ((cdf_xmid > x_uni_tmp)*x_mid + (cdf_xmid <= x_uni_tmp)*x2_array).astype(int)


        # print(x_mid,xmin,xmax,gamma)

    np.random.shuffle(x_random)
    return np.array(x_random).astype('int')


# def draw_stretched_exponential_cont(a,b,N,cont=True):
#     '''
#     The weibull in scipy stats gives the following pdf ('standard form'):
#     f(x,c) = c*x**(c-1)*np.exp(-x**c).
    
#     In our parametrization of the stretched exponential this corresponds to:
#         - a = 1
#         - b = c.
        
#     transformation from tau' (standard form) to tau via: tau' = tau*( a**(1/b) )
#     connects the two.
    
#     We can pass this factor via the scale-argument.
    
#     When setting cont=True we map the continuous random numbers to discrete ones. 
#     '''
#     scale = a**(-1./b)
#     x_random = weibull_min.rvs(b,scale=scale,size=N)
    
#     if cont == False:
#          x_random = map_cont_to_disc(x_random)
#     return x_random
# def map_cont_to_disc(x_data):
#     '''
#     take the integer part.
#     add +1 in order to avoid x_data = 0.
    
#     We could potentially change this part in the mapping
#     '''
#     return x_data.astype('int')+1 ## in order to avoid 0
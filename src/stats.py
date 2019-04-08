import numpy as np
from scipy.stats import percentileofscore
from collections import Counter

from statsmodels.tsa.stattools import acf

def pval_from_score(list_x,x,kind = 'right'):
    score = percentileofscore(list_x,x)
    if kind == 'right':
        pval = 1-score/100.
    if kind == 'left':
        pval = score/100.
    if kind == 'both':
        if score>50.:
            pval = 1.-score/100.
        else:
            pval = score/100.
    return pval

def p_dict_to_array(dict_xy,norm = True):
    '''
    Convert dictionary {var:val} into two arrays sorted by increasing var
    '''
    arr_x = np.array([])
    arr_y = np.array([])
    for x in sorted(dict_xy.keys()):
        y = dict_xy[x]
        arr_x=np.append(arr_x,x)
        arr_y=np.append(arr_y,y)
    if norm == True:
        arr_y=arr_y/np.sum(arr_y)
    return arr_x, arr_y

def KS_stat(x,px,px_fit):
    return np.max( np.abs(np.cumsum(px)- np.cumsum(px_fit)) )

def KL_stat(x,px,px_fit):
    return -np.nansum(  px*np.log( px_fit/px )  )

def sample_to_xpx(text,xmin=1,xmax=None,method='pk'):
    if method=='pk':
        x,nx = sample_to_kpk(text,xmin=xmin,xmax=xmax)
    elif method=='fr':
        x,nx = sample_to_rfr(text,xmin=xmin,xmax=xmax)
    else:
        x,nx = [],[]
    return x,nx

def sample_to_kpk(text,xmin=1,xmax=None):
    c=Counter(list(Counter(text).values()))
    if xmax==None:
        xmax=max(c)
    arr_k = np.arange(xmin,xmax+1)
    arr_nk = np.array([])
    for i_k,k in enumerate(arr_k):
        try:
            nk = c[k]
        except KeyError:
            nk = 0
        arr_nk = np.append(arr_nk,nk)
    return arr_k,arr_nk

def sample_to_rfr(text,xmin=1,xmax=None):
    c=Counter(text)
    if xmax==None:
        xmax=len(c)
    arr_r = np.arange(xmin,xmax+1)
    arr_nr = np.array([])
    for w,n in c.most_common():
        arr_nr = np.append(arr_nr,n)
    return arr_r,arr_nr

def xdata_to_xnx(x_data,xmin=1,xmax=None, norm = False):
    '''
    Count the number of occurrences in a data sample.
    Return the empirical histogram x,nx.
    We filter all observations that lie xmin<=x_data<=xmax
    INPUT:
    - x_data, list/array, with len(x_data)=#observations
    - xmin, int, minimum cutoff (default=1)
    - xmax, int, maximum cutoff (default=None --> infinity)
    - norm, bool, whether to normalize the histogram (default: False)
    OUTPUT:
    - x, nx: arrays; only return where nx>0.
    '''
    if xmax==None:
        xmax=np.inf

    c=Counter(x_data)
    x,nx = [],[]
    x_keys = sorted(list(c.keys()))

    for x_tmp in x_keys:
        if x_tmp>=xmin and x_tmp <= xmax:
            nx_tmp = c[x_tmp]
            x+=[x_tmp]
            nx+=[nx_tmp]
    x = np.array(x)
    nx = np.array(nx)
    if norm == True:
        nx = nx/float(np.sum(nx))
    return x,nx

def xdata_to_cum(x_data):
    arr_x = np.sort(x_data)
    V = len(x_data)
    arr_y = np.cumsum(np.ones(V)/V)
    arr_y[-1]=1
    return arr_x,arr_y


def x_autocorr(x_data,tau):
    '''
    autocorrelation for a timeseries x_data with lag tau.
    same result as pandas.Series.autocorr(lag=tau)
    '''
    if tau==0:
        x_t = x_data
        x_t_tau = x_data
    else:
        x_t = x_data[:-tau]
        x_t_tau = x_data[tau:]

    mu = np.mean(x_data)
    sigma = np.std(x_data)

    C_tau = np.mean( (x_t-mu)*(x_t_tau-mu)/sigma**2 )
    return C_tau

def x_autocorr_sm(x_data,nrep = 100, q = [2.5,97.5]):
    '''
    autocorrelation using statsmodels.
    faster for longer timeseries dueto use of fft
    - x_data timeseries
    - nrep, int, number of random realizations for null model 
    - q, percentiles for error

    Support: tau= 1,2,...,N/2 where N=len(x_data)
    '''

    N = int( len(x_data)/2 )
    x = np.arange(N+1)
    y = acf(x_data,unbiased=True,fft=True,nlags=N)#[1:]

    x_data_random = np.copy(x_data)
    y_random = np.zeros((nrep,N+1))
    for i_nrep in range(nrep):
        np.random.shuffle(x_data_random)
        y_random[i_nrep,:] =  acf(x_data_random,unbiased=True,fft=True,nlags=N)#[1:]
    y_mu = np.mean(y_random,axis=0)
    y_1,y_2 = np.percentile(y_random,q=[5,95],axis=0)


    ## calculate: F = \sum_t |C(t)|**2
    ## here we calculate from t=0,1,...,t^* where t^* is the first point
    ## where true C(t) is within the q-percentiles of the random
    ind_t =  np.where( (y<=y_2)*(y_1<=y) )[0] ## the first occurrence is always t=0
    # print(ind_t)
    if len(ind_t) == 1:
        ind_t_star = len(y)
    else:
        ind_t_star = ind_t[2]-1

    F = np.sum(y[:ind_t_star]**2)

    result = {}
    result['tau'] = x
    result['C'] = y
    result['C_r'] = y_mu
    result['C_r_err'] = [y_1,y_2] 
    result['F'] = F
    return result

def x_autocorr_sm_ext(x_data,nrep = 100, q = [2.5,97.5]):
    '''
    autocorrelation using statsmodels.
    faster for longer timeseries dueto use of fft
    - x_data timeseries
    - nrep, int, number of random realizations for null model 
    - q, percentiles for error

    Support: tau= 1,2,...,N/2 where N=len(x_data)
    '''

    N = int( len(x_data)/2 )
    x = np.arange(N+1)
    y_original = np.zeros((nrep,N+1))
    y_random = np.zeros((nrep,N+1))
    for i_nrep in range(nrep):
        ## periodic boundary conditions with randomly selected starting point
        i_rand = np.random.randint(N)
        x_data_i = np.append(x_data[i_nrep:],x_data[:i_nrep])
        y_original[i_nrep,:] = acf(x_data_i,unbiased=True,fft=True,nlags=N)#[1:]

        ## randomize
        np.random.shuffle(x_data_i)
        y_random[i_nrep,:] =  acf(x_data_i,unbiased=True,fft=True,nlags=N)#[1:]


    y_mu = np.mean(y_original,axis=0)
    y_1,y_2 = np.percentile(y_original,q=q,axis=0)

    y_mu_rand = np.mean(y_random,axis=0)
    y_1_rand,y_2_rand = np.percentile(y_random,q=q,axis=0)

    result = {}
    result['tau'] = x
    result['C'] = [y_mu,y_1,y_2]
    result['C_rand'] = [y_mu_rand,y_1_rand,y_2_rand]
    result['tmp'] = [y_original,y_random]
    return result

def KS_2sample(xdata1,xdata2):
    ## transform to x and px, respectively
    x1,px1 = xdata_to_xnx(xdata1,norm=True)
    x2,px2 = xdata_to_xnx(xdata2,norm=True)
    
    ## find common support
    x12 = np.sort(np.unique(np.concatenate([x1,x2])))
    
    ## where do original indices have to be put in common support
    inds12_1 = np.searchsorted(x12,x1,side='left')
    inds12_2 = np.searchsorted(x12,x2,side='left')
    
    ## redefine px over the common support
    px12_1 = 0.0*x12
    px12_1[inds12_1] = px1
    Fx12_1 = np.cumsum(px12_1)

    px12_2 = 0.0*x12
    px12_2[inds12_2] = px2
    Fx12_2 = np.cumsum(px12_2)
    
    KS = np.max(np.abs(Fx12_1-Fx12_2))
    return KS

def determine_tau(x_data,nrep=1000,q1=1.,q2=99.,log = False):
    '''
    Determine the correlation time tau:
    the mean-original is between the lower (q1) and the upper (q2) percentile of the random;
    TODO: 
    '''
    if log == True:
        x_data = np.log(x_data)
    result = x_autocorr_sm_ext(x_data,nrep=nrep,q=[q1,q2])
    x = result['tau']
    y_mu,y_1,y_2 = result['C']
    y_mu_rand,y_1_rand,y_2_rand = result['C_rand']

    ## we consider the autocorrelation cmopatible with random if:
    ## the mean-original is between the lower and the upper percentile of the random
    ## we take the third occurrence to prevent weird behaviour for small tau
    tau = x[np.where( (y_mu<=y_2_rand)&(y_mu>=y_1_rand) )[0][2]]
    return tau

def determine_tau_gasser(x_data,nrep=1000,q1=1.,q2=99.,log = False):
    '''
    Determine the correlation time tau according to Gasser, Biometrika (1975)
    sum C(tau)^2
    we pick a finite cut-off for the summation, where 
    the mean-original is between the lower (q1) and the upper (q2) percentile of the random
    '''
    if log == True:
        x_data = np.log(x_data)
    result = x_autocorr_sm_ext(x_data,nrep=nrep,q=[q1,q2])
    x = result['tau']
    y_mu,y_1,y_2 = result['C']
    y_mu_rand,y_1_rand,y_2_rand = result['C_rand']

    ## finite integration time
    ind_star = np.where( (y_mu<=y_2_rand)&(y_mu>=y_1_rand) )[0][2]
    tau = np.sum( y_mu[:ind_star]**2 )
    return tau
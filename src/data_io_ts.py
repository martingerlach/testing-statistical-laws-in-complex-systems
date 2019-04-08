import numpy as np
from collections import Counter
import os,sys
import pandas as pd

from modules_mcmc_zipf import sequence_muProcess
from modules_gaussian_lrc import gaussian_lrc_make_sequence
from modules_powerlaw_anticorrelated import powerlaw_anticorrelated_make_sequence
'''
Module to load different real and synthetic datasets
that are available as timeseries.
'''

def get_xdata_ts_wrapper(
    label, 
    dict_args ):
    '''
    wrapper for getting different datasets.
    label:
    - text_pg:  
        -pg_number
    - nw_s_01: network from edge-sampling a la network_sampling_01
        -p
        -i_nrep
    - earthquake: earthquake from sc-dataset
        - xmin
    - mcmc_zipf: markov chain from zipf distribution
        - Ntypes
        - Ntokens
        - alpha
        - mu
        - k
    '''
    if label == 'text_pg':
        X = get_text_pg(
            dict_args['pg_number'],
            dict_args['src_dir'])
    elif label == 'interevent_pg':
        X = get_interevent_pg(
            dict_args['pg_number'],
            dict_args['w'],
            dict_args['src_dir'],
            xmin = dict_args['xmin'])
    elif label == 'nw_s_01':
        X = get_network_sampling_01(
            dict_args['p'],
            dict_args['i_nrep'],
            dict_args['src_dir'])
    elif label == 'earthquakes':
        X = get_earthquakes(
            dict_args['src_dir'],
            xmin=dict_args['xmin'])
    elif label == 'mcmc_zipf':
        X = get_mcmc_zipf(
            dict_args['Ntypes'],
            dict_args['Ntokens'],
            dict_args['alpha'],
            dict_args['mu'],
            dict_args['k'])
    elif label == 'gaussian_lrc':
        X = get_gaussian_lrc(
            dict_args['N'],
            dict_args['src_dir'],)
    elif label == 'powerlaw_anticorrelated':
        X = get_powerlaw_anticorrelated(
            dict_args['Ntypes'],
            dict_args['Ntokens'],
            dict_args['xmin'],
            dict_args['alpha'],
            dict_args['B'])
    else:
        X = []
    ## transform to logarithm
    try:
        log_transform = dict_args['log']
    except KeyError:
        log_transform = False
    if log_transform == True:
        X = np.log(np.array(X))
    return np.array(X)


def get_network_sampling_01(
    p,
    i_nrep,
    src_dir, 
    ranks = True
    ):
    '''
    Sampling of edges in a network with local-nonlocal parameter p
    -p, float, probability to sample a local edge
    -i_nrep, int, realization
    -src_dir, str, path to src-folder
    -ranks, bool, True; assign ranks to the nodes
    Return
    -X, list, timeseries
    '''
    path_read = os.path.join(src_dir,os.pardir,'data','networks','sampling')
    # fname_read = 'networkI-p%s.dat_%s'%(p,i_nrep)
    fname_read = 'Snet-p%s-r%s.dat'%(p,i_nrep)

    filename_read = os.path.join(path_read,fname_read)

    with open(filename_read,'r') as f:
        x = f.readlines()

    X = [int(h) for h in x[0].split()]

    if ranks == True:
        X = xdata_rerank(X)

    # ## node-pairs are ordered low-high
    # ## switch up the ordering
    # X_new = 0*X
    # for i in range(int(len(X)/2)):
    #     x_tmp = 1*X[i*2:i*2+2]
    #     np.random.shuffle(x_tmp)
    #     X_new[i*2:i*2+2] = x_tmp
    # X = 1*X_new
    return X

def get_text_pg(
    pg_number,
    src_dir,
    ranks = True):
    '''
    Get text from a book from Project Gutenberg
    -pg, int, id for project gutenberg book
    -src_dir, str, path to src-folder
    -ranks, bool, True; assign ranks to the nodes
    Return
    -X, list, timeseries
    '''
    path_read = os.path.join(src_dir,os.pardir,'data','books')
    fname_read = 'PG%s_tokens.txt'%(pg_number)
    filename_read = os.path.join(path_read,fname_read)
    with open(filename_read,'r') as f:
        x= f.readlines()
        
    X = [ h.strip() for h in x]
    if ranks == True:
        X = xdata_rerank(X)
    return X

def get_interevent_pg(
    pg_number,
    word,
    src_dir,
    xmin = None):
    '''
    Get intervent times for a word in a book from Project Gutenberg
    -pg, int, id for project gutenberg book
    - word, str, 
    -src_dir, str, path to src-folder
    -xmin, int minimum event time filter (default: None == no filtering)
    -X, list, timeseries
    '''
    text = get_text_pg(pg_number,src_dir, ranks=False)
    ind_sel = np.where( np.array(text)==word)[0]
    list_tau = ind_sel[1:]-ind_sel[:-1]
    X = np.array(list_tau)
    if xmin is not None:
        X = np.array([ h for h in X if h>=xmin ])

    return X

def get_earthquakes(
    src_dir,
    fname_read = 'YSH_2010.hash',
    xmin = 2.0
    ):
    '''
    Eartquake data from 
    http://scedc.caltech.edu/research-tools/alt-2011-yang-hauksson-shearer.html

    -src_dir, str, path to src-folder
    -fname_read, str, name of file (default name from download)
    -xmin, float, minimum cutoff for magnitude
    Return
    -X, list, timeseries

    Note results are in magnitude (not seismic moments) with 2 digit precision
    '''
    path_read = os.path.join(src_dir,os.pardir,'data','earthquakes')
    filename_read = os.path.join(path_read,fname_read)
    # df = pd.read_table(filename_read,sep=' ')
    df=pd.read_table(filename_read,delim_whitespace=True,header=-1)

    ##http://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/SouthernCalifornia_1981-2011_focalmec_Format.pdf
    df=df.rename(
     columns={0: "year",
              1: "month",
              2:"day",
             3:"hour",
             4:"minute",
             5:"second",
             6:"cid",
             7:"latitude",
             8:"longitude",
             9:"depth",
             10:"magnitude"})
    X = np.array([h for h in df['magnitude'] if h >=xmin ])
    return X

def get_mcmc_zipf(
    Ntypes,
    Ntokens,
    alpha,
    mu,
    k):
    '''
    Draw a sequence from the correlated powerlaw markov chain sampler.
    - Ntypes, int, number of different states (support)
    - Ntokens, int, number of steps
    - alpha, float, power law exponent
    - mu, float, prob of global step (i.e. mu=0:only local, mu=1: only global)
    - k, int, # of neighbors for correlation
    '''

    X = sequence_muProcess(Ntypes,Ntokens,alpha,mu,k,xmin=1)
    return X

def get_gaussian_lrc(
    N,
    src_dir,):
    
    log2N = int(np.log2(N))
    X = gaussian_lrc_make_sequence(log2N, src_dir = src_dir)
    return X

def get_powerlaw_anticorrelated(
    Ntypes,
    Ntokens,
    xmin,
    alpha,
    B):
    X = powerlaw_anticorrelated_make_sequence(Ntokens, Ntypes, xmin, alpha, B, )
    return X

def xdata_rerank(X):
    ## replace labels by ranks
    c = Counter(X)

    list_r = []
    list_w = []
    list_n = []
    dict_w_r = {}
    r = 1
    for w,n in c.most_common():
        list_r += [r]
        list_w += [w]
        list_n += [n]
        dict_w_r[w]=r
        r+=1
    X_rank = []
    for w in X:
        X_rank += [dict_w_r[w]]
    return X_rank
        
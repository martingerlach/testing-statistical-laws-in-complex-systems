import numpy as np
import os,sys

def gaussian_lrc_make_sequence(log2N, src_dir = '', zscore = False, seed = None):
    '''
    gaussian long range correlations
    requies random_phase.c;
    - compile to exe via 
        gcc random_phase.c -o random_phase.exe -lm
    - generate 2**12 points
        ./random_phase.exe 12 output.out

    Note log2N generates 2**log2N points
    output array with N points
    '''
    if seed == None:
        rand_int = np.random.randint(2**16)
    else:
        rand_int = seed
    fname = 'output_%s.out'%(rand_int)
    os.system('cd %s; ./random_phase.exe %s %s %s'%(src_dir,log2N, fname, -rand_int))

    filename = os.path.join(src_dir,fname)
    with open(filename) as f:
        x = f.readlines()
    # print(filename)
    os.system('cd %s; rm -f %s'%(src_dir, fname))
    x_data = np.array([float(h.strip()) for h in x])
    if zscore == True:
        mu = np.mean(x_data)
        sigma = np.std(x_data)
        x_data = (x_data-mu)/sigma
    return x_data




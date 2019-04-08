# testing-statistical-laws-in-complex-systems

This repository contains the data and the code for the paper.

Martin Gerlach, Eduardo G. Altmann, [Testing Statistical Laws in Complex Systems](https://journals.aps.org/prl/accepted/a9073Y74J0119c5d69421da643af45289fa1d030a), Physical Review Letters, to appear (2019).



## Setup:

- get the repository:
```git clone https://github.com/martingerlach/testing-statistical-laws-in-complex-systems
```
- install the following packages (e.g. via pip): ``` numpy, scipy, statsmodels, pandas, matplotlib, mpmath```


## Navigation

- The folder ```code/``` contains notebooks with the analysis of each Dataset (.ipynb or .html)
  - Earthquake data: ```Analysis_real-data_earthquakes.ipynb```
  - Text Interevent data: ```Analysis_real-data_texts-interevent-times.ipynb```
  - Text Frequency-rank data: ```Analysis_real-data_texts-rank-frequency.ipynb```
  - Network data: ```Analysis_real-data_networks-degree.ipynb```
  - Synthetic data: ```Analysis_synthetic-data.ipynb```
- The code for all functions can be found in ```src/```


## Datasets

We analyze several datasets which we included in the repo. They are contained in the folder ```data/```

- Earthquakes: from the [Southern California Earthquake Data Center](http://scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/YSH_2010.hash)
- Books from Project Gutenberg: from the [Standardized Project Gutenberg Corpus](https://github.com/pgcorpus/gutenberg)
- Networks: [KONECT Project: Internet Topology](http://konect.cc/networks/topology) in the folder ```data/networks/as20000102```. The timeseries data for the sequence of degrees is in the folder ```data/networks/sampling``` and was obtained using the code in [ADD.]
- Synthetic data: The timeseries can be generated using the code in ```src/modules_mcmc_zipf.py```. There is an example of a correlated timeseries in ```data/synthetic/ts_synthetic_Ntypes1000_Ntokens100000_alpha1.5_mu0.01_k5```

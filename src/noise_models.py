
import numpy as np
import scipy as sp
import numpy.random as rand
import pandas as pd


import sklearn.linear_model as skllin
from numba import njit, vectorize, float64

import seaborn as sns

class BackgroundDist:
    # needs to have bin edge at zero, probability function, bin edges, will assume each bin has same weight, raise qcut error if not unique bins
    def __init__(self, data, ppbin=100):
        
        self.df = pd.DataFrame(np.c_[data], columns=['vals'])
        
        self.ppbin = ppbin
        
        self.ndata = len(self.df.index)
        self.nbins = self.ndata//self.ppbin
        
        bin_index, bin_edges = pd.qcut(self.df['vals'], self.nbins,  labels=False, retbins=True)
        self.df['bin_index'] = bin_index
        self.bin_edges = bin_edges
        
        self.density = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
        
        self.bin_mean = self.df.groupby('bin_index')['vals'].mean().values
        
        self.mean = np.mean(self.df['vals'])
        
    def get_data(self):
        return self.df['vals'].values
    
    def sample(self, size):
        return rand.choice(self.bin_mean, size=size, replace=True)
        
        
    def get_bin_index(self, vals):
                
        bin_idx = np.searchsorted(self.bin_edges, vals)
        
        # index zero indicates lower than smallest bin
        # index nbins+1 indicates larger than largest bin
        
        
        return bin_idx
        
    def bin_idx(self, vals, log_scale=False):
        
        bin_idx = self.get_bin_index(vals)
                        
        p = np.zeros_like(vals)
        
        idx = (bin_idx != 0) & (bin_idx != self.nbins+1)
        
        p[idx] = self.density[bin_idx[idx]-1]
          
        return p
    
    def plot(self, ax, color='b', label=None):
        
        ax.hist(self.bin_edges[:-1], self.bin_edges, weights=np.log(10)/(np.log(self.bin_edges[1:])-np.log(self.bin_edges[:-1]))/self.nbins,
       histtype='step', color=color, label=label)


@vectorize([float64(float64)])
def verf(x):
    return sp.special.erf(x)

class LogNormalBGNoise:
    
    def __init__(self, bg):
        
        self.bg = bg

    def calc_prob_meas(self, meas, predict, sigma):
        
        bin_idx = self.bg.get_bin_index(meas)
        edges = self.bg.bin_edges
        
        return LogNormalBGNoise.calc_prob_meas_(np.array(meas), np.array(predict), sigma, bin_idx, edges, self.bg.nbins)
        
    
    @staticmethod
    @njit
    def calc_prob_meas_(meas, predict, sigma, bin_idx, edges, nbins):

        p = np.zeros_like(meas)

        for i in range(len(meas)):

            x = verf((np.log(meas[i] - edges[:bin_idx[i]]) - np.log(predict[i]+1e-8))/sigma/np.sqrt(2))

            if bin_idx[i] > 1:

                p[i] += np.sum((x[:-1]-x[1:]) / (edges[1:bin_idx[i]] - edges[:bin_idx[i]-1]))

            if bin_idx[i] > 0 and bin_idx[i] <= nbins:

                p[i] += (x[-1] + 1) / (edges[bin_idx[i]] - edges[bin_idx[i]-1])

        p *= 0.5
        p /= nbins

        return p

    def cal_mean_conc(self, meas, predict, sigma):

        bin_idx = self.bg.get_bin_index(meas)
        edges = self.bg.bin_edges
        
        return LogNormalBGNoise.cal_mean_conc_(np.array(meas), np.array(predict), sigma, bin_idx, edges, self.bg.nbins)
        

    @staticmethod
    @njit
    def cal_mean_conc_(meas, predict, sigma, bin_idx, edges, nbins):


        probm = np.zeros_like(meas)
        meanc = np.zeros_like(meas)

        for i in range(len(meas)):

            x0 = verf((np.log(meas[i] - edges[:bin_idx[i]]) - np.log(predict[i]+1e-8))/sigma/np.sqrt(2))
            x1 = verf((np.log(meas[i] - edges[:bin_idx[i]]) - np.log(predict[i]+1e-8)-sigma**2)/sigma/np.sqrt(2))

            if bin_idx[i] > 1:
                probm[i] += np.sum((x0[:-1]-x0[1:]) / (edges[1:bin_idx[i]] - edges[:bin_idx[i]-1]))
                meanc[i] += np.sum((x1[:-1]-x1[1:]) / (edges[1:bin_idx[i]] - edges[:bin_idx[i]-1]))

            # think about the x[-1] index...
            if bin_idx[i] > 0 and bin_idx[i] <= nbins:
                probm[i] += (x0[-1] + 1) / (edges[bin_idx[i]] - edges[bin_idx[i]-1])
                meanc[i] += (x1[-1] + 1) / (edges[bin_idx[i]] - edges[bin_idx[i]-1])

        probm *= 1.0/2.0
        probm /= nbins

        meanc *= np.exp(np.log(predict+1e-8) + sigma**2/2.0)/2.0
        meanc /= nbins

        meanc[probm == 0] = 0.0
        meanc[probm > 0.0] /= probm[probm > 0.0]

        return meanc
    
    def sample(self, predict, sigma, decompose=False):
        
        background = self.bg.sample(len(predict))
        
        signal = rand.lognormal(mean=np.log(predict+1e-8), sigma=sigma)
        
        if decompose:
            return background, signal
        else:
            return background + signal
            
    
class LinearNoise:
    def __init__(self, in_data, out_data, verbose=False):
                
        self.df = pd.DataFrame(np.c_[in_data, out_data], columns=['in_data', 'out_data'])
        
        self.reg = skllin.LinearRegression()
        self.reg.fit(np.log10(np.c_[self.df['in_data']]), np.log10(self.df['out_data']))
        
    def transform(self, in_data, in_factor=1.0, out_factor=1.0):
        
        return out_factor*10**self.reg.predict(np.log10(np.c_[in_factor*in_data])) 
    
    def inverse_transform(self, out_data, in_factor=1.0, out_factor=1.0):
        
        return 10**((np.log10(out_data/out_factor)-self.reg.intercept_) / self.reg.coef_)/in_factor
        
        
    def plot(self, ax, cbar_ax=None, line=False):   
        
        print(self.reg.intercept_, self.reg.coef_)
        
        sns.histplot(self.df, x='in_data', y='out_data', 
                              bins=(100, 100), 
                         log_scale=(True, True), ax = ax, color='b')
        
        if line:
            x = np.logspace(np.log10(self.df['in_data'].min()), np.log10(self.df['in_data'].max()), base=10)
            y = self.transform(x)

            ax.plot(x, y, 'k--')

    
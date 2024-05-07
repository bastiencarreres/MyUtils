import numpy as np
import scipy.stats as sci_stat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fitnormtest(data, nbins=None, plot=False, range=[-5, 5]):
    N = len(data)

    ydata, xdata = np.histogram(data, bins=nbins, range=range)
    ynorm = ydata / (len(data) * (xdata[1] - xdata[0]))
    
    xcenters = 0.5 * (xdata[:-1] + xdata[1:])

    pars, cov = curve_fit(lambda x, mu, sig : sci_stat.norm.pdf(x, loc=mu, scale=sig), xcenters, ynorm, p0=[0,1])

    mask = ydata > 5
    
    cdf = sci_stat.norm.cdf(xdata, loc=pars[0], scale=pars[1]) 
    P = (cdf[1:] - cdf[:-1])[mask]

    testchi2 =  np.sum((ydata[mask] -  np.sum(ydata[mask]) * P)**2 / (np.sum(ydata[mask]) * P))
        
    print(f'mu = {pars[0]:.4f} +/- {np.sqrt(cov[0,0]):.4f}, sig = {pars[1]:.4f} +/- {np.sqrt(cov[1, 1]):.4f}')
    
    print(f'CHI2 = {testchi2:.2f}, Critical value at 95 % is {sci_stat.chi2.ppf(1 - 0.05, np.sum(mask) - 3):.2f}')
    
    k, p = sci_stat.normaltest(data[~data.isna()])
    
    print(f"D'agostino test p-value = {p:.4f}")
    if p < 0.05:
        print("Data are not compatible (alpha = 0.05) with normal distribution (D’Agostino test)")
    else:
        print("Data are compatible (alpha = 0.05) with normal distribution (D’Agostino test)")
    
    if np.sum(~data.isna()) < 5000:
        k,  p = sci_stat.shapiro(data[~data.isna()])
        print(f"Shapiro test p-value = {p:.4f}")
        if p < 0.05:
            print("Data are not compatible (alpha = 0.05) with normal distribution (Shapiro test)")
        else:
            print("Data are compatible (alpha = 0.05) with normal distribution (Shapiro test)")
    
    if plot:
        fig, ax = plt.subplots()
        ax.hist(data, bins=nbins, range=range, density=True)
        stats = '\n'.join([r'$\mu = $' + f'{pars[0]:.4f}' + r'$\pm$' + f'{np.sqrt(cov[0,0]):.4f}',
                           r'$\sigma = $' + f'{pars[1]:.4f}' + r'$\pm$' + f'{np.sqrt(cov[1,1]):.4f}'])
        props = dict(boxstyle='square', facecolor='none', edgecolor='none')
        plt.text(0.96, 0.85, stats, transform=ax.transAxes,  bbox=props, ha='right')
        x = np.linspace(range[0], range[1], 500)
        plt.plot(x, sci_stat.norm.pdf(x, loc=pars[0], scale=pars[1]))   
    
    
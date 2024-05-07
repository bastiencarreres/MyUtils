"""Module that contains sn recurent computation."""
import numpy as np
import astropy.cosmology as acosmo

def c_mu(mb, x1, c, M0, alpha, beta, 
         e_mb=None, e_x1=None, e_c=None, 
         cov_mb_x1=None, cov_mb_c=None, cov_x1_c=None, sint=None):
    mu = mb - (M0 - alpha * x1 +  beta * c)
    mu_cov = np.zeros(len(mb))
    if e_mb is not None:
        mu_cov += e_mb**2 + (alpha * e_x1)**2 + (beta * e_c)**2
        mu_cov += 2 * alpha * cov_mb_x1
        mu_cov += -2 * beta * cov_mb_c
        mu_cov += -2 * alpha * beta * cov_x1_c
        
    if sint is not None:
        mu_cov += sint**2
            
    return mu, np.sqrt(mu_cov)
    

    
    
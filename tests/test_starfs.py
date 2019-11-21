__all__ = ['test_spectra', 'test_photometry'] 

import os 
import pytest
import numpy as np 
# --- gqp_mc --- 
from starfs import dat_dir
from starfs import starfs as sFS

@pytest.mark.parametrize('dlogm', (0.2, 0.4))
def test_starfs(dlogm): 
    # read in M*, SFR
    mstar, logssfr = np.loadtxt(os.path.join(dat_dir(), 'tinker_SDSS_centrals_M9.7.dat'), 
            unpack=True, skiprows=2, usecols=[0,7]) 
    # calculate log M* and log SFR
    logm = np.log10(mstar)
    logsfr = logssfr + logm
    
    starFS = sFS.starFS()
    logm_sfs, logsfr_sfs, sig_logsfr_sfs = starFS.fit(logm, logsfr, n_bootstrap=10) 
    assert np.sum(np.isnan(logsfr_sfs)) == 0 

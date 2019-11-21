'''

fStarForMS = fitting the STAR FORming Main Sequence 

'''
import numpy as np 
import scipy as sp 
import warnings 
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture as GMix
# --- local --- 
from . import util as UT 


class starFS(object): 
    ''' class object for identifying the star formation sequence 
    (hereafter SFS) given the SFRs and M*s of a galaxy population 

    Main functionality of this class include : 
    * fitting log SFR of the SFMS in bins of log M*  
    * fitting parameterizations of the SFMS using log SFR 

    to-do
    * implement calculate f_SFS (the SFMS fraction)
    '''
    def __init__(self, fit_range=None):
        ''' initialize the fitting.

        Parameters
        ----------
        fit_range : list, optional 
            2 element list specifying the M* range -- [logM*_min, logM*_max]
        '''
        self._method = None
        self._error_method = None 
        self._fit_logm = None 
        self._fit_logssfr = None
        self._fit_logsfr = None
        self._sfms_fit = None 
    
        # log M* range of the fitting 
        self._logM_min = None 
        self._logM_max = None 
        if fit_range is not None: 
            self._logM_min = fit_range[0]
            self._logM_max = fit_range[1]

    def fit(self, logmstar, logsfr, logsfr_err=None, method='gaussmix', dlogm=0.2,
            Nbin_thresh=100, slope_prior=[0.0, 2.], max_comp=3, error_method='bootstrap', n_bootstrap=None,
            silent=False): 
        '''
        Given log SFR and log Mstar values of a galaxy population, 
        return the power-law best fit to the SFMS. After some initial 
        common sense cuts, P(log SSFR) in bins of stellar mass are fit 
        using specified method. 

        Parameters
        ----------
        logmstar : array 
            array of log M* of galaxy population 
        logsfr : array
            array of log SFR of galaxy population 
        method : str
            string specifying the method of identifying the SFS.
            - 'gaussmix' uses Gaussian Mixture Models to fit the P(SSFR) distribution. 
            - 'gaussmix_err' uses Gaussian Mixture Models *but* accounts for uncertainty
            in the SFRs. 
        dlogm : float, optional
            Default 0.2 dex. Width of logM* bins. 
        Nbin_thresh : float, optional 
             Default is 100. If a logM* bin has less than 100 galaxies, the bin is omitted. 
        error_method : str, optional
            Default is 'boostrap'. Method for estimating the uncertainties of the SFS fit 
            - 'bootstrap' for bootstrap uncertainties
            - 'jackknife' for jackknife uncertainties 
        n_bootstrap : int, optional
            Default is None. If error_method=='bootstrap', this specifies the number of 
            bootstrap samples. 
        silent : boolean, optiona
            If True, does not output any messages

        Returns 
        -------
        fit_logm, fit_logsfr : (array, array)
             
        Notes
        -----
        - Since the inputs are logM* and logSFR, SFR=0 is by construction 
        not accepted. 
            
        References
        ---------- 
        - Bluck et al., 2016 (arXiv:1607.03318)
        - Feldmann, 2017 (arXiv:1705.03014) 
        - Bisigello et al., 2017 (arXiv: 1706.06154)
        Gaussian Mixture Modelling: 
        - Kuhn and Feigelson, 2017 (arXiv:1711.11101) 
        - McLachlan and Peel, 2000
        - Lloyd, 1982 (k-means algorithm)
        - Dempster, Laird, Rubin, 1977 (EM algoritmh) 
        - Wu, 1983 (EM convergence) 
        '''
        if method not in ['gaussmix']: raise ValueError(method+" is not one of the methods!") 
        self._method = method 

        if error_method not in ['bootstrap']: raise ValueError("fitting currently only supports 'bootstrap'") 
        self._error_method = error_method

        self._dlogm = dlogm                 # logM* bin width
        self._Nbin_thresh = Nbin_thresh     # Nbin threshold 

        self._check_input(logmstar, logsfr, logsfr_err) # check inputs 

        if (self._logM_min is None) or (self._logM_max is None): # get M* range from the input logM*
            # the fit range will be a bit padded.
            self._logM_min = int(logmstar.min()/dlogm)*dlogm
            self._logM_max = np.ceil(logmstar.max()/dlogm)*dlogm

        mass_cut = (logmstar > self._logM_min) & (logmstar < self._logM_max)
        if np.sum(mass_cut) == 0: 
            print("trying to fit SFMS over range %f < log M* < %f" % (self._logM_min, self._logM_max))
            print("input spans %f < log M* < %f" % (logmstar.min(), logmstar.max()))
            raise ValueError("no galaxies within that cut!")
    
        self._mbins = self._get_mbin(self._logM_min, self._logM_max, self._dlogm) # log M* binning 

        # log M* bins with more than Nbin_thresh galaxies
        bin_cnt, _ = np.histogram(logmstar, bins=np.append(self._mbins[:,0], self._mbins[-1,1]))
        self._has_nbinthresh = (bin_cnt > Nbin_thresh) 
        #self._mbins_sfs = self._mbins_nbinthresh.copy() 

        if 'gaussmix' in method:
            self._GMMfit(logmstar, logsfr, max_comp=max_comp, slope_prior=slope_prior, n_bootstrap=n_bootstrap)
        else: 
            raise NotImplementedError

        # save the fit ssfr and logm 
        self._fit_logm = self._theta_sfs[:,0]
        self._fit_logssfr = self._theta_sfs[:,1]
        self._fit_logsfr = self._fit_logm + self._fit_logssfr
        self._fit_err_logssfr = self._err_sfs[:,1] 
        return [self._fit_logm, self._fit_logsfr, self._fit_err_logssfr]
    
    def _GMMfit(self, logmstar, logsfr, max_comp=3, slope_prior=[0., 2.], n_bootstrap=None): 
        ''' First fit the p(log SSFR) distributions of each logM* bin using
        a Guassian mixture model with at most `max_comp` components and some
        common sense priors, which are based on the fact that the SFS is 
        roughly a log-normal distribution. Afterwards, the SFS is identified 
        using `self._GMM_compID`. 


        This method does not require a log(SSFR) cut like gaussfit and negbinomfit 
        and can be flexibly applied to a wide range of SSFR distributions.
        '''
        # fit GMMs 
        logm_median, gbests, nbests, _gmms, _bics = self._GMMfit_bins(logmstar, logsfr, max_comp=max_comp)
    
        self._mbins_median = logm_median
        self._gbests = gbests # best-fit GMMs
        self._nbests = nbests # number of components in the best-fit GMMs 
        self._gmms = _gmms # all the GMMs
        self._bics = _bics # all the BICs

        # identify the SFS and other components from the best-fit GMMs 
        icomps = self._GMM_compID(gbests, logm_median, slope_prior=slope_prior) # i_sfs's, i_q's, i_int's, i_sb's     
        i_sfss, i_qs, i_ints, i_sbs = icomps 

        m_gmm, s_gmm, w_gmm = self._GMM_comps_msw(gbests, icomps)

        # get bootstrap errors for all GMM parameters
        m_gmm_boot, s_gmm_boot, w_gmm_boot = [], [], []  # positions, widths, and weights of the bootstrap GMMs  
        for ibs in range(n_bootstrap): 
            # resample the data w/ replacement 
            i_boot = np.random.choice(np.arange(len(logmstar)), size=len(logmstar), replace=True) 
            # fit GMMs with n_best components 
            gboots = self._GMMfit_bins_nbest(logmstar[i_boot], logsfr[i_boot], nbests)
            # identify the components 
            icomps_boot = self._GMM_compID(gboots, logm_median, slope_prior=slope_prior) # i_sfs's, i_q's, i_int's, i_sb's     
            
            _m_gmm_boot, _s_gmm_boot, _w_gmm_boot = self._GMM_comps_msw(gboots, icomps_boot)

            m_gmm_boot.append(_m_gmm_boot) 
            s_gmm_boot.append(_s_gmm_boot) 
            w_gmm_boot.append(_w_gmm_boot) 
        m_gmm_boot = np.array(m_gmm_boot)
        s_gmm_boot = np.array(s_gmm_boot)
        w_gmm_boot = np.array(w_gmm_boot)
        
        # now loop through the logM* bins and calculate bootstrap uncertainties 
        merr_boot = np.tile(-999., (len(gbests), 8))
        serr_boot = np.tile(-999., (len(gbests), 8))
        werr_boot = np.tile(-999., (len(gbests), 8))
        for i in range(len(gbests)): 
            for icomp, comp in zip([0, 1, 2, 5], [i_sfss, i_qs, i_ints, i_sbs]):
                if comp[i] is None: continue 
                if icomp <= 1: 
                    hascomp = (m_gmm_boot[:,i,icomp] != -999.) 
                    merr_boot[i,icomp] = np.std(m_gmm_boot[hascomp,i,icomp]) 
                    serr_boot[i,icomp] = np.std(s_gmm_boot[hascomp,i,icomp]) 
                    werr_boot[i,icomp] = np.std(np.concatenate([w_gmm_boot[hascomp,i,icomp], np.zeros(np.sum(~hascomp))]))
                else: 
                    for ii in range(len(comp[i])): 
                        hascomp = (m_gmm_boot[:,i,icomp+ii] != -999.) 
                        merr_boot[i,icomp+ii] = np.std(m_gmm_boot[hascomp,i,icomp+ii]) 
                        serr_boot[i,icomp+ii] = np.std(s_gmm_boot[hascomp,i,icomp+ii]) 
                        werr_boot[i,icomp+ii] = np.std(np.concatenate([w_gmm_boot[hascomp,i,icomp+ii], np.zeros(np.sum(~hascomp))]))
        tt_sfs  = [] # logM*, mu, sigma, weight of bestfit GMM  
        tt_q    = [] 
        tt_int  = [] 
        tt_int1 = [] 
        tt_int2 = [] 
        tt_sbs  = [] 
        tt_sbs1 = [] 
        tt_sbs2 = [] 
        tt_sfs_boot     = [] # mu_err, sigma_err, weight_err from boostrap 
        tt_q_boot       = [] 
        tt_int_boot     = [] 
        tt_int1_boot    = [] 
        tt_int2_boot    = [] 
        tt_sbs_boot     = [] 
        tt_sbs1_boot    = [] 
        tt_sbs2_boot    = [] 
        for _i, logm_med in enumerate(logm_median): 
            if i_sfss[_i] is not None: 
                tt_sfs.append(np.array([logm_med, m_gmm[_i,0], s_gmm[_i,0], w_gmm[_i,0]]))
                tt_sfs_boot.append(np.array([merr_boot[_i,0], serr_boot[_i,0], werr_boot[_i,0]]))
            if i_qs[_i] is not None: 
                tt_q.append(np.array([logm_med, m_gmm[_i,1], s_gmm[_i,1], w_gmm[_i,1]]))
                tt_q_boot.append(np.array([merr_boot[_i,1], serr_boot[_i,1], werr_boot[_i,1]]))
            if i_ints[_i] is not None: 
                for ii, _tt_int, _tt_int_boot in zip(range(len(i_ints[_i])), [tt_int, tt_int1, tt_int2], [tt_int_boot, tt_int1_boot, tt_int2_boot]): 
                    _tt_int.append(np.array([logm_med, m_gmm[_i,2+ii], s_gmm[_i,2+ii], w_gmm[_i,2+ii]]))
                    _tt_int_boot.append(np.array([merr_boot[_i,2+ii], serr_boot[_i,2+ii], werr_boot[_i,2+ii]]))
            if i_sbs[_i] is not None: 
                for ii, _tt_sbs, _tt_sbs_boot in zip(range(len(i_sbs[_i])), [tt_sbs, tt_sbs1, tt_sbs2], [tt_sbs_boot, tt_sbs1_boot, tt_sbs2_boot]): 
                    _tt_sbs.append(np.array([logm_med, m_gmm[_i,5+ii], s_gmm[_i,5+ii], w_gmm[_i,5+ii]])) 
                    _tt_sbs_boot.append(np.array([merr_boot[_i,5+ii], serr_boot[_i,5+ii], werr_boot[_i,5+ii]]))

        self._theta_sfs     = np.array(tt_sfs) 
        self._err_sfs       = np.array(tt_sfs_boot) 
        self._theta_q       = np.array(tt_q) 
        self._err_q         = np.array(tt_q_boot) 
        self._theta_int     = np.array(tt_int) 
        self._err_int       = np.array(tt_int_boot) 
        self._theta_int1    = np.array(tt_int1) 
        self._err_int1      = np.array(tt_int1_boot) 
        self._theta_int2    = np.array(tt_int2) 
        self._err_int2      = np.array(tt_int2_boot) 
        self._theta_sbs     = np.array(tt_sbs) 
        self._err_sbs       = np.array(tt_sbs_boot) 
        self._theta_sbs1    = np.array(tt_sbs1) 
        self._err_sbs1      = np.array(tt_sbs1_boot) 
        self._theta_sbs2    = np.array(tt_sbs2) 
        self._err_sbs2      = np.array(tt_sbs2_boot) 
        return None 

    def _GMMfit_bins(self, logmstar, logsfr, max_comp=3): 
        ''' Fit GMM components to P(SSFR) of given data and return best-fit
        '''
        n_bin = self._mbins.shape[0] # number of stellar mass bins.
    
        # sort logM* into M* bins
        i_bins = np.digitize(logmstar, np.append(self._mbins[:,0], self._mbins[-1,1]))
        i_bins -= 1
    
        bin_mid, gbests, nbests, _gmms, _bics = [], [], [], [], [] 

        # fit GMM to p(SSFR) in each log M* bins  
        for i in range(n_bin): 
            # if there are not enough galaxies 
            if not self._has_nbinthresh[i]: continue 
            in_bin = (i_bins == i)  

            x = logsfr[in_bin] - logmstar[in_bin] # logSSFRs
            x = np.reshape(x, (-1,1))

            bin_mid.append(np.median(logmstar[in_bin])) 
            
            # fit GMMs with a range of components 
            ncomps = range(1, max_comp+1)
            gmms, bics = [], []  
            for i_n, n in enumerate(ncomps): 
                gmm = GMix(n_components=n)
                gmm.fit(x)
                bics.append(gmm.bic(x)) # bayesian information criteria
                gmms.append(gmm)

            # components with the lowest BIC (preferred)
            i_best = np.array(bics).argmin()
            n_best = ncomps[i_best] # number of components of the best-fit 
            gbest = gmms[i_best] # best fit GMM 
            
            # save the best gmm, all the gmms, and bics 
            nbests.append(n_best) 
            gbests.append(gbest)
            _gmms.append(gmms) 
            _bics.append(bics)
        
        if bin_mid[0] > 10.: 
            warnings.warn("The lowest M* bin is greater than 10^10, this may compromise the SFS identification scheme") 
        return bin_mid, gbests, nbests, _gmms, _bics
    
    def _GMMfit_bins_nbest(self, logmstar, logsfr, nbests): 
        ''' Fit GMM components to P(SSFR) of given data and return best-fit
        '''
        n_bin = self._mbins.shape[0]
        i_bins = np.digitize(logmstar, np.append(self._mbins[:,0], self._mbins[-1,1]))
        i_bins -= 1 

        gmms = [] 
        ii = 0 
        for i in range(n_bin): 
            # if there are not enough galaxies 
            if not self._has_nbinthresh[i]: continue 
            in_bin = (i_bins == i)  

            x = logsfr[in_bin] - logmstar[in_bin] # logSSFRs
            x = np.reshape(x, (-1,1))
    
            gmm = GMix(n_components=nbests[ii])
            gmm.fit(x)
            # save the best gmm, all the gmms, and bics 
            gmms.append(gmm) 
            ii += 1
        return gmms

    def _GMM_compID(self, gbests, logm, slope_prior=[0., 2.]): 
        ''' Given the best-fit GMMs for all the stellar mass bins, identify the SFS 
        and the other components. STarting from the lowest M* bin, we identify
        the SFS based on the highest weight component. Then in the next M* bin, we 
        iteratively determine whether the highest weight component is with dev_thresh 
        of the previous M* bin. 
        '''
        i_sfss, i_qs, i_ints, i_sbs = [], [], [], [] 
        for ibin, gmm in enumerate(gbests): 
            mu_gmm = gmm.means_.flatten()
            w_gmm  = gmm.weights_
            n_gmm  = len(mu_gmm) 

            i_sfs, i_q, i_int, i_sb = None, None, None, None

            if ibin == 0: # lowest M* bin SFS is the highest weight bin 
                i_sfs       = np.argmax(w_gmm)
                mu_sfs_im1  = mu_gmm[i_sfs]
                logm_im1    = logm[ibin]
            else: 
                # only GMMs with SFRs within the SFS slope priors can be 
                # selected as SFS component.
                dlogm = logm[ibin] - logm_im1 
                ssfr_range = [dlogm * (slope_prior[0] - 1.), dlogm * (slope_prior[1] - 1.)] # the -1 comes from converting to SSFR
                #print '-------------------------------------------------'
                #print logm_im1, logm[ibin], dlogm 
                #print mu_sfs_im1
                #print mu_sfs_im1 + ssfr_range[0], mu_sfs_im1 + ssfr_range[1]
                #print mu_gmm

                potential_sfs = ((mu_gmm > mu_sfs_im1 + ssfr_range[0]) & (mu_gmm < mu_sfs_im1 + ssfr_range[1]))
                if np.sum(potential_sfs) > 0: 
                    # select GMM with the highest weight within the bins 
                    i_sfs = (np.arange(n_gmm)[potential_sfs])[w_gmm[potential_sfs].argmax()]
                    mu_sfs_im1  = mu_gmm[i_sfs]
                    logm_im1    = logm[ibin]
        
            # if there's a component with high SFR than SFMS -- i.e. star-burst 
            if i_sfs is not None: 
                above_sfs = (mu_gmm > mu_gmm[i_sfs])
                if np.sum(above_sfs) > 0: 
                    i_sb = np.arange(n_gmm)[above_sfs]

            # lowest SSFR component with SSFR less than SFMS will be designated as the 
            # quenched component 
            if i_sfs is not None: 
                notsf = (mu_gmm < mu_gmm[i_sfs]) #& (mu_gbest < -11.) 
                if np.sum(notsf) > 0: 
                    i_q = (np.arange(n_gmm)[notsf])[mu_gmm[notsf].argmin()]
                    # check if there's an intermediate population 
                    interm = (mu_gmm < mu_gmm[i_sfs]) & (mu_gmm > mu_gmm[i_q]) 
                    if np.sum(interm) > 0: 
                        i_int = np.arange(n_gmm)[interm]
            else: # no SFMS 
                i_q = (np.arange(n_gmm))[mu_gmm.argmin()]
                # check if there's an intermediate population 
                interm = (mu_gmm > mu_gmm[i_q]) 
                if np.sum(interm) > 0: 
                    i_int = np.arange(n_gmm)[interm]
            i_sfss.append(i_sfs)
            i_qs.append(i_q)
            i_ints.append(i_int) 
            i_sbs.append(i_sb) 
        return i_sfss, i_qs, i_ints, i_sbs

    def _GMM_comps_msw(self, gbests, icomps): 
        ''' given the best fit GMMs and the component indices, put them into arrays
        '''
        i_sfss, i_qs, i_ints, i_sbs = icomps 

        m_gmm = np.tile(-999., (len(gbests), 8)) # positions of the best-fit GMM components in each of the logM* bins 
        s_gmm = np.tile(-999., (len(gbests), 8)) # width of the best-fit GMM components in each of the logM* bins 
        w_gmm = np.tile(-999., (len(gbests), 8)) # weights of the best-fit GMM components in each of the logM* bins 

        for i, i0, i1, i2, i3 in zip(range(len(gbests)), i_sfss, i_qs, i_ints, i_sbs): 
            if i0 is not None:
                m_gmm[i,0] = gbests[i].means_.flatten()[i0]
                s_gmm[i,0] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i0]))
                w_gmm[i,0] = gbests[i].weights_[i0]
            if i1 is not None:
                m_gmm[i,1] = gbests[i].means_.flatten()[i1]
                s_gmm[i,1] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i1]))
                w_gmm[i,1] = gbests[i].weights_[i1]
            if i2 is not None:
                assert len(i2) <= 3
                for ii, i2i in enumerate(i2): 
                    m_gmm[i,2+ii] = gbests[i].means_.flatten()[i2i]
                    s_gmm[i,2+ii] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i2i]))
                    w_gmm[i,2+ii] = gbests[i].weights_[i2i]
            if i3 is not None:
                assert len(i3) <= 3
                for ii, i3i in enumerate(i3): 
                    m_gmm[i,5+ii] = gbests[i].means_.flatten()[i3i]
                    s_gmm[i,5+ii] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i3i]))
                    w_gmm[i,5+ii] = gbests[i].weights_[i3i]
        return m_gmm, s_gmm, w_gmm 

    def powerlaw(self, logMfid=None, mlim=None, silent=True): 
        ''' Find the best-fit power-law parameterization of the 
        SFMS from the logM* and log SFR_SFMS fit from the `fit` 
        method above. This is the simplest fit possible

        f_SFMS(log M*)  = a * (log M* - logM_fid) + b 

        Parameters
        ----------
        logMfid : (float) 
            Fiducial log M_*. 

        Returns
        -------
        sfms_fit : (function)
            f_SFMS(logM*)
        '''
        if self._fit_logm is None  or self._fit_logssfr is None or self._fit_logsfr is None: 
            raise ValueError('Run `fit` method first')

        # fiducial log M*  
        if logMfid is None: 
            logMfid = int(np.round(np.median(self._fit_logm)/0.5))*0.5
            print('fiducial log M* ='+str(logMfid))
        self._logMfid = logMfid

        # now fit line to the fit_Mstar and fit_SSFR values
        xx = self._fit_logm - logMfid  # log Mstar - log M_fid
        yy = self._fit_logsfr
        err = self._fit_err_logssfr
        if mlim is not None: 
            mcut = ((self._fit_logm > mlim[0]) & (self._fit_logm < mlim[1])) 
            xx = xx[mcut]
            yy = yy[mcut] 
            err = err[mcut]

        # chi-squared
        chisq = lambda theta: np.sum((theta[0] * xx + theta[1] - yy)**2/err**2)

        #A = np.vstack([xx, np.ones(len(xx))]).T
        #m, c = np.linalg.lstsq(A, yy)[0] 
        tt = sp.optimize.minimize(chisq, np.array([0.8, 0.3])) 

        self._powerlaw_m = tt['x'][0]
        self._powerlaw_c = tt['x'][1]
        
        sfms_fit = lambda mm: tt['x'][0] * (mm - logMfid) + tt['x'][1]
        self._sfms_fit = sfms_fit 
        if not silent: 
            print('logSFR_SFMS = %s (logM* - %s) + %s' % (str(round(m, 3)), str(round(logMfid,3)), str(round(c, 3))))
        return sfms_fit 
    
    def d_SFS(self, logmstar, logsfr, method='interpexterp', err_thresh=None, silent=True): 
        ''' Calculate the `distance` from the best-fit star-forming sequence 
        '''
        # check that .fit() has been run
        if self._fit_method is None: 
            msg_err = ''.join(["Cannot calculate the distance to the best-fit", 
                " star forming sequence without first fitting the sequence"]) 
            raise ValueError(msg_err) 

        if method == 'powerlaw':  
            # fit a powerlaw to the GMM SFS fits and then use it 
            # to measure the dSFS 
            fsfms = lambda mm: self._powerlaw_m * (mm - self._logMfid) + self._powerlaw_c 
            dsfs = logsfr - fsfms(logmstar) 

        elif method in ['interpexterp', 'nointerp']: 
            # instead of fitting a powerlaw use the actual SFS fits in order 
            # to calculate the dSFS values 

            # get stellar mass bins 
            mlow, mhigh = self._mbins.T
            n_mbins = len(mlow) 
            hasfit = np.zeros(n_mbins).astype(bool) 
            for i_m in range(n_mbins): 
                fitinmbin = ((self._fit_logm >= mlow[i_m]) & (self._fit_logm < mhigh[i_m])) 
                if np.sum(fitinmbin) == 1: 
                    hasfit[i_m] = True
                elif np.sum(fitinmbin) > 1: 
                    raise ValueError 
            mlow = mlow[hasfit]
            mhigh = mhigh[hasfit]
            n_mbins = np.sum(hasfit)
        
            # impose stellar mass limit based on the stellar mass range of the SFMS fits
            inmlim = ((logmstar > mlow.min()) & (logmstar < mhigh.max()) & np.isfinite(logsfr)) 
            if not silent: 
                print('SFMS fit ranges in logM* from %f to %f' % (mlow.min(), mhigh.max())) 
                print('%i objects are outside of this range and be assigned d_SFS = -999.' % (np.sum(~inmlim)))
            
            # error threshold to remove noisy SFMS bins
            if err_thresh is not None: 
                if self._fit_err_logssfr is None: 
                    raise ValueError("run fit with fit_error enabled")

                notnoisy = (self._fit_err_logssfr < err_thresh) 
                fit_logm = self._fit_logm[notnoisy]
                fit_logsfr = self._fit_logsfr[notnoisy]
            else: 
                fit_logm = self._fit_logm
                fit_logsfr = self._fit_logsfr

            # calculate dsfs 
            dsfs = np.tile(-999., len(logmstar))
            if method == 'interpexterp': 
                # linear interpolation with extrapolation beyond
                fsfms = sp.interpolate.interp1d(fit_logm, fit_logsfr, kind='linear', 
                        fill_value='extrapolate') 
                #if not extrap: 
                #    dsfs[inmlim] = logsfr[inmlim] - fsfms(logmstar[inmlim]) 
                dsfs = logsfr - fsfms(logmstar) 
            elif method == 'nointerp': 
                fsfms = sp.interpolate.interp1d(fit_logm, fit_logsfr, kind='nearest') 
                in_interp = (logmstar >= self._fit_logm.min()) & (logmstar <= self._fit_logm.max())
                dsfs[inmlim & in_interp] = logsfr[inmlim & in_interp] - fsfms(logmstar[inmlim & in_interp]) 
                below = (logmstar < self._fit_logm.min())
                dsfs[inmlim & below] = logsfr[inmlim & below] - \
                        self._fit_logsfr[np.argmin(self._fit_logm)]
                above = (logmstar > self._fit_logm.max())
                dsfs[inmlim & above] = logsfr[inmlim & above] - \
                        self._fit_logsfr[np.argmax(self._fit_logm)]
        return dsfs 

    def frac_SFMS(self): 
        ''' Return the estimate of the fraction of galaxies that are on 
        the star formation main sequence as a function of mass produce from 
        the fit. 
        '''
        if self._fit_logm is None  or self._frac_sfms is None:
            raise ValueError('Run `fit` method first')
        if isinstance(self._frac_sfms[0], str): 
            raise NotImplementedError(self._frac_sfms[0]) 
        return [self._fit_logm, self._frac_sfms]

    def _GMM_idcomp(self, gbest, SSFR_cut=None, silent=True): 
        ''' Given the best-fit GMM, identify all the components
        '''
        if SSFR_cut is None: 
            SSFR_cut = -11.
        mu_gbest = gbest.means_.flatten()
        w_gbest  = gbest.weights_
        n_gbest  = len(mu_gbest) 
        
        i_sfms = None # sfms 
        i_sb = None # star-burst 
        i_int = None # intermediate 
        i_q = None # quenched
    
        highsfr = (mu_gbest > SSFR_cut) 
        if np.sum(highsfr) == 1: 
            # only one component with high sfr. This is the one 
            i_sfms = np.arange(n_gbest)[highsfr][0]
        elif np.sum(mu_gbest > SSFR_cut) > 1: 
            # if best fit has more than one component with high SFR (logSSFR > -11), 
            # we designate the component with the highest weight as the SFMS 
            highsfr = (mu_gbest > SSFR_cut)
            i_sfms = (np.arange(n_gbest)[highsfr])[w_gbest[highsfr].argmax()]
        else: 
            # no components with high sfr -- i.e. no SFMS component 
            pass 

        # lowest SSFR component with SSFR less than SFMS will be designated as the 
        # quenched component 
        if i_sfms is not None: 
            notsf = (mu_gbest < mu_gbest[i_sfms]) #& (mu_gbest < -11.) 
            if np.sum(notsf) > 0: 
                i_q = (np.arange(n_gbest)[notsf])[mu_gbest[notsf].argmin()]
                # check if there's an intermediate population 
                interm = (mu_gbest < mu_gbest[i_sfms]) & (mu_gbest > mu_gbest[i_q]) 
            #else: 
            #    interm = (mu_gbest < mu_gbest[i_sfms]) & (mu_gbest > -11.) 
                if np.sum(interm) > 0: 
                    i_int = np.arange(n_gbest)[interm]
        else: # no SFMS 
            #notsf = (mu_gbest < -11.) 
            #if np.sum(notsf) > 0: 
                #i_q = (np.arange(n_gbest)[notsf])[mu_gbest[notsf].argmin()]
            i_q = (np.arange(n_gbest))[mu_gbest.argmin()]
            # check if there's an intermediate population 
            interm = (mu_gbest > mu_gbest[i_q]) 
            if np.sum(interm) > 0: 
                i_int = np.arange(n_gbest)[interm]

        # if there's a component with high SFR than SFMS -- i.e. star-burst 
        if i_sfms is not None: 
            above_sfms = (mu_gbest > mu_gbest[i_sfms])
            if np.sum(above_sfms) > 0: 
                i_sb = np.arange(n_gbest)[above_sfms]
        return [i_sfms, i_q, i_int, i_sb] 
    
    def _check_input(self, logmstar, logsfr, logsfr_err): 
        ''' check whether input logMstar or logSFR values make sense!
        '''
        if len(logmstar) != len(logsfr): 
            raise ValueError("logmstar and logsfr are not the same length arrays") 
        if np.sum(logmstar < 0.) > 0: 
            raise ValueError("There are negative values of log M*")  
        if np.sum(logmstar > 13.) > 0: 
            warnings.warn("There are galaxies with log M* > 13. ... that's weird") 
        if np.sum(np.invert(np.isfinite(logsfr))) > 0: 
            raise ValueError("There are non-finite log SFR values")  
        if logsfr_err is not None: 
            if not np.all(np.isfinite(logsfr_err)): 
                raise ValueError("There are non-finite log SFR error values")  
        return None 
    
    def _get_mbin(self, logMmin, logMmax, dlogm):  
        ''' return log M* binning  
        '''
        mbin_low = np.arange(logMmin, logMmax, dlogm)
        mbin_high = mbin_low + dlogm
        return np.array([mbin_low, mbin_high]).T


class xdGMM(object): 
    ''' Wrapper for extreme_deconovolution. Methods are structured similar
    to GMM
    '''
    def __init__(self, n_components): 
        '''
        '''
        self.n_components = n_components
        self.l = None
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X, Xerr): 
        ''' fit GMM to X and Xerr
        '''
        from extreme_deconvolution import extreme_deconvolution
        X, Xerr = self._X_check(X, Xerr)
        self._X = X 
        self._Xerr = Xerr
        gmm = GMM(self.n_components, n_iter=10, covariance_type='full').fit(X)
        w, m, c = gmm.weights_.copy(), gmm.means_.copy(), gmm.covars_.copy()
        l = extreme_deconvolution(X, Xerr, w, m, c)
        self.l = l 
        self.weights_ = w 
        self.means_ = m
        self.covariances_ = c
        return None

    def logL(self, X, Xerr): 
        ''' log Likelihood of the fit. 
        '''
        if (self.l is None) or (not np.array_equal(X, self._X)) or (not np.array_equal(Xerr, self._Xerr)): 
            self.fit(X, Xerr)
        X, Xerr = self._X_check(X, Xerr)
        return self.l * X.shape[0]

    def _X_check(self, X, Xerr): 
        ''' correct array shape of X and Xerr to be compatible 
        with all the methods in this class 
        '''
        if len(X.shape) == 1: 
            X = np.reshape(X, (-1,1))
        if len(Xerr.shape) == 1:   
            Xerr = np.reshape(Xerr, (-1,1,1))
        return X, Xerr
    
    def bic(self, X, Xerr): 
        ''' calculate the bayesian information criteria
        -2 ln(L) + Npar ln(Nsample) 
        '''
        if (self.l is None) or (not np.array_equal(X, self._X)) or (not np.array_equal(Xerr, self._Xerr)): 
            self.fit(X, Xerr)
        X, Xerr = self._X_check(X, Xerr)
        assert np.array_equal(X, self._X)  
        return (-2 * self.l * X.shape[0] + self._n_parameters() * np.log(X.shape[0])) 
    
    def _n_parameters(self): 
        ''' number of paramters in the model. 
        '''
        _, n_features = self.means_.shape
        cov_params = self.n_components * n_features * (n_features + 1) / 2.
        mean_params = n_features  * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)


def sfr_mstar_gmm(logmstar, logsfr, n_comp_max=30, silent=False): 
    ''' Fit a 2D gaussian mixture model to the 
    log(M*) and log(SFR) sample of galaxies, 
    '''
    # only keep sensible logmstar and log sfr
    sense = (logmstar > 0.) & (logmstar < 13) & (logsfr > -5) & (logsfr < 4) & (np.isnan(logsfr) == False)
    if (len(logmstar) - np.sum(sense) > 0) and not silent: 
        warnings.warn(str(len(logmstar) - np.sum(sense))+' galaxies have nonsensical logM* or logSFR values')  
    logmstar = logmstar[np.where(sense)]
    logsfr = logsfr[np.where(sense)]

    X = np.array([logmstar, logsfr]).T # (n_sample, n_features) 

    gmms, bics = [], []  
    for i_n, n in enumerate(range(1, n_comp_max)): 
        gmm = GMix(n_components=n)
        gmm.fit(X)
        gmms.append(gmm)
        bics.append(gmm.bic(X)) # bayesian information criteria
    ibest = np.array(bics).argmin() # lower the better!
    gbest = gmms[ibest]

    if not silent: 
        print(str(len(gbest.means_))+' components') 
    return gbest 

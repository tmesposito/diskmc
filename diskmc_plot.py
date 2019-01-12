#!/usr/bin/env python

import os
import pdb
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from emcee import autocorr


class sedMcfost:
    """
    Object class for MCFOST SED models.
    
    Typical dimensions: Stokes I, Q, U, V, star only, star scattered light, thermal, scattered light from dust thermal emission
    """
    
    def __init__(self, hdu, inc=slice(None), az=slice(None), fluxconv=1.):
        """
        hdu: HDUList from an MCFOST sed_rt.fits file.
        inc: int index of single inclination to use. If None, retain all inclinations.
        az: int index of single azimuth to use. If None, retain all azimuths.
        fluxconv: flt flux conversion factor to apply.
        """
        
        data = hdu[0].data
        self.wave = hdu[1].data
        
        # Apply flux conversion to all components of SED.
        self.fluxconv = fluxconv
        self.fluxconv2 = None
        
        # Different formats based on MCFOST version.
        if data.shape[0] == 5:
            self.I = fluxconv*data[0,az,inc,:] # total intensity, star + dust
            self.star_I = fluxconv*data[1,az,inc,:] # total intensity from star only
            self.star_scat = fluxconv*data[2,az,inc,:] # scattered total intensity from star only
            self.thermal = fluxconv*data[3,az,inc,:] # thermal emission from disk
            self.dust_scat = fluxconv*data[4,az,inc,:] # scattered light from dust only
        else:
            self.I = fluxconv*data[0,az,inc,:] # total intensity, star + dust
            self.Q = fluxconv*data[1,az,inc,:] # Stokes Q, star + dust
            self.U = fluxconv*data[2,az,inc,:] # Stokes U, star + dust
            self.V = fluxconv*data[3,az,inc,:] # Stokes V, star + dust
            self.star_I = fluxconv*data[4,az,inc,:] # total intensity from star only
            self.star_scat = fluxconv*data[5,az,inc,:] # scattered total intensity from star only
            self.thermal = fluxconv*data[6,az,inc,:] # thermal emission from disk
            self.dust_scat = fluxconv*data[7,az,inc,:] # scattered light from dust only
        
        return
    
    def merge_sed(self, data2, inc=slice(None), az=slice(None), fluxconv2=1.):
        data2 *= fluxconv2
        self.fluxconv2 = fluxconv2
        
        if data2.shape[0] == 5:
            self.I += data2[0,az,inc,:] # total intensity, all stars + dust
            self.star_I += data2[1,az,inc,:] # total intensity from star only
            self.star_scat += data2[2,az,inc,:] # scattered total intensity from star only
            self.thermal += data2[3,az,inc,:] # thermal emission from disk
            self.dust_scat += data2[4,az,inc,:] # scattered light from dust only

        else:
            self.I += data2[0,az,inc,:] # total intensity, all stars + dust
            self.Q += data2[1,az,inc,:] # Stokes Q, star + dust
            self.U += data2[2,az,inc,:] # Stokes U, star + dust
            self.V += data2[3,az,inc,:] # Stokes V, star + dust
            self.star_I += data2[4,az,inc,:] # total intensity from star only
            self.star_scat += data2[5,az,inc,:] # scattered total intensity from star only
            self.thermal += data2[6,az,inc,:] # thermal emission from disk
            self.dust_scat += data2[7,az,inc,:] # scattered light from dust only
        
        return
    
    def merge_dust_only(self, data2, inc=slice(None), az=slice(None), fluxconv2=1.):
        data2 *= fluxconv2
        self.fluxconv2 = fluxconv2
        
        if data2.shape[0] == 5:
            self.I += data2[3,az,inc,:] + data2[4,az,inc,:] # total intensity, star + multiple disks
            self.thermal += data2[3,az,inc,:] # thermal emission from disk
            self.dust_scat += data2[4,az,inc,:] # scattered light from dust only
        else:
            self.I += data2[6,az,inc,:] + data2[7,az,inc,:] # total intensity, star + multiple disks
            self.thermal += data2[6,az,inc,:] # thermal emission from disk
            self.dust_scat += data2[7,az,inc,:] # scattered light from dust only
        
        return


def mc_analyze(s_ident, path='.', partemp=True, ntemp_view=None, nburn=0, nthin=1, 
                nstop=None, add_fn=None, add_ind=0,
                make_maxlk=False, make_medlk=False, lam=1.6,
                range_dict=None, range_dict_tri=None, prior_dict_tri=None,
                xticks_dict_tri=None, contour_colors='k', plot=True, save=False):
    """
    Plot walker chains, histograms, and corner plot for a given sampler.
    
    Inputs:
        s_ident: str identifier of the sampler to display.
        path: str absolute path to the sampler files; include trailing /.
        partemp: True if sampler contains multiple temperatures.
        ntemp_view: int index of temperature to examine (if partemp == True).
        nburn: int number of (burn-in) steps to skip at start of sampler
            (usually this is 0 if you reset the chain after the burn-in phase).
        nthin: int sample thinning factor; e.g. nthin=10 will only use every 10th
            sample in the chain.
        nstop: iteration to truncate the chain at (ignore steps beyond nstop).
        add_fn: str filename after ".../logs/" for a second sampler to add to the
            first. Must have exactly the same setup.
        add_ind: int index of iteration at which to insert the second sampler.
            NOTE that this only works at the end of a sampler, for now.
        make_maxlk: True to make the MCFOST model for the max likelihood params.
        make_medlk: True to make the MCFOST model for the median likelihood params.
        lam: wavelength at which to make maxlk and medlk models [microns].
        range_dict:
        range_dict_tri:
        prior_dict_tri:
        xticks_dict_tri:
        contour_colors: str or list of str matplotlib color names to use as corner
            plot contour line colors.
    
    Outputs:
        Returns nothing, but creates a bunch of figures.
    
    """
    import gzip, corner, hickle
    from diskmc import make_mcfmod
    matplotlib.rc('text', usetex=False)
    
    print "\nThinning samples by %d" % nthin
    
    # Expand any paths.
    path = os.path.expanduser(path)
    
    # Get array of parameter values for each walker at each step in chain.
    # Has dimensions [nwalkers, nstep, pl.shape].
    # ind_ch = 4
    
    
    if partemp:
        try:
            sampler = hickle.load(path + s_ident + '_mcmc_full_sampler.hkl')
        except:
            print("FAILED to load sampler from log. Check s_ident and path and try again.")
            return
        ch = sampler['_chain']
        betas = sampler['_betas'] # temperature ladder
        ntemps = ch.shape[0]
        nwalkers = ch.shape[1]
        nstep = ch.shape[2]
        ndim = ch.shape[3]
        if ntemp_view is not None:
            ch = ch[ntemp_view]
            beta = betas[ntemp_view]
            lnprob = sampler['_lnprob'][ntemp_view]/beta # log probability; dim=[nwalkers, nsteps]
            print "Analyzing temperature=%d chain only, as specified." % ntemp_view
        else:
            ch = ch[0]
            beta = betas[0]
            lnprob = sampler['_lnprob']/beta
            print "Analyzing temperature=0 chain only, because ntemp_view=None."
    else:
        try:
            sampler = hickle.load(path + s_ident + '_mcmc_full_sampler.hkl')
        except:
            print("FAILED to load sampler from log. Check s_ident and path and try again.")
            return
        ch = sampler['_chain']
        beta = 1.
        lnprob = sampler['_lnprob']/beta
        nwalkers = ch.shape[0]
        nstep = ch.shape[1]
        ndim = ch.shape[2]
    
    # Autocorrelation function.
    try:
        # Compute it for each walker in each parameter. Do not thin this, generally.
        acf_all = np.array([[autocorr.function(ch[ii, :nstop, jj]) for ii in range(nwalkers)] for jj in range(ndim)])
        # Integrated autocorrelation time.
        act = np.array([[autocorr.integrated_time(ch[ii, :nstop, jj], c=1) for ii in range(nwalkers)] for jj in range(ndim)])
        act_means = np.mean(act, axis=1)
        print("\nIntegrated Autocorrelation Times (steps): " + str(np.round(act_means)))
        if np.any(act_means*50 >= nstep):
            print("WARNING: At least one autocorrelation time estimate is shorter than 50*nstep and should not be trusted.")
    except:
        acf_all = None
        print("Failed to calculate autocorrelation functions. Chain may be too short.")
    
    # # Optionally load a second sampler and add it to the main one.
    if add_fn is not None:
        sampler_add = hickle.load(path + add_fn + '.hkl')
        ch_add = sampler_add['_chain'][ntemp_view]
        beta_add = sampler_add['_betas'][ntemp_view]
        lnprob_add = sampler_add['_lnprob'][ntemp_view]/beta_add
        # Insert the sampler.
        ch[:, add_ind:, :] = ch_add
        lnprob[:, add_ind:] = lnprob_add
    
    
    # Get parameter labels from sampler.
    try:
        pkeys = sampler['pkeys_all']
    except:
        pkeys = np.array(ndim*['None'])
    
    
    # Make sure that mass fractions are properly normalized to sum to 1.0.
    try:
        # Find the mass fraction elements.
        wh_mf = np.array([ind for ind, key in enumerate(pkeys) if 'dust_pop' in key])
        # Copy the original ch to a new variable as a backup.
        ch_orig = ch.copy()
        # Select only the mass fraction steps from the chain.
        ch_mf = ch.copy()[:, :, wh_mf]
        # Normalize the mass fractions by their sum at each step.
        ch[:, :, wh_mf] /= np.reshape(np.sum(ch_mf[:, :], axis=2), (ch.shape[0], ch.shape[1], 1))
    except:
        pass
    
    # Discard first nburn samples as "burn-in" period and keep rest (and flatten).
    # Note that burn-in period is qualitative and may need tuning.
    # If nthin > 1, take only every nthin-th sample.
    samples = ch[::nthin, nburn:nstop, :].reshape((-1, ndim))
    
    if (nstop is not None) & (nburn is not None):
        nstep = nstop - nburn
        ch = ch[:, nburn:nstop, :]
        lnprob = lnprob[:, nburn:nstop]
    elif nburn is not None:
        nstep = ch.shape[1] - nburn
        ch = ch[:, nburn:, :]
        lnprob = lnprob[:, nburn:]
    
    
    # Max likelihood params values.
    ind_lk_max = np.where(lnprob==np.nanmax(lnprob))
    lk_max = np.e**lnprob.max()
    if len(ind_lk_max[0]) > 1:
        print "\nHEY! More than one sample matches the max likelihood! Only reporting the first here (full set is:", ind_lk_max, ")."
        ind_lk_max = ([ind_lk_max[0][0]], [ind_lk_max[1][0]])
    
    params_maxlk = {}
    params_medlk = {}
    
    print "\nMax Likelihood = %.3f  (chain %d, step %d) and param values:" % (lk_max, ind_lk_max[0][0], ind_lk_max[1][0])
    for kk, key in enumerate(pkeys[pkeys!='spl_br']):
        print key, '= %.3f' % ch[ind_lk_max[0], ind_lk_max[1], kk]
        params_maxlk[key] = ch[ind_lk_max[0], ind_lk_max[1], kk][0]
    pl_maxlk = np.array([params_maxlk[pk] for pk in pkeys])
    
    perc_list = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
    perc_3sigma_list = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [0.27, 50, 99.73], axis=0)))
        
    print "\nMedian (50%) +/- 34% confidence intervals (84%, 16% values) ; (99.7%, 0.3%):"
    for kk, key in enumerate(pkeys[pkeys!='spl_br']):
        print key, '= %.3f +/- %.3f/%.3f  (%.3f, %.3f) ; 99.7%% (%.3f, %.3f)' % (perc_list[kk][0], perc_list[kk][1], perc_list[kk][2], perc_list[kk][0]+perc_list[kk][1], perc_list[kk][0]-perc_list[kk][2], perc_3sigma_list[kk][0]+perc_3sigma_list[kk][1], perc_3sigma_list[kk][0]-perc_3sigma_list[kk][2])
        params_medlk[key] = perc_list[kk][0]
    pl_medlk = np.array([params_medlk[pk] for pk in pkeys])
    
    # Make MCFOST models from maxlk and meanlk params if desired.
    if make_maxlk:
        dir_newmod = path + '../diskmc_%s/' % s_ident
        print("\nMaking max-likelihood model with MCFOST...")
        make_mcfmod(pkeys, dict(zip(pkeys, pl_maxlk)), path + '../diskmc_init_%s.para' % s_ident,
                    dir_newmod, s_ident, fnstring='%s_mcmc_maxlk' % s_ident, lam=lam)
    if make_medlk:
        dir_newmod = path + '../diskmc_%s/' % s_ident
        print("\nMaking median-likelihood model with MCFOST...")
        make_mcfmod(pkeys, dict(zip(pkeys, pl_medlk)), path + '../diskmc_init_%s.para' % s_ident,
                    dir_newmod, s_ident, fnstring='%s_mcmc_medlk' % s_ident, lam=lam)
    
    # Variable labels for display.
    labels_dict = dict(aexp='$q$', amin=r'log $a_{min}$',
        debris_disk_vertical_profile_exponent=r'$\gamma$',
        disk_pa='$PA$', dust_mass=r'log $M_d$', #r'$M_{d}$ ($M_\odot$)',
        dust_pop_0_mass_fraction=r'$m_{Si}$', dust_pop_1_mass_fraction=r'$m_{aC}$',
        dust_pop_2_mass_fraction=r'$m_{H20}$', dust_vmax=r'$V_{max}$',
        gamma_exp=r'$\alpha_{in}$', inc='$i$', porosity='porosity',
        r_critical=r'$R_c$', r_in=r'$R_{in}$', r_out=r'$R_{out}$',
        scale_height=r'$H_0$', surface_density_exp=r'$\alpha_{out}$')
    
    # Trimmed ranges for walker display plots.
    if range_dict is None:
        range_dict = dict()
        for pk in pkeys:
            range_dict[pk] = (None, None)
    else:
        for pk in pkeys:
            if pk not in range_dict.keys():
                range_dict[pk] = (None, None)
    
    # Boundaries for the prior to display in the triangle plot.
    if prior_dict_tri is None:
        # Ideally, priors are stored in the sampler log- try to grab those.
        try:
            prior_dict_tri = sampler['logpargs'][1]
        except:
            prior_dict_tri = dict()
            for pk in pkeys:
                prior_dict_tri[pk] = (None, None)
    
    # Compute Gelman-Rubin statistic of chain convergence. https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
    # mm denotes the mth simulated chain, with m = 1, ..., M.
    # M is the total number of chains at given temp. N is the length of each chain.
    M = float(nwalkers)
    N = float(nstep)
    # dd is the degrees of freedom estimate of a t distribution.
    dd = M*N - 1
    # Std deviation of each chain for each parameter, dim [M, ndim].
    sigma_m = np.std(ch, axis=1)
    # Posterior mean of each chain for each parameter, dim [M, ndim].
    theta_m = np.mean(ch, axis=1)
    # Overall posterior mean for each parameter.
    theta = (1./M)*np.sum(theta_m, axis=0)
    # B is between-chains variance and W is within-chain variance (tends to
    # underestimate within-chain variance for early steps).
    B = N/(M - 1)*np.sum((theta_m - theta)**2, axis=0)
    W = (1./M)*np.sum(sigma_m**2, axis=0)
    # Pooled variance (overestimates variance until step-->inf).
    V = W*(N - 1)/N + B*(M + 1)/M/N
    # Potential scale reduction factor (PSRF) will be close to 1 if converged.
    R = np.sqrt(((dd + 3)/(dd + 1))*(V/W))
    print "\nPSRF:", R
    
    # Make histograms of chain parameter values for chunks of chain.
    try:
        chunk_size = nstep/10 # [steps]
        N_hists = ch.shape[1]/chunk_size # number of histogram chunks per axis
        nR = int(np.ceil(len(pkeys)/5.)) # number of rows in figure
        fig_hist, ax_array_hist = axMaker(nR*5, axRC=[nR,5], axSize=[2., 2.],
                                spEdge=[0.55, 0.6, 0.2, 0.1], spR=np.array((nR-1)*[[0.6]]),
                                spC=np.array(nR*[[0.1, 0.1, 0.1, 0.1]]), figNum=49)
        ax_array_hist = ax_array_hist.flatten()
        hist_colors = [matplotlib.cm.viridis(jj) for jj in np.linspace(0, 0.9, N_hists)]
        for jj, pkey in enumerate(pkeys):
            ax = ax_array_hist[jj]
            priors = prior_dict_tri[pkey]
            max_ind = chunk_size - 1
            hist_ind = 0
            while max_ind <= ch.shape[1]:
                # chunk = ch[:, jj*chunk_size:max_ind, jj]
                chunk = ch[:, max_ind, jj]
                ax.hist(chunk, bins=np.linspace(priors[0], priors[1], 20), color=hist_colors[hist_ind],
                        normed=False, histtype='bar', alpha=hist_ind*(chunk_size/float(ch.shape[1])) + (chunk_size/float(ch.shape[1])),
                        rwidth=0.9, label=str(max_ind + 1))
                max_ind += chunk_size
                hist_ind += 1
            ax.set_xlabel(pkey, fontsize=10)
            ax.set_ylim(0, nwalkers*1.01)
            ax.tick_params('both', direction='in', labelsize=14)
            if jj == 0:
                ax.legend(numpoints=1, fontsize=8, frameon=False)
            if jj > 0:
                ax.set_yticklabels([''])
        
        # Hide empty axes.
        if len(ax_array_hist) > len(pkeys):
            for jj in range(len(ax_array_hist) - len(pkeys)):
                ax_array_hist[len(pkeys) + jj].set_visible(False)
        
        plt.draw()
    except:
        print "\nFailed to draw chunk histograms."
    
    if plot:
        fontSize = 12
        
        fig = plt.figure(50)
        fig.clf()
        for aa, ff in enumerate(range(0, min(6, ndim), 1)):
            sbpl = "32%d" % (aa+1)
            ax = fig.add_subplot(int(sbpl))
            for ww in range(nwalkers):
                ax.plot(range(0,nstep), ch[ww,:,ff], 'k-', alpha=10./nwalkers)
            # if nburn > 0:
            #     ax.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
            # ax.set_ylabel(r'%.12s (d %d)' % (pkeys[ff], ff))
            ax.set_ylabel(r'%s (%d)' % (labels_dict[pkeys[ff]], ff), fontsize=fontSize+2)
            if aa%2==1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            ax.tick_params(labelsize=fontSize+1)
            ax.set_ylim(range_dict[pkeys[ff]][0], range_dict[pkeys[ff]][1])
            # ax.set_xlabel('step number')
            # ax.set_title(r'%s' % pkeys[ff])
            # pdb.set_trace()
        # Remove xtick labels from all but bottom panels.
        for ax in fig.get_axes()[:-2]:
            ax.set_xticklabels(['']*ax.get_xticklabels().__len__())
        fig.suptitle('Walkers, by step number', fontsize=fontSize+3)
        fig.subplots_adjust(0.16, 0.06, 0.84, 0.92, wspace=0.05, hspace=0.1)
        # plt.tight_layout()
        plt.draw()
            
        if ndim > 6:
            fig = plt.figure(51)
            fig.clf()
            for aa, ff in enumerate(range(6, min(12, ndim), 1)):
                sbpl = "32%d" % (aa+1)
                ax = fig.add_subplot(int(sbpl))
                for ww in range(nwalkers):
                    ax.plot(range(0,nstep), ch[ww,:,ff], 'k', alpha=10./nwalkers)
                # if nburn > 0:
                #     ax.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
                # ax.set_ylabel(r'%s (d %d)' % (pkeys[ff], ff))
                ax.set_ylabel(r'%s (%d)' % (labels_dict[pkeys[ff]], ff), fontsize=fontSize+2) 
                if aa%2==1:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                ax.tick_params(labelsize=fontSize+1)
                ax.set_ylim(range_dict[pkeys[ff]][0], range_dict[pkeys[ff]][1])
                # plt.xlabel('step number')
                # ax.title(r'%s' % pkeys[ff])
            for ax in fig.get_axes()[:-2]:
                ax.set_xticklabels(['']*ax.get_xticklabels().__len__())
            fig.suptitle('Walkers, by step number', fontsize=fontSize+3)
            fig.subplots_adjust(0.16, 0.06, 0.84, 0.92, wspace=0.05, hspace=0.1)
            # plt.tight_layout()
            plt.draw()
        
        if ndim > 12:
            fig = plt.figure(52)
            fig.clf()
            for aa, ff in enumerate(range(12, min(18, ndim), 1)):
                sbpl = "32%d" % (aa+1)
                ax = fig.add_subplot(int(sbpl))
                for ww in range(nwalkers):
                    ax.plot(range(0,nstep), ch[ww,:,ff], 'k', alpha=10./nwalkers)
                # if nburn > 0:
                #     ax.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
                # ax.set_ylabel(r'%s (d %d)' % (pkeys[ff], ff))
                ax.set_ylabel(r'%s (%d)' % (labels_dict[pkeys[ff]], ff), fontsize=fontSize+2) 
                if aa%2==1:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                ax.tick_params(labelsize=fontSize+1)
                ax.set_ylim(range_dict[pkeys[ff]][0], range_dict[pkeys[ff]][1])
                # plt.xlabel('step number')
                # ax.title(r'%s' % pkeys[ff])
            for ax in fig.get_axes()[:-2]:
                ax.set_xticklabels(['']*ax.get_xticklabels().__len__())
            fig.suptitle('Walkers, by step number', fontsize=fontSize+3)
            fig.subplots_adjust(0.16, 0.06, 0.84, 0.92, wspace=0.05, hspace=0.1)
            # plt.tight_layout()
            plt.draw()
        
        # Plot lnprob per step.
        fig4 = plt.figure(54)
        fig4.clf()
        ax1 = fig4.add_subplot(211)
        ax2 = fig4.add_subplot(212)
        fig4.subplots_adjust(0.18, 0.13, 0.97, 0.93, hspace=0.5)
        for wa in lnprob:
            ax1.plot(wa, alpha=0.25) #5./nwalkers)
        # if nburn > 0:
        #     ax1.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
        ax1.set_xlim(-nstep*0.01, nstep+(nstep*0.01))
        # ax1.set_xlabel('Step', fontsize=fontSize+1)
        ax1.set_ylabel('ln prob', fontsize=fontSize+1)
        ax1.set_title('ln(prob) at each step: Ntemp=%d' % ntemp_view, fontsize=fontSize+1)
        # for wa in lnprob:
        #     ax2.plot(np.exp(wa), alpha=0.15) #5./nwalkers)
        ax2.axhline(y=0, c='gray')
        for stp in range(nstep):
            ax2.plot(stp, np.where(np.isnan(lnprob[:,stp]))[0].size, 'r.')
            ax2.plot(stp, np.where(lnprob[:,stp]==-np.inf)[0].size, 'k.')
        # if nburn > 0:
        #     ax2.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
        ax2.set_ylim(-nwalkers*0.02, nwalkers+1)
        ax2.set_xlim(-nstep*0.01, nstep+(nstep*0.01))
        ax2.set_xlabel('Step', fontsize=fontSize+1)
        ax2.set_ylabel('# walkers', fontsize=fontSize+1)
        ax2.set_title(r'walkers w/ lnprob = -$\infty$ (black) or NaN (red): Ntemp=%d' % ntemp_view, fontsize=fontSize+1)
        # fig4.tight_layout()
        plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.setp(ax2.xaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.draw()
        
        # Plot cumulative distribution function.
        fig6 = plt.figure(55)
        fig6.clf()
        ax1 = fig6.add_subplot(111)
        # ax2 = fig4.add_subplot(212)
        fig6.subplots_adjust(0.18, 0.13, 0.97, 0.93)
        ax1.hist(-lnprob.flatten(), bins=100, normed=True, cumulative=True, histtype='step')
        # for wa in lnprob:
        #     ax1.plot(wa, alpha=0.25) #5./nwalkers)
        # if nburn > 0:
        #     ax1.axvspan(0, nburn, facecolor='lightgray', zorder=0, edgecolor='None')
        # ax1.set_xlim(-nstep*0.01, nstep+(nstep*0.01))
        # ax1.set_xlabel('Step', fontsize=fontSize+1)
        ax1.set_ylabel('CDF', fontsize=fontSize+1)
        ax1.set_xlabel('-ln(prob): Ntemp=%d' % ntemp_view, fontsize=fontSize+1)
        plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=fontSize+1)
        plt.draw()
        
        # # Plot mean autocorrelation lags.
        # fig5 = plt.figure(54, figsize=(5,4))
        # fig5.clf()
        # ax1 = fig5.add_subplot(111)
        # fig5.subplots_adjust(0.18, 0.17, 0.97, 0.97)
        # # ax2 = fig4.add_subplot(212)
        # for ii in range(ndim):
        #     ax1.plot(acf[ii][:], "-", label=labels_dict[pkeys[ii]])
        # # Draw line at 20% correlation.
        # ax1.axhline(0.2, c='r')
        # ax1.set_xlabel('tau = lag (steps)')
        # ax1.set_ylabel('Autocorrelation', labelpad=8)
        # # ax1.set_xlim(0,600)
        # ax1.set_ylim(-0.1,1.1)
        # plt.legend(fontsize=12, loc=1, labelspacing=0.1, handlelength=1.3, handletextpad=0.5)
        # plt.draw()
        
        # Plot autocorrelation functions for each walker and parameter.
        fig7 = plt.figure(56, figsize=(5,8))
        fig7.clf()
        if acf_all is not None:
            for aa, ff in enumerate(range(0, ndim, 1)):
                sbpl = "%d1%d" % (ndim, aa+1)
                ax = fig7.add_subplot(int(sbpl))
                ax.axhline(y=0, color='c', linestyle='--')
                for ac in acf_all[aa]:
                    ax.plot(ac, c='k', alpha=10./nwalkers)
                ax.plot(np.mean(acf_all[aa], axis=0), c='r', linewidth=1)
                ax.text(0.7, 0.8, pkeys[aa], color='r', fontsize=10, weight='bold', transform=ax.transAxes)
                # Hide all but the bottom axis labels.
                if aa != ndim - 1:
                    ax.set_yticklabels(['']*ax.get_yticklabels().__len__())
                    ax.set_xticklabels(['']*ax.get_xticklabels().__len__())
                ax.set_ylim(-1, 1)
                plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
                plt.setp(ax.xaxis.get_majorticklabels(), fontsize=12)
        fig7.suptitle("Normalized Autocorrelation Functions", fontsize=14)        
        plt.draw()
        
        
 # # TEMP!!!
 #        # Draw and make random models from chain.
 #        dir_randomdraw = '/Users/Tom/Research/data/gpi/Reduced/hd35841_160228_H_spec/models/morph_mcmc_43+44/random200/'
 #        make_mcfost_models(ch, pkeys, dir_randomdraw,
 #                           '../mcfost_param_35841_init_mmcmc_43+44_fullsed',
 #                           '43+44_mmcmc_random200',
 #                           random=True, Ndraw=200, lam=1.647,
 #                           dust_only=False, silent=True)
        
        
        answer = raw_input("\nContinue to make the triangle plot?\n [y/N]?: ").lower()
        if answer in ('y', 'yes'):
            print "Drawing triangle plot (may be slow)..."
        else:
            pdb.set_trace()
            return
        
        
        # Custom order for corner plot.
        pkeys_tri = np.array(['inc', 'r_critical', 'r_in', 'r_out', 'scale_height',
            'disk_pa', 'debris_disk_vertical_profile_exponent', 'surface_density_exp',
            'gamma_exp', 'dust_mass', 'amin', 'aexp', 'porosity', 'dust_vmax',
            'dust_pop_0_mass_fraction', 'dust_pop_1_mass_fraction',
            'dust_pop_2_mass_fraction'])
        # Tack any other keys onto the end of pkeys_tri.
        if np.any(~np.in1d(pkeys, pkeys_tri)):
            pkeys_tri = np.append(pkeys_tri, pkeys[~np.in1d(pkeys, pkeys_tri)])
        
        # Sorting indices for keys.
        tri_sort = []
        tri_incl = []
        for ii, key in enumerate(pkeys_tri):
            if key in pkeys:
                tri_sort.append(np.where(pkeys==key)[0][0])
                tri_incl.append(key)
        
        # New array of triangle plot pkeys.
        tri_incl = np.array(tri_incl)
        # Thin out samples for triangle plot by only plotting every nthin sample.
        samples_tri = samples[::nthin, tri_sort]
        # Convert disk_pa to on-sky PA (E of N) measured to front edge CCW from min scattering angle.
        if params_medlk['disk_pa'] < 270:
            samples_tri[:, np.where(tri_incl=='disk_pa')[0]] = samples_tri[:, np.where(tri_incl=='disk_pa')[0]] + 90.
        elif params_medlk['disk_pa'] >= 270:
            samples_tri[:, np.where(tri_incl=='disk_pa')[0]] = samples_tri[:, np.where(tri_incl=='disk_pa')[0]] - 270.
        
        labels_tri = [labels_dict[key] for key in tri_incl]
        if range_dict_tri is not None:
            range_tri = []
            for ii, key in enumerate(tri_incl):
                if key in range_dict_tri.keys():
                    range_tri.append(range_dict_tri[key])
                else:
                    # Default range is set to include every sample.
                    range_tri.append(1.0)
        else:
            range_tri = None
        
        fontsize_tri = 12
        # Make triangle figure and save if desired.
        # data_kwargs dict controls the datapoints parameters ('ms'==markersize).
        fig_tri = corner.corner(samples_tri, labels=labels_tri, quantiles=[0.16, 0.5, 0.84],
                                label_kwargs={"size":fontsize_tri},
                                show_titles=False, verbose=True, max_n_ticks=3,
                                plot_datapoints=True, plot_contours=True, fill_contours=True,
                                range=range_tri, plot_density=False,
                                data_kwargs={'color':'0.6', 'alpha':0.2, 'ms':1.},
                                contour_kwargs={'colors':contour_colors}) #, hist_kwargs=dict(align='left'))
        
        # Adjust figure/axes attributes after creation.
        # fig_tri.set_size_inches(18, fig_tri.get_size_inches()[1], forward=True)
        fig_tri.set_size_inches(10., 8., forward=True)
        # tri_axes[0] is top left corner; index increase from left to right across rows.
        tri_axes = fig_tri.get_axes()
        
        # Set x-axis ticks and ticklabels.
        # rr for row, cc for column of triangle plot.
        if xticks_dict_tri is not None:
            for cc in range(0, samples_tri.shape[1]):
                ax_list = [tri_axes[cc + jj*samples_tri.shape[1]] for jj in range(samples_tri.shape[1])]
                try:
                    for rr, ax in enumerate(ax_list):
                        # Ignore the empty upper-right corner of triangle plot.
                        if rr >= cc:
                            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks_dict_tri[tri_incl[cc]]))
                            if rr != cc:
                                ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks_dict_tri[tri_incl[:][rr]]))
                                # pdb.set_trace()
                            elif rr==cc:
                                ax.yaxis.set_visible(False)
                        if ax==ax_list[-1]:
                            ax.set_xticklabels(xticks_dict_tri[tri_incl[cc]])
                        if ((cc==0) and (rr > 0)):
                            ax.set_yticklabels(xticks_dict_tri[tri_incl[rr]])
                except:
                    continue
        
        for ax in tri_axes:
            ax.tick_params(axis='both', which='both', labelsize=fontsize_tri-1) #, direction='in') # change tick label size
            ax.tick_params(axis='x', which='both', pad=1) # reduce space between ticks and tick labels
            ax.tick_params(axis='y', which='both', pad=1) # reduce space between ticks and tick labels
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0) # make y tick labels horizontal
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70) # make x tick labels more vertical
            ax.xaxis.set_label_coords(0.5, -0.45) # move x labels downward
            ax.yaxis.set_label_coords(-0.4, 0.5) # move y labels leftward
        # Adjust margins of figure.
        plt.subplots_adjust(bottom=0.1, top=0.99, left=0.09, right=0.99)
        
        # Middle diagonal is tri_axes[ii*samples_tri.shape[1] + ii] for ii in range(0,samples_tri.shape[1]).
        tri_diag = [tri_axes[ii*samples_tri.shape[1] + ii] for ii in range(0,samples_tri.shape[1])]
        for aa, ax in enumerate(tri_diag):
            priors = prior_dict_tri[tri_incl[aa]]
            ax.axvline(x=priors[0], c='k', linestyle='-', linewidth=1.5)
            ax.axvline(x=priors[1], c='k', linestyle='-', linewidth=1.5)
        
        # Get only the axes in the left column and bottom row.
        tri_edges = [tri_axes[jj*samples_tri.shape[1]] for jj in range(samples_tri.shape[1])] + \
                    tri_axes[-samples_tri.shape[1]:]
        # Get every axis EXCEPT the left column and bottom row.
        tri_notedges = np.array(tri_axes)[np.isin(tri_axes, tri_edges, invert=True)]
        for ax in tri_notedges:
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
        # Hide x-axis on all but bottom row.
        for ax in tri_edges[:samples_tri.shape[1]-1]:
            ax.xaxis.set_visible(False)
        # Hide y-axis on all but left column.
        for ax in tri_edges[-(samples_tri.shape[1]-1):]:
            ax.yaxis.set_visible(False)
        
        plt.draw()
        
        # fig_tri = triangle.corner(samples, labels=pkeys)
        if save:
            fig_tri.savefig(path + '%s_corner.pdf' % s_ident, dpi=300, format='pdf')
            print("\nCorner plot saved as %s\n" % (path + s_ident + '_corner.pdf'))
    
    # To save corner plot manually:
    # figsave = plt.figure(##)
    # figsave.savefig(path + '##_mmcmc_corner.png', dpi=300, format='png')
    
    pdb.set_trace()
    
    return


def axMaker(axNum, axRC=None, axSize=[4.3,7], axDim=None, wdw=None, spR=None,
            spC=None, spEdge=None, hold=False, figNum=999):
    """
    Create a figure with appropriately arranged axes.
    
    Inputs:
        axNum= total number of axes.
        axRC= list of # of rows and columns for axes [rows, columns].
        axSize= size of individual axes in [inches].
        axDim= optional list of dimensions for axes as fraction of figure size
        [height, width].
        wdw= optional size of imshow plotting window in axes in [pixels].
        spR= row spacing; default=0.5 in.; list of lists, or single value to set all the same. [inches]
        spC= column spacing; default=0.5 in.; array of shape((#Rows, #Cols-1)), or single value to set all the same. [inches]
        spEdge= spacing between figure and axes edges; [1., 0.7, 0.5, 0.5] [inches]
        by default (left, bottom, right, top); array or list.
        hold= if True, will not clear figure before plotting.
        figNum= matplotlib figure number.
        
    Outputs:
        (fig, ax_array) tuple of figure created and array of axes inhabiting it.
    """
    
    # Set default arrangement of axes as 1 column with axNum rows.
    if axRC is None:
        axRC = [axNum, 1]
    # Set R and C to # number of rows and columns, respectively.
    R = axRC[0]
    C = axRC[1]
    
    ax_array = [] #np.zeros((R, C))
    pos_array = [] #np.zeros((R, C))
    
    axSizeY = axSize[0] # [inches]
    if wdw is not None:
        # Half dimensions of image window to show in each axis of stack.
        # Golden ratio is 1./1.618 = 0.618 (height/width).
        wdwSizeY = wdw[0] # [pix]
        wdwSizeX = wdw[1] # [pix]
        
        # Ratio (height/width) of image window for each axis in stack.
        axRatio = wdwSizeY/wdwSizeX
        ## Multiply half window dimensions by 2 and add 1 to get full window.
        #axRatio = (2*wdwSizeY + 1)/(2*wdwSizeX + 1)
        #axSizeX = axSizeY/axRatio # [inches]
        axSizeX = axSizeY/axRatio # [inches]
    else:
        axSizeX = axSize[1] # [inches]
        axRatio = axSizeY/axSizeX
    
    # If no row spacing specified, set all to 0.5 in.
    if spR is None:
        if R==1:
            spR = np.zeros((1, C))
        else:
            spR = np.zeros((R-1, C))
            spR[:] = 0.5
    elif type(spR) in [int, float]:
        spR_tmp = spR
        if R==1:
            spR = np.zeros((1, C))
        else:
            spR = np.zeros((R-1, C))
            spR[:] = spR_tmp
    
    # If no column spacing specified, set all to 0.5 in.
    if spC is None:
        if C==1:
            spC = np.zeros((R, 1))
        else:
            spC = np.zeros((R, C-1))
            spC[:] = 0.5
    elif type(spC) in [int, float]:
        spC_tmp = spC
        if C==1:
            spC = np.zeros((R, 1))
        else:
            spC = np.zeros((R, C-1))
            spC[:] = spC_tmp
    
    # Set spacing between axes edges and figure edges.
    if spEdge is None:
        spEdge = np.array([1., 0.7, 0.5, 0.5]) # [inches]
    spLeft = spEdge[0]
    spBot = spEdge[1]
    spRight = spEdge[2]
    spTop = spEdge[3]
    
    plotSizeY = axSizeY*R + np.sum(spR, axis=0)[0] + spTop + spBot # [inches]
    plotSizeX = axSizeX*C + np.max(np.sum(spC, axis=1)) + spLeft + spRight # [inches]
    
    # Axes height and width (individual) as fraction of fig size.
    he_0 = axSizeY/plotSizeY
    wi_0 = axSizeX/plotSizeX
    
    # Create fractional spacing arrays.
    spR_fr = spR/plotSizeY
    spC_fr = spC/plotSizeX
    spLeft_fr = spLeft/plotSizeX
    spBot_fr = spBot/plotSizeY
    spTop_fr = spTop/plotSizeY
    spRight_fr = spRight/plotSizeX
    
    # Create figure with size [plotSizeX, plotSizeY] in [inches].
    fig1 = plt.figure(figNum, figsize=[plotSizeX, plotSizeY])
    
    if hold is False:
        plt.clf()
    
    # Calculate position of axes based on row and column.
    for ro in range(R):
        he = he_0
        wi = wi_0
        
        for co in range(C):
            # Left position.
            le = spLeft_fr + co*wi_0 + spC_fr[ro, :co].sum()
            # Bottom position.
            bo = spBot_fr + (R - ro - 1)*he_0 + spR_fr[ro:].sum(axis=0)[0]
            # Define new position for subplot axes.
            # [left, bottom, width (right-left), height (top-bottom)].
            pos = [le, bo, wi, he]
            
            pos_array.append(pos)
            
            # Add axes to figure.
            ax = fig1.add_axes([le, bo, wi, he])
            ax_array.append(ax)
            
            # pdb.set_trace()
    
    ax_array = np.array(ax_array).reshape(R,C)
    pos_array = np.array(pos_array).reshape(R,C,4)
    
 #   ## [Left, bottom, right, top position] of ax1 as fraction of figure window.
 #   #curPos1 = np.reshape(ax1.get_position().get_points(), 4)
 #   
 #   ## Left, bottom, width (right-left), height (top-bottom).
 #   #le = curPos1[0]
 #   #bo = curPos1[1]
 #   #wi = curPos1[2] - le
 #   #he = curPos1[3] - bo
    
    return fig1, ax_array


def plot_sed(path_mod, path_mod2=None, scale_th=1, scale_sc=1,
             draw=None, lam_def=False, plot_residuals=True, plot_inset=False,
             save=None, path_model=None, figNum=10):
    """
    
    Inputs:
        
        scale_th: multiplicative scale factor for dust thermal flux.
        scale_sc: multiplicative scale factor for dust scattering flux.
        lam_def: if True, use a 
        draw:
    """
    
    import glob
    from astropy.io import fits
    # from scipy.io import readsav
    from scipy import interpolate
    from scipy.stats import binned_statistic
    from astropy import constants
    from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
    
    # if s_ident is not None:
    #     if mctype.lower()=='dust':
    #         path_fn = glob.glob(path_model + 'models/dust_mcmc_%s/%s_dmcmc_%slk*' % (s_ident, s_ident, which))[0]
    #     elif mctype.lower()=='sed':
    #         path_fn = glob.glob(path_model + 'models/sed_mcmc_%s/%s_smcmc_%slk*' % (s_ident, s_ident, which))[0]
    #     elif mctype.lower()=='morph':
    #         path_fn = glob.glob(path_model + 'models/morph_mcmc_%s/%s_mmcmc_%slk*' % (s_ident, s_ident, which))[0]
    #     elif mctype.lower()=='all':
    #         path_fn = glob.glob(path_model + 'models/mcmc_%s/%s_mcmc_%slk*' % (s_ident, s_ident, which))[0]
    #     model_sed_hdu = [fits.open(path_fn + '/data_th/sed_rt.fits.gz')]
    # elif draw is not None:
    #     fn_list = glob.glob(path_model + 'models/' + draw + '/*')
    #     model_sed_hdu = [fits.open(fn + '/data_th/sed_rt.fits.gz') for fn in fn_list]
    #     # model_sed_hdu = model_sed_hdulist[0] # just a placeholder
    #     path_fn_mean = glob.glob(path_model + 'models/morph_mcmc_43+44/43+44_mmcmc_medlk_aexp*/')[0]
    #     # path_fn_mean = glob.glob(path_model + 'models/morph_mcmc_24/24_mmcmc_meanlk_aexp*/')[0]
    #     meanlk_sed_hdu = fits.open(path_fn_mean + 'data_th/sed_rt.fits.gz')
    # else:
    #     # model_sed_hdu = fits.open(path + "models/sed_mcmc_13/data_th/sed_rt.fits.gz")
    #     model_sed_hdu = [fits.open(path_fn + 'sed_rt.fits.gz')]
    
     # Load model from file.
    model_sed_hdu = [fits.open(glob.glob(path_mod)[0])]
    
    if path_mod2 is not None:
        model_sed2_hdu = fits.open(path_mod2)
    
    try:
        N_bins_mcfost = model_sed_hdu[0][0].data.shape[3]
    except:
        N_bins_mcfost = 500
    
    
    # Start plotting data.
    ax_left = 0.122
    ax_width = 0.855
    majTickLength = 6
    minTickLength = 3
    fontSize = 14
    alpha_model = 1.
    
    if plot_residuals:
        # fig = plt.figure(figNum, figsize=(8,5.6))
        fig = plt.figure(figNum, figsize=(7.14,5.))
        plt.clf()
        ax0 = plt.axes([ax_left, 0.25, ax_width, 0.74]) # leave room for bottom axis
    else:
        fig = plt.figure(figNum, figsize=(8,5.))
        plt.clf()
        ax0 = plt.axes([ax_left, 0.11, ax_width, 0.88])
    
    # Draw inset.
    if plot_inset:
        if plot_residuals:
            ax1 = plt.axes([0.21, 0.32, 0.24, 0.34])
        else:
            ax1 = plt.axes([0.21, 0.25, 0.24, 0.34])
    
    # For explicitly defined SED wavelengths.
    if lam_def:
        # model_wave = np.sort(np.concatenate((wave_irs_binned, wave_mips))) # [microns]
        model_wave = np.sort(np.concatenate((wave_irs_binned, wave_mips, wave_sons))) # [microns]
    # For MCFOST logarithmically-spaced SED wavelengths.
    else:
        # model_wave = np.logspace(np.log10(0.35), np.log10(250.), 40) # [microns]
        # model_wave = np.logspace(np.log10(0.1), np.log10(3000.), N_bins_mcfost) # [microns]
        model_wave = model_sed_hdu[0][1].data
    
    model_nu = constants.c.value/(1e-6*model_wave) # [Hz]
    conv_WtoJy = 1.e26/model_nu
    
    if path_mod2 is not None:
        model_sed2 = sedMcfost(model_sed2_hdu, inc=0, az=0, fluxconv=conv_WtoJy)
    
    if draw is not None:
        meanlk_sed = sedMcfost(meanlk_sed_hdu, inc=0, az=0, fluxconv=conv_WtoJy)
        meanlk_sed.wave = model_wave
        # meanlk_combo_sed = meanlk_sed.I + model_sed2.I
        meanlk_combo_sed = sedMcfost(meanlk_sed_hdu, inc=0, az=0, fluxconv=conv_WtoJy)
        meanlk_combo_sed.merge_dust_only(conv_WtoJy*model_sed2_hdu[0].data, inc=0, az=0, fluxconv2=1.)
        alpha_model = 2./len(model_sed_hdu)
    
    # Handle the model depending on case.
    for mm in range(len(model_sed_hdu)):
        # model_sed = model_sed_hdu[mm][0].data.copy() # [W m^-2]
        model_sed = sedMcfost(model_sed_hdu[mm], inc=0, az=0, fluxconv=conv_WtoJy) # [W m^-2]
        model_sed.wave = model_wave
        # Dimensions: Stokes I, Q, U, V, star only, star scattered light, thermal, scattered light from dust thermal emission
        
        if path_mod2 is not None:
            model_sed_dust_only = model_sed.thermal.copy()+model_sed.dust_scat.copy()
            # model_sed2 = sedMcfost(model_sed2_hdu, inc=0, az=0, fluxconv=conv_WtoJy)
            model_sed.merge_dust_only(conv_WtoJy*model_sed2_hdu[0].data, inc=0, az=0, fluxconv2=1.)
        
        # # Interpolated models.
        # f_model = interpolate.interp1d(model_wave, model_spec_total)
        # model_irs = f_model(wave_irs_binned)
        # model_mips = f_model(wave_mips)
        # model_combo = f_model(wave_combo)
        model_combo = model_sed.I
        
        # try:
        #     f_phot = interpolate.interp1d(model_wave, model_sed.star_I + model_sed.star_scat)
        #     model_phot = f_phot(wave_irs_binned)
        #     model_phot_star = f_phot(photosph_wave)
        #     spec_irs_modelphotsub = spec_irs_binned - model_phot
        # except:
        #     model_phot_star = np.nan*np.ones(photosph_wave.shape)
        #     spec_irs_modelphotsub = np.nan*np.ones(wave_irs_binned.shape)
        
    
        # irs_chi2 = np.sum(((spec_irs_binned - model_irs)/spec_err_irs_binned)**2)
        # irs_chi2_red = irs_chi2/(model_irs.shape)
        # mips_chi2 = np.sum(((spec_mips - model_mips)/spec_err_mips)**2)
        # mips_chi2_red = mips_chi2/(model_mips.shape)
        # print "IRS chi2_red: %.2f" % irs_chi2_red
        # print "MIPS chi2_red: %.2f" % mips_chi2_red
        # chi2 = np.sum(((spec_combo - model_combo)/spec_err_combo)**2)
        # chi2_red = chi2/(spec_combo.shape[0] - 4)
        # print "Total chi2: %.3f" % chi2
        # print "Total chi2_red: %.2f" % chi2_red
        # print "Weighted chi2_reds: IRS = %.2f , MIPS = %.2f" % (irs_chi2_red/
    
        ax0.plot(model_wave, model_sed.I, 'k-', label='Total', alpha=alpha_model, linewidth=1., zorder=900)
        ax0.plot(model_wave, model_sed.star_I + model_sed.star_scat, 'k--', label='Star', alpha=1./len(model_sed_hdu))
        ax0.plot(model_wave, scale_th*model_sed.thermal + scale_sc*model_sed.dust_scat, c='#CD00ED', label='Dust', alpha=alpha_model, linewidth=1., zorder=901)
        
        if plot_inset:
            # ax1.errorbar(wave_irs_binned, spec_photsub_irs_binned, yerr=spec_err_irs_binned, fmt='.', c='0.5', elinewidth=1., markersize=4, alpha=0.8, label=r'IRS$-$phot')
            ax1.errorbar(wave_irs_binned, spec_irs_modelphotsub, yerr=spec_err_irs_binned, fmt='.', c='0.5', elinewidth=1.5, markersize=6, alpha=1.0)
            ax1.plot(model_wave, model_sed.I, 'k-', markersize=2, alpha=alpha_model, linewidth=1., zorder=900)
            ax1.plot(model_wave, model_sed.star_I + model_sed.star_scat, 'k--', alpha=1./len(model_sed_hdu))
            ax1.plot(model_wave, model_sed.thermal + model_sed.dust_scat, c='#CD00ED', alpha=alpha_model, zorder=901)
    
    if (path_mod2 is not None) and draw:
        # Add blank dummy lines for easy legend hack.
        ax0.plot(model_wave, np.nan*np.ones(model_wave.shape), '#CD00ED', label='2-Comp Disk', linewidth=1.5)
        # ax0.plot(model_wave, conv_WtoJy*(model_sed_hdu[0][0].data[6,0,0]+model_sed_hdu[0][0].data[7,0,0]), 'b:')
        # ax0.plot(model_wave, conv_WtoJy*(model_sed2[6,0,0]+model_sed2[7,0,0]), 'c:')
        ax0.plot(model_wave, meanlk_sed.thermal+meanlk_sed.dust_scat, 'C3-.', label='Outer Disk', linewidth=1.5)
        ax0.plot(model_wave, model_sed2.thermal+model_sed2.dust_scat, 'c:', label='Inner Disk', linewidth=2.)
        # Add blank dummy lines for easy legend hack.
        ax0.plot(model_wave, np.nan*np.ones(model_wave.shape), 'k', label='Disk+Star', linewidth=1.5)
        ax0.plot(model_wave, meanlk_combo_sed.I, 'k-', alpha=1.0, linewidth=1., zorder=901)
    elif (path_mod2 is not None) and not draw:
        ax0.plot(model_wave, np.nan*np.ones(model_wave.shape), '#CD00ED', label='2-Comp Disk', linewidth=1.5)
        ax0.plot(model_wave, model_sed_dust_only, 'C3:', label='Outer Disk', linewidth=1.5)
        ax0.plot(model_wave, model_sed2.thermal+model_sed2.dust_scat, 'c:', label='Inner Disk', linewidth=2.)
        ax0.plot(model_wave, np.nan*np.ones(model_wave.shape), 'k', label='Disk+Star', linewidth=1.5)
        
    
    # Finish cleaning up plot format.
    ax0.set_yscale('log', nonposy='clip')
    ax0.set_xscale('log', nonposy='clip')
    ax0.set_ylim(0.5*np.nanmin(model_sed.I), None)
    ax0.set_ylim(1e-4, 2.)
    # plt.xlim(0.31, 280) # best for testing
    ax0.set_xlim(0.4, 1000) # best for display
    
    # # Define tick positions and labels.
    # majorYticklabels = np.arange(0.001, 0.9, 0.4)
    # majorLocY = majorYticklabels/pscale + 140
    # majorLocatorY = matplotlib.ticker.FixedLocator(majorLocY)
    # ax0.xaxis.set_ticklabels(['', '', '1', '10', '100', ''])
    ax0.xaxis.set_ticklabels([])
    ax0.tick_params(axis='x', which='both', direction='in')
    # majorLocX = majorXticklabels/pscale + 140
    # majorLocatorX = matplotlib.ticker.FixedLocator(majorLocX)
    
    if not plot_residuals:
        ax0.set_xlim(ax0.get_xlim()) # best for display
        ax0.set_xscale('log', nonposy='clip')
        ax0.set_xlabel("Wavelength (microns)", labelpad=2)
    
    ax0.set_ylabel("Flux (Jy)", labelpad=8)
    # plt.title("Spectral Energy Distribution")
    ax0.legend(loc=1, fontsize=fontSize-2, handlelength=1.3,
               handletextpad=0.3, labelspacing=0.3,
               ncol=1, columnspacing=1., borderpad=0.2)
    # plt.legend(loc=(0.8, 0.08), fontsize=12, handletextpad=0.2)
    
    # ---- PLOT INSET ----
    if plot_inset:
        # ax1.set_yscale('log', nonposy='clip')
        ax1.set_xscale('log', nonposy='clip')
        ax1.set_ylim(0., 8e-2)
        ax1.set_xlim(15, 38)
        
        minorLocatorX = matplotlib.ticker.FixedLocator([15, 20, 25, 30, 35])
        ax1.xaxis.set_minor_locator(minorLocatorX)
        # ax1.xaxis.set_minorticklabels(['20', '30', '40'])
        ax1.tick_params(axis='both', which='both', labelsize=fontSize)
        ax1.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
        # ax1.tick_params(axis='y', which='major', labelsize=12)
        
    minor_locator = AutoMinorLocator(3)
    
    # if len(model_combo) > 33:
    #     model_combo = model_combo[:-2]
    
    if not lam_def:
        try:
            f_model = interpolate.interp1d(model_wave, model_combo) #model_spec_total)
            # model_irs = f_model(wave_irs_binned)
            # model_mips = f_model(wave_mips)
            # model_combo = f_model(wave_combo)
            wave_combo = np.concatenate((photosph_wave, wise_wave, wave_combo))
            bandpass_combo = np.concatenate((bandpass_photosph, bandpass_wise, bandpass_combo))
            spec_combo = np.concatenate((photosph, wise, spec_combo))
            spec_err_combo = np.concatenate((photosph_err, wise_err, spec_err_combo))
            model_combo = f_model(wave_combo)
        except:
            # wave_combo = np.concatenate((photosph_wave, wise_wave, wave_combo))
            # bandpass_combo = np.concatenate((bandpass_photosph, bandpass_wise, bandpass_combo))
            model_combo = None #np.ones(spec_combo.shape)*np.nan
            # spec_combo = np.concatenate((photosph, wise, spec_combo))
            # spec_err_combo = np.concatenate((photosph_err, wise_err, spec_err_combo))
    
    if plot_residuals:
        # Plot residuals.
        ax2 = plt.axes([ax_left, 0.11, ax_width, 0.14])
        ax2.axhline(y=0., c='0.5', alpha=0.5, linestyle='--')
        # ax2.errorbar(wave_combo, spec_combo - model_combo, yerr=spec_err_combo, c='k',
        #              fmt='.', markersize=4)
        ax2.errorbar(wave_combo, (spec_combo - model_combo)/spec_err_combo, yerr=None, c='k',
                     fmt='.', markersize=4)
        # ax2.set_ylim(-0.08, 0.055)
        ax2.set_ylim(-5.2, 5.2)
        ax2.set_xlim(ax0.get_xlim()) # best for display
        ax2.set_xscale('log', nonposy='clip')
        ax2.set_yscale('linear')
        ax2.set_xlabel(r"Wavelength ($\mu$m)", labelpad=2)
        ax2.set_ylabel("Residuals", labelpad=18)
        ax2.yaxis.set_minor_locator(minor_locator)
        ax2.yaxis.set_ticks([-3, 0, 3])
        ax2.yaxis.set_ticklabels([r'-3$\sigma$', 0, r'3$\sigma$'])
    
    # Set axis tick lengths, widths, and colors
    for ax_i in fig.get_axes():
        ax_i.tick_params(axis='y', which='major', length=majTickLength, width=1, color='0.6', direction='in')
        ax_i.tick_params(axis='y', which='minor', length=minTickLength, width=1, color='0.6', direction='in')
        ax_i.tick_params(axis='x', which='major', length=majTickLength, width=1, color='0.6', direction='in')
        ax_i.tick_params(axis='x', which='minor', length=minTickLength, width=1, color='0.6', direction='in')
        plt.setp(ax_i.get_xticklabels(), fontsize=fontSize)
        #plt.setp(ax.get_xminorticklabels(), fontsize=fontSize)
        plt.setp(ax_i.get_yticklabels(), fontsize=fontSize)
        #plt.setp(ax.get_yminorticklabels(), fontsize=fontSize)
        # Change spine color to gray.
        for s in ax_i.spines.values():
            s.set_ec('0.6')
    
    plt.draw()
    
    # # Cushing G goodness-of-fit statistic (Cushing+ 2008).
    # # Weight each point by its spectral width in microns (Cushing+ 2008).
    # weights = (bandpass_combo/wave_combo)/(np.sum(bandpass_combo/wave_combo))
    # # # Weight each point by its log space spectral width (Naud+ 2014, Gagne+ 2015).
    # # weights = (ln_delta_wave_combo/wave_combo)/np.sum(ln_delta_wave_combo/wave_combo)
    # C = np.sum(weights*spec_combo*model_combo/spec_err_combo**2)/np.sum(weights*model_combo**2/spec_err_combo**2)
    # G = np.sum(weights*((spec_combo - C*model_combo)/spec_err_combo)**2)
    # print "Cushing G: %.3f" % G
    # print "exp(-0.5*G): %.3e" % np.exp(-0.5*G)
    # # print "-ln(1+G): %.3e" % (-np.log(1+G))
    # 
    # # Photosphere fit.
    # weights_phot = (bandpass_photosph/photosph_wave)/(np.sum(bandpass_photosph/photosph_wave))
    # C_phot = np.sum(weights_phot*photosph*model_phot_star/photosph_err**2)/np.sum(weights_phot*model_phot_star**2/photosph_err**2)
    # G_phot = np.sum(weights_phot*((photosph - C_phot*model_phot_star)/photosph_err)**2)
    # 
    # print "Photosphere Cushing G: %.3f" % G_phot
    # print "Photosphere exp(-0.5*G): %.3e" % np.exp(-0.5*G_phot)
    
    
    if save is not None:
        if s_ident is not None:
            fig.savefig(path_fn + '/../plots/%s.pdf' % save, dpi=300, transparent=True, format='pdf')
            # fig.savefig(d_mcmc_%s/plots/%s' % (s_ident, save)) + '.pdf', dpi=300, transparent=True, format='pdf')
        else:
            fig.savefig(os.path.expanduser('~/Research/hd35841/hd35841_sed_%s' % save) + '.pdf', dpi=300, transparent=True, format='pdf')
    
    # pdb.set_trace()
    
    return

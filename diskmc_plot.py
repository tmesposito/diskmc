#!/usr/bin/env python

import os
import pdb
import numpy as np
import matplotlib, matplotlib.pyplot as plt


def mc_analyze(s_ident, path='.', nburn=0, partemp=True, ntemp_view=None, nthin=1, 
                nstop=None, add_fn=None, add_ind=0,
                make_maxlk=False, make_medlk=False, lam=1.6,
                range_dict=None, range_dict_tri=None, prior_dict_tri=None, xticks_dict_tri=None,
                plot=True, save=False):
    """
    !! WARNING: PARTIALLY FUNCTIONAL !!
    
    Plot walker chains, histograms, and corner plot for a given sampler.
    
    Inputs:
        s_ident: str identifier of the sampler to display.
        path: str absolute path to the sampler files; include trailing /.
        nburn: int number of (burn-in) steps to skip at start of sampler
            (usually this is 0 if you reset the chain after the burn-in phase).
        nthin: int sample thinning factor; e.g. nthin=10 will only use every 10th
            sample in the chain.
        ntemp_view: int index of temperature to examine (if partemp == True).
        nstop: iteration to truncate the chain at (ignore steps beyond nstop).
        add_fn: str filename after ".../logs/" for a second sampler to add to the
            first. Must have exactly the same setup.
        add_ind: int index of iteration at which to insert the second sampler.
            NOTE that this only works at the end of a sampler, for now.
        make_maxlk: True to make the MCFOST model for the max likelihood params.
        make_medlk: True to make the MCFOST model for the median likelihood params.
        lam: wavelength at which to make maxlk and medlk models [microns].
    
    """
    import gzip, corner, acor, hickle
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
            # Autocorrelation function.
            # Get autocorrelation for each dimension as a function of step, averaged over all walkers.
            # Do not thin this, generally.
            acf = [acor.function(np.mean(ch[::nthin,nburn:nstop,ii], axis=0)) for ii in range(ndim)]
            print "Analyzing temperature=%d chain only, as specified." % ntemp_view
        else:
            ch = ch[0]
            beta = betas[0]
            lnprob = sampler['_lnprob']/beta
            acf = [acor.function(np.mean(ch[::nthin,nburn:nstop,ii], axis=0)) for ii in range(ndim)]
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
        acf = [acor.function(np.mean(ch[::nthin,nburn:,ii], axis=0)) for ii in range(ndim)]
    
    
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
        #print "mu0 = %.2f +/- %.2f/%2f \nmu1 = %.2f +/- %.2f/%2f \nsig0 = %.2f +/- %.2f/%2f \nsig1 = %.2f +/- %.2f/%2f" % (mu0_mcmc[0], mu0_mcmc[1], mu0_mcmc[2], mu1_mcmc[0], mu1_mcmc[1], mu1_mcmc[2], sig0_mcmc[0], sig0_mcmc[1], sig0_mcmc[2], sig1_mcmc[0], sig1_mcmc[1], sig1_mcmc[2])
        print key, '= %.3f' % ch[ind_lk_max[0], ind_lk_max[1], kk]
        params_maxlk[key] = ch[ind_lk_max[0], ind_lk_max[1], kk][0]
    pl_maxlk = np.array([params_maxlk[pk] for pk in pkeys])
    
    perc_list = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
    perc_3sigma_list = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [0.27, 50, 99.73], axis=0)))
        
    print "\nMedian (50%) and 34% confidence intervals:"
    for kk, key in enumerate(pkeys[pkeys!='spl_br']):
        #print "mu0 = %.2f +/- %.2f/%2f \nmu1 = %.2f +/- %.2f/%2f \nsig0 = %.2f +/- %.2f/%2f \nsig1 = %.2f +/- %.2f/%2f" % (mu0_mcmc[0], mu0_mcmc[1], mu0_mcmc[2], mu1_mcmc[0], mu1_mcmc[1], mu1_mcmc[2], sig0_mcmc[0], sig0_mcmc[1], sig0_mcmc[2], sig1_mcmc[0], sig1_mcmc[1], sig1_mcmc[2])
        print key, '= %.3f +/- %.3f/%.3f  (%.3f, %.3f) ; 99.7%% (%.3f, %.3f)' % (perc_list[kk][0], perc_list[kk][1], perc_list[kk][2], perc_list[kk][0]+perc_list[kk][1], perc_list[kk][0]-perc_list[kk][2], perc_3sigma_list[kk][0]+perc_3sigma_list[kk][1], perc_3sigma_list[kk][0]-perc_3sigma_list[kk][2])
        params_medlk[key] = perc_list[kk][0]
    pl_medlk = np.array([params_medlk[pk] for pk in pkeys])
    
    # Make MCFOST models from maxlk and meanlk params if desired.
    if make_maxlk:
        pl_maxlk[pkeys=='amin'] = 10**pl_maxlk[pkeys=='amin']
        dir_newmod = path + '../diskmc_%s/' % s_ident
        print("\nMaking max-likelihood model with MCFOST...")
        make_mcfmod(pkeys, dict(zip(pkeys, pl_maxlk)), path + '../diskmc_init_%s.para' % s_ident,
                    dir_newmod, s_ident, fnstring='%s_mcmc_maxlk' % s_ident, lam=lam)
        # make_mcfost_models([pl_maxlk], pkeys, dir_newmod,
        #         '../diskmc_init_%s' % s_ident,
        #         '%s_mcmc_maxlk' % s_ident,
        #         random=False, lam=lam, dust_only=False, silent=True)
    if make_medlk:
        pl_medlk[pkeys=='amin'] = 10**pl_medlk[pkeys=='amin']
        dir_newmod = path + '../diskmc_%s/' % s_ident
        print("\nMaking median-likelihood model with MCFOST...")
        make_mcfmod(pkeys, dict(zip(pkeys, pl_medlk)), path + '../diskmc_init_%s.para' % s_ident,
                    dir_newmod, s_ident, fnstring='%s_mcmc_medlk' % s_ident, lam=lam)
        # make_mcfost_models([pl_medlk], pkeys, dir_newmod,
        #         '../diskmc_init_%s' % s_ident,
        #         '%s_mcmc_medlk' % s_ident,
        #         random=False, lam=lam, dust_only=False, silent=True)
    
    # Variable labels for display.
    labels_dict = dict(aexp='$q$', amin=r'log $a_{min}$',
        debris_disk_vertical_profile_exponent=r'$\gamma$',
        disk_pa='$PA$', dust_mass=r'log $M_d$', #r'$M_{d}$ ($M_\odot$)',
        dust_pop_0_mass_fraction=r'$m_{Si}$', dust_pop_1_mass_fraction=r'$m_{aC}$',
        dust_pop_2_mass_fraction=r'$m_{H20}$', gamma_exp=r'$\alpha_{in}$',
        inc='$i$', porosity='porosity',
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
        fig_hist, ax_array_hist = axMaker(15, axRC=[3,5], axSize=[2., 2.],
                                spEdge=[0.55, 0.6, 0.2, 0.1], spR=np.array([[0.6], [0.6]]),
                                spC=np.array(3*[[0.1, 0.1, 0.1, 0.1]]), figNum=49)
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
        fig4 = plt.figure(53)
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
        
        # Plot autocorrelation function.
        fig5 = plt.figure(54, figsize=(5,4))
        fig5.clf()
        ax1 = fig5.add_subplot(111)
        fig5.subplots_adjust(0.18, 0.17, 0.97, 0.97)
        # ax2 = fig4.add_subplot(212)
        for ii in range(ndim):
            ax1.plot(acf[ii][:], "-", label=labels_dict[pkeys[ii]])
        # Draw line at 20% correlation.
        ax1.axhline(0.2, c='r')
        ax1.set_xlabel('tau = lag (steps)')
        ax1.set_ylabel('Autocorrelation', labelpad=8)
        # ax1.set_title('ln(prob) for each walker at each step')
        # ax1.set_xlim(0,600)
        ax1.set_ylim(-0.1,1.1)
        plt.legend(fontsize=12, loc=1, labelspacing=0.1, handlelength=1.3, handletextpad=0.5)
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
            'gamma_exp', 'dust_mass', 'amin', 'aexp', 'porosity',
            'dust_pop_0_mass_fraction', 'dust_pop_1_mass_fraction',
            'dust_pop_2_mass_fraction'])
        
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
            range_tri = [range_dict_tri[key] for key in tri_incl]
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
                                data_kwargs={'color':'0.6', 'alpha':0.2, 'ms':1.}) #, hist_kwargs=dict(align='left'))
        
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

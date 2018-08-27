#!/usr/bin/env python3

__author__ = 'Tom Esposito'
__copyright__ = 'Copyright 2018, Tom Esposito'
__credits__ = ['Tom Esposito']
__license__ = 'GNU General Public License v3'
__version__ = '0.0.1'
__maintainer__ = 'Tom Esposito'
__email__ = 'espos13@gmail.com'
__status__ = 'Development'

import os, sys, argparse, warnings
import subprocess 
from shutil import rmtree
import pdb
import time, datetime as dt
import glob
import pickle, hickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants
from scipy.ndimage import zoom

import acor
from emcee import PTSampler, EnsembleSampler
#from emcee.utils import MPIPool
#from mpi4py import MPI

from diskmc_tools import get_ann_stdmap, make_radii, get_radial_stokes
#import pyklip.instruments.GPI as GPI
#from pyklip import parallelized, klip, fm
#from pyklip.fmlib import diskfm
from mcfost.paramfiles import Paramfile

# !!!WARNING!!! Ignoring some annoying warnings about FITS files.
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Turn off interactive plotting.
plt.ioff() 


class MCData:
    """
    Class that contains data and associated info.
    """
    
    def __init__(self, data, stars, uncerts, bin_facts, mask_params, s_ident):
        """
        Initialization code for ModObj.
        
        Inputs:
            
        
        """
        
        self.data = data
        self.stars = stars
        self.uncerts = uncerts
        self.bin_facts = bin_facts
        self.mask_params = mask_params
        self.s_ident = s_ident


class MCMod:
    """
    Class that contains basic model info.
    """
    
    def __init__(self, parfile, inparams, psigmas_lib, plims_lib,
                 lam, unit_conv, mod_bin_factor, model_path, log_path, s_ident):
        """
        Initialization code for MCMod.
        
        Inputs:
            
        
        """
        pl_dict = dict()
        
        self.pl = None
        self.pkeys = None
        self.pl_dict = None
        self.parfile = parfile
        self.inparams = inparams
        self.psigmas_lib = psigmas_lib
        self.plims_lib = plims_lib
        self.lam = lam
        self.unit_conv = unit_conv
        self.mod_bin_factor = mod_bin_factor
        self.model_path = model_path
        self.log_path = log_path
        self.s_ident = s_ident


# Define your prior function here. The value that it returns will be added
# to the ln probability of each model.
def mc_lnprior(pl, pkeys):
    """
    Define the flat prior boundaries.
    Takes parameter list pl and parameter keys pkeys as inputs.
    
    Returns 0 if successful, or -infinity if failure.
    """
    
    # edge can't be more than ~1/6 of r_in or MCFOST fails (either
    # "disk radius is smaller than stellar radius" or "r_min < 0.0").
    
    if  2.0 < pl[pkeys=='aexp'] <= 6.5 and \
        1.0 < pl[pkeys=='amin'] <= 40. and \
        -8.8 < pl[pkeys=='dust_mass'] < -6.0: # and \
        # 0.1 < pl[pkeys=='debris_disk_vertical_profile_exponent'] <= 3. and \
        # 0.001 < pl[pkeys=='dust_pop_0_mass_fraction'] < 1. and \
        # 0.001 < pl[pkeys=='dust_pop_1_mass_fraction'] < 1. and \
        # 0.001 < pl[pkeys=='dust_pop_2_mass_fraction'] < 1. and \
        # -3.0 < pl[pkeys=='gamma_exp'] < 3.0 and \
        # 76.0 <= pl[pkeys=='inc'] <= 86. and \
        # 0.001 < pl[pkeys=='porosity'] < 0.95 and \
        # 10. < pl[pkeys=='r_in'] <= 78. and \
        # 0.3 < pl[pkeys=='scale_height'] <= 15. and \
        # -3.0 < pl[pkeys=='surface_density_exp'] < 3.0:
        
        return 0.
    else:
        return -np.inf


def make_mcfmod(pkeys, pl_dict, parfile, model_path, s_ident, fnstring=None, lam=1.6):
    """
    Make an MCFOST model by writing a .para file and calling MCFOST.
    
    Inputs:
        pl_dict: 
            ...
    
    Outputs:
        Technically nothing, but writes MCFOST models to disk.
    """
        
    # Create a new MCFOST .para file object.
    par = Paramfile(parfile)
    
    # NOTE: hard-coded for MCFOST model FITS to have single inclination only.
    par.RT_n_incl = 1
    for pkey in pkeys:
        if pkey=='inc':
            par.RT_imax = pl_dict['inc']
            par.RT_imin = pl_dict['inc']
        # Handle multiple dust populations.
        elif 'dust_pop' in pkey:
            pkey_split = pkey[9:].split('_')
            pop_num = int(pkey_split[0])
            dust_key = "_".join(pkey_split[1:])
 # FIX ME!!! Only indexes correctly for 1 density zone right now. Would need to loop
 # over the [0] index here to go through multiples density zones.
            par.density_zones[0]['dust'][pop_num][dust_key] = pl_dict[pkey]
        # Must loop over all dust populations for some dust parameters.
        elif pkey in ['amin', 'amax', 'aexp', 'ngrains', 'porosity']:
            for dp in range(len(par.density_zones[0]['dust'][:])):
                par.density_zones[0]['dust'][dp][pkey] = pl_dict[pkey]
        elif pkey in ['debris_disk_vertical_profile_exponent', 'edge', 'flaring_exp', 'gamma_exp', 'surface_density_exp']:
            for dz in par.density_zones:
                dz[pkey] = pl_dict[pkey]
        # Log parameters.
        elif pkey in ['dust_mass']:
            par.set_parameter('dust_mass', 10**pl_dict['dust_mass'])
        else:
            par.set_parameter(pkey, pl_dict[pkey])
    
    if fnstring is None:
        fnstring = "%s_mcmc_%s%.5e" % \
                   (s_ident, pkeys[0], pl_dict[pkeys[0]])
    
    modeldir = model_path + fnstring
    try:
        os.mkdir(modeldir)
    except OSError:
        rmtree(modeldir)
        time.sleep(0.2) # short pause after removing directory to make sure completes
        os.mkdir(modeldir)
    
    # Write the .para file in the directory specific to given model.
    par.writeto(modeldir + '/%s.para' % fnstring)
    
    # Modify model directory permissions and cd to it.
    subprocess.call('chmod -R g+w '+modeldir, shell=True)
    os.chdir(modeldir)
    
    # Use try/except to prevent failed MCFOST models from killing the MCMC.
    try:
        # Run MCFOST to create the given model.
        # Re-direct terminal output to a .txt file.
 # FIX ME!!! Specific to scattered-light only.
        subprocess.call('mcfost '+fnstring+'.para -img '+str(lam)+' -rt2 -only_scatt >> imagemcfostout.txt', shell=True)        
    except:
        pass
    
    # cd back into the outer MCMC directory.
    os.chdir(model_path)
    
    return


# FIX ME!!! This whole chi2_morph function needs generalization to
# multiple kinds and numbers of images.

def chi2_morph(path, data, uncerts, mod_bin_factor,
               phi_stokes, unit_conv):
    
    # Use try/except to prevent failed MCFOST models from killing the MCMC.
    # try:
    # Load latest model from file.
    model = fits.getdata(path + '/RT.fits.gz') # [W/m^2...]
    # Convert models to mJy/arcsec^2 to match data.
    # model_I = unit_conv*model[0,0,0,:,:] # [mJy/arcsec^2]
    model_Qr, model_Ur = get_radial_stokes(model[1,0,0,:,:], model[2,0,0,:,:], phi_stokes) # [W/m^2...]
    model_Qr *= unit_conv # [mJy/arcsec^2]
    
# FIX ME!!! Forward modeling is currently disabled.
    # if algo=='loci':
    #     model_I = do_fm_loci(dataset, model_I.copy(), c_list)
    # elif algo=='pyklip':
    #     # Forward model to match the KLIP'd data.
    #     model_I = do_fm_pyklip(modfm, dataset, model_I.copy())

    if mod_bin_factor not in [None, 1]:
        # model_I = zoom(model_I.copy(), 1./mod_bin_factor)*mod_bin_factor
        model_Qr = zoom(model_Qr.copy(), 1./mod_bin_factor)*mod_bin_factor
    
    # Calculate simple chi^2 for I and Qr data.
    # chi2_I = np.nansum(((data_I - model_I)/uncertainty_I)**2)
    chi2_Qr = np.nansum(((data[0] - model_Qr)/uncerts[0])**2)
    
    print(chi2_Qr)
    
    # # Or, calculate reduced chi^2 for I and Qr data.
    # chi2_I = chi_I/(np.where(np.isfinite(data_I))[0].size + len(theta))
    # chi2_Qr = chi_Qr/(np.where(np.isfinite(data_Qr))[0].size + len(theta))
    return np.array(chi2_Qr)
    
        # except:
        #     return np.array(np.inf) #, np.inf


def mc_lnlike(pl, pkeys, data, uncerts, mod_bin_factor, phi_stokes,
             parfile, model_path, unit_conv,
             lam, partemp, ndim, write_model, s_ident):
    """
    Computes and returns the natural log of the likelihood 
    value for a given model.
    
    if par.l_stokes == True:
        imageuncert, imagechi, seduncert, sedchi, poluncert, polchi = mcmcwrapper(theta)
        
        lnpimage = -0.5*np.log(2*np.pi)*imageuncert.size-0.5*imagechi-np.sum(-np.log(imageuncert))
        lnppol = -0.5*np.log(2*np.pi)*poluncert.size-0.5*polchi-np.sum(-np.log(poluncert))
        lnpsed = -0.5*np.log(2*np.pi)*seduncert.size-0.5*sedchi-np.sum(-np.log(seduncert))
        
        return lnpimage + lnppol + lnpsed
    
    else:
        imageuncert, imagechi, seduncert, sedchi = mcmcwrapper(theta)
        
        lnpimage = -0.5*np.log(2*np.pi)*imageuncert.size-0.5*imagechi-np.sum(-np.log(imageuncert))
        lnpsed = -0.5*np.log(2*np.pi)*seduncert.size-0.5*sedchi-np.sum(-np.log(seduncert))
        
        return lnpimage + lnpsed
    """
    
    pl_dict = dict()
    pl_dict.update(zip(pkeys, pl))
    
    # Create a new model object for this set of parameters.
 # FIX ME!!! Remove this formalism.
    # mobj = ModObj(pl, pkeys, parfile, gen_path, model_path, lam, s_ident)
    # pl_dict = mobj.pl_dict
    
    # For affine-invariant sampler, run the prior test.
    if not partemp:
        if not np.isfinite(mc_lnprior(pl, pkeys)):
            return -np.inf
    
    # Assign a unique name to this model directory based on first 3 (or 1) parameters.
    # NOTE: MCFOST seems to get confused when fnstring is longer than ~47 characters.
    try:
        fnstring = "%s_mcmc_%s%.3e_%s%.3e_%s%.3e" % \
                   (s_ident, pkeys[0], pl_dict[pkeys[0]], pkeys[1], pl_dict[pkeys[1]],
                    pkeys[2], pl_dict[pkeys[2]])
    except:
        fnstring = "%s_mcmc_%s%.5e" % \
                   (s_ident, pkeys[0], pl_dict[pkeys[0]])
    
    # Write the MCFOST .para file and create the model.
    make_mcfmod(pkeys, pl_dict, parfile, model_path, s_ident, fnstring, lam)
    
    # Calculate Chi2 for all images in data.
    chi2s = chi2_morph(model_path+fnstring+'/data_%s' % str(lam),
                        data, uncerts, mod_bin_factor,
                        phi_stokes, unit_conv)
    # # Calculate reduced Chi2 for images and SED.
    # chi2_red_I = chi2_I/(np.where(np.isfinite(data_I))[0].size - ndim)
    # chi2_red_Qr = chi2_Qr/(np.where(np.isfinite(data_Qr))[0].size - ndim)
    
    if not write_model:
        try:
            rmtree(model_path+fnstring)
        except:
            time.sleep(0.5)
            subprocess.call('rm -rf %s' % model_path+fnstring, shell=True)
            time.sleep(0.5)
    
    # lnpimage = -0.5*np.log(2*np.pi)*uncertainty_I.size - 0.5*imagechi - np.nansum(-np.log(uncertainty_I))
    # lnpimage = -0.5*np.log(2*np.pi)*uncertainty_I.size - 0.5*chi_red_I - np.nansum(-np.log(uncertainty_I)) + -0.5*np.log(2*np.pi)*uncertainty_Qr.size - 0.5*chi_red_Qr - np.nansum(-np.log(uncertainty_Qr))
    
    return -0.5*(np.sum(chi2s))


def mc_main(s_ident, ntemps, nwalkers, niter, nburn, nthin, nthreads,
            mcdata, mcmod, partemp=True, mc_a=2., init_samples_fn=None,
            write_model=False, plot=False, save=False):
    
    start = time.ctime()
    time_start_secs = time.time()
    
    print("\nSTART TIME:", start)
    
    data = mcdata.data
    uncerts = mcdata.uncerts
    stars = mcdata.stars
    
    inparams = mcmod.inparams
    model_path = mcmod.model_path
    log_path = mcmod.log_path
    lam = mcmod.lam # [microns]
    unit_conv = mcmod.unit_conv

    
    # Sort the parameter names.
    # NOTE: this must be an array (can't be a list).
    pkeys_all = np.array(sorted(inparams.keys()))
    
    # Create log file.
    mcmc_log = open(log_path + '%s_mcmc_log.txt' % s_ident, 'wb')
    
 # FIX ME!!! Need to handle this specific case better.
    # Make phi map specifically for conversion of Stokes to radial Stokes.
    yy, xx = np.mgrid[:data[0].shape[0], :data[0].shape[1]]
    phi_stokes = np.arctan2(yy - stars[0][0], xx - stars[0][1])
    
    # Bin data by factors specified in mcdata.bin_facts list.
    # Do nothing if mcdata.bin_facts is None or 1.
    if mcdata.bin_facts is None:
        mcdata.bin_facts = len(data)*[1]
    
    for ii, bin_fact in enumerate(mcdata.bin_facts):
        data_orig = []
        uncerts_orig = []
        stars_orig = []
        if bin_fact not in [1, None]:
            # Store the original data as backup.
            data_orig.append(data[ii].copy())
            uncerts_orig.append(uncerts[ii].copy())
            # Bin data, uncertainties, and mask by interpolation.
            datum_binned = zoom(np.nan_to_num(data_orig[ii].data), 1./bin_fact)*bin_fact
            uncert_binned = zoom(np.nan_to_num(uncerts_orig[ii]), 1./bin_fact)*bin_fact
 # FIX ME!!! Interpolating the mask doesn't quite work perfectly. Need to re-make from first principles.
            try:
                mask_binned = zoom(np.nan_to_num(data_orig[ii].mask), 1./bin_fact)*bin_fact
            except:
                mask_binned = False
            
            stars_orig.append(stars[ii].copy())
            star_binned = stars_orig[ii]/int(bin_fact)
            
            # radii_binned = make_radii(datum_binned, star_binned)
            
            # mask_fit = np.ones(datum_binned.shape).astype(bool)
            # mask_fit[star_binned[0]-int(hw_y/bin_fact):star_binned[0]+int(hw_y/bin_fact)+1, star_binned[1]-int(hw_x/bin_fact):star_binned[1]+int(hw_x/bin_fact)+1] = False
 # FIX ME!!! Need to specify this inner region mask or happend automatically?
            # mask_fit[radii_binned < r_fit/int(bin_fact)] = True
            
            data[ii] = np.ma.masked_array(datum_binned, mask=mask_binned)
            uncerts[ii] = uncert_binned
            stars[ii] = star_binned
    
    
    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Dont worry, the walkers quickly branch out and explore the
    # rest of the space.

    # Set initial fit parameters from inparams.
    p0 = np.array([inparams[pkey] for pkey in pkeys_all])
    
    ndim = len(p0)
    
    # Just for stdout display purposes.
    print("\nInitial fit parameters:\n", inparams)
    
    # Means for normal distribution of initial walker positions.
    pmeans = dict(zip(pkeys_all, p0))
    
    # Get sigma values only for the input parameters in inparams.
    psigmas = dict()
    plims = dict()
    for pkey in pkeys_all:
        psigmas.update(zip([pkey], [mcmod.psigmas_lib.get(pkey)]))
        plims.update(zip([pkey], [mcmod.plims_lib.get(pkey)]))
    
    # Sort parameter means and sigmas for walker initializationp.
    pmeans_sorted = [val for (key, val) in sorted(pmeans.items())]
    psigmas_sorted = [val for (key, val) in sorted(psigmas.items())]
    plims_sorted = np.array([val for (key, val) in sorted(plims.items())])
    
    if init_samples_fn is None:
        if partemp:
            # # Gaussian ball initialization.
            # p0 = np.random.normal(loc=pmeans_sorted, scale=psigmas_sorted, size=(ntemps, nwalkers, ndim))
            # Uniform initialization between priors.
            p0 = np.random.uniform(low=plims_sorted[:,0], high=plims_sorted[:,1], size=(ntemps, nwalkers, ndim))
            print("\nNtemps = %d, Ndim = %d, Nwalkers = %d, Nstep = %d, Nburn = %d, Nthreads = %d" % (ntemps, ndim, nwalkers, niter, nburn, nthreads))
        else:
            p0 = np.random.normal(loc=pmeans_sorted, scale=psigmas_sorted, size=(nwalkers, ndim))
    else:
        init_step_ind = -1
        try:
            init_samples = hickle.load(log_path + init_samples_fn)
        except:
            init_samples = pickle.load(gzip.open(log_path + init_samples_fn))
        if partemp:
            p0 = init_samples['_chain'][:,:,init_step_ind,:]
            lnprob_init = init_samples['_lnprob'][:,:,init_step_ind]
            lnlike_init = init_samples['_lnlikelihood'][:,:,init_step_ind]
        else:
            p0 = init_samples['_chain'][:,init_step_ind,:]
            lnprob_init = init_samples['_lnprob'][:,init_step_ind]
            lnlike_init = init_samples['_lnlikelihood'][:,init_step_ind]
        print("\nLoaded init_samples from %s.\nWalkers will start from those final positions." % init_samples_fn)
    
    
    log_preamble = ['|---MCMC LOG---|\n\n', '%s' % s_ident,
                        '\nLOG DATE: ' + dt.date.isoformat(dt.date.today()),
                        '\nJOB START: ' + start,
                        '\nNPROCESSORS: %d\n' % nthreads,
                        # '\nDATASET: %s' % ds, '\nDATA IDENT: %s' % d_ident,
                        '\nINITIAL PARAMS: ', str(inparams),
                        '\nMCFOST PARFILE: ', mcmod.parfile,
                        '\n\nMCMC PARAMS: Ndim: %d, Nwalkers: %d, Nburn: %d, Niter: %d, Nthin: %d, Nthreads: %d' % (ndim, nwalkers, nburn, niter, nthin, nthreads),
                        '\nParallel-tempered?: ' + str(partemp), ' , Ntemps: %d' % ntemps,
                        '\na = %.2f' % mc_a,
                        '\nWavelength = %s microns' % str(lam),
                        '\n'
                        ]
    mcmc_log.writelines(log_preamble)
    
    if partemp:
        # Pass data_I, uncertainty_I, and parfile as arguments to emcee sampler.
        sampler = PTSampler(ntemps, nwalkers, ndim, mc_lnlike, mc_lnprior,
                      loglargs=[pkeys_all, data, uncerts, mcmod.mod_bin_factor, phi_stokes,
                        mcmod.parfile, model_path, unit_conv,
                        lam, partemp, ndim, write_model, s_ident],
                      logpargs=[pkeys_all],
                        threads=nthreads)
                      #pool=pool)
        
        # Insert pkeys into the sampler dict for later use.
        sampler.__dict__['pkeys_all'] = pkeys_all
        
    else:
        sampler = EnsembleSampler(nwalkers, ndim, mc_lnlike,
                      args=[pkeys_all, data, uncerts, mcmod.mod_bin_factor, phi_stokes,
                        mcmod.parfile, model_path, unit_conv,
                        lam, partemp, ndim, write_model, s_ident],
                      # logpargs=[pkeys_all],
                        threads=nthreads)
    
    pdb.set_trace()
 # ------ BURN-IN PHASE ------ #
    if nburn > 0:
        print("BURN-IN START...")
        for bb, (pburn, lnprob_burn, lnlike_burn) in enumerate(sampler.sample(p0, iterations=nburn)):
            # Print progress every 25%.
            if bb in [nburn/4, nburn/2, 3*nburn/4]:
                print("PROCESSING ITERATION %d; BURN-IN %.1f%% COMPLETE..." % (bb, 100*float(bb)/nburn))
            pass
    
        # Print burn-in autocorrelation time and acceptance fraction.
        try:
            max_acl_burn = np.nanmax(sampler.acor) # fails if too few iterations
        except:
            max_acl_burn = -1.
        print("Largest Burn-in Autocorrelation Time = %.1f" % max_acl_burn)
        if partemp:
            print("Mean, Median Burn-in Acceptance Fractions: %.2f, %.2f" % (np.mean(sampler.acceptance_fraction[0]), np.median(sampler.acceptance_fraction[0])))
        else:
            print("Mean, Median Burn-in Acceptance Fractions: %.2f, %.2f" % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))
    
    # # TEMP TESTING!!! Plot the burn-in walker chains. Red dots are p0.
        # try:
        #     ch_burn = sampler.chain[0]
        #     fig1 = plt.figure(11)
        #     fig1.clf()
        #     for aa in range(0,ndim):
        #         sbpl = "32%d" % (aa+1)
        #         ax = fig1.add_subplot(int(sbpl))
        #         for ww in range(nwalkers):
        #             ax.plot(range(0,nburn), ch_burn[ww,:,aa], 'k', alpha=0.3)
        #             ax.plot(0., p0[0,ww,aa], 'ro')
        #         ax.set_xlim(-0.5, niter)
        #         # ax.set_ylabel(r'%s (dim %d)' % (pkeys[ff], ff))
        #     plt.draw()
        # except:
        #     pass
        
        
        # Pause interactively between burn-in and main phase.
        # Comment this out if you don't want the script to pause here.
        # pdb.set_trace()
        
        # Reset the sampler chains and lnprobability after burn-in.
        sampler.reset()
    
        print("BURN-IN COMPLETE!")
    
    elif (nburn==0) & (init_samples_fn is not None):
        print("Walkers initialized from file and no burn-in samples requested.")
        sampler.reset()
        pburn = p0
        lnprob_burn = None #lnprob_init 
        lnlike_burn = None #lnlike_init
    else:
        print("No burn-in samples requested.")
        pburn = p0
        lnprob_burn = None
        lnlike_burn = None
    
    
 # ------ MAIN PHASE ------ #
    print("\nMAIN-PHASE MCMC START...")
    if partemp:
        for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn, lnprob0=lnprob_burn, lnlike0=lnlike_burn, iterations=niter)):
            # if np.any(np.isnan(lnprob)):
            #     print("WE'VE GOT A NAN LNPROB!!")
            #     pdb.set_trace()
            # Print progress every 25%.
            if nn in [niter/4, niter/2, 3*niter/4]:
                print("PROCESSING ITERATION %d; MCMC %.1f%% COMPLETE..." % (nn, 100*float(nn)/niter))
                # Log the full sampler or chain (all temperatures) every so often.
                try:
                    # Delete some items from the sampler that don't h/pickle well.
                    sampler_dict = sampler.__dict__.copy()
                    for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
                        sampler_dict.__delitem__(item)
                    # with open(log_path + '%s_smcmc_full_sampler.hkl' % s_ident) as sampler_log:    
                    hickle.dump(sampler_dict, log_path + '%s_mmcmc_full_sampler.hkl' % s_ident, mode='w') #, compression='gzip', compression_opts=7)
                    print("Sampler logged at iteration %d." % nn)
                except:
                    full_chain_log = gzip.open(log_path + '%s_mmcmc_full_chain.txt.gz' % s_ident, 'wb', 7)
                    pickle.dump(sampler.chain, full_chain_log)
                    full_chain_log.close()
    else:
        for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn, lnprob0=lnprob_burn, iterations=niter)):
            # Print progress every 25%.
            if nn in [niter/4, niter/2, 3*niter/4]:
                print("PROCESSING ITERATION %d; MCMC %.1f%% COMPLETE..." % (nn, 100*float(nn)/niter))
                # Log the full sampler or chain (all temperatures) every so often.
                try:
                    # Delete some items from the sampler that don't h/pickle well.
                    sampler_dict = sampler.__dict__.copy()
                    for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
                        sampler_dict.__delitem__(item)
                    # with open(log_path + '%s_smcmc_full_sampler.hkl' % s_ident) as sampler_log:    
                    hickle.dump(sampler_dict, log_path + '%s_mmcmc_full_sampler.hkl' % s_ident, mode='w') #, compression='gzip', compression_opts=7)
                    print("Sampler logged at iteration %d." % nn)
                except:
                    full_chain_log = gzip.open(log_path + '%s_mmcmc_full_chain.txt.gz' % s_ident, 'wb', 7)
                    pickle.dump(sampler.chain, full_chain_log)
                    full_chain_log.close()
    
    
    print('\nMCMC RUN COMPLETE!\n')
    
    
    # Take zero temperature walkers because only they sample posterior distribution.
    # Does not include burn-in samples.
    # Has dimensions [nwalkers, nstep, pl.shape].
    if partemp:
        ch = sampler.chain[0,:,:,:] # zeroth temperature chain only
        samples = ch[:, :, :].reshape((-1, ndim)) #ch[0,:,:,:]
    else:
        ch = sampler.chain
        samples = ch[:, :, :].reshape((-1, ndim))
    
    # Renormalize mass_fraction values so sum to 1.0 (more intuitive).
    wh_pops = [ind for ind, key in enumerate(pkeys_all) if 'dust_pop' in key]
    samples_orig = samples.copy()
    samples[:, wh_pops] /= np.reshape(np.sum(samples_orig[:, wh_pops], axis=1), (samples_orig.shape[0], 1))
    
    # Chain has shape (ntemps, nwalkers, nsteps/nthin, ndim).
    if partemp:
        assert sampler.chain.shape == (ntemps, nwalkers, niter/nthin, ndim)
    else:
        assert sampler.chain.shape == (nwalkers, niter/nthin, ndim)
    
    # Get array of parameter values for each walker at each step in chain.
    # Log the final full chain (all temperatures) for post-analysis- might be huge.
    try:
        hickle.dump(sampler.chain, log_path + '%s_mmcmc_full_chain_gzip.hkl' % s_ident, mode='w', compression='gzip', compression_opts=7)
    except:
        full_chain_log = gzip.open(log_path + '%s_mmcmc_full_chain.txt.gz' % s_ident, 'wb', 7)
        pickle.dump(sampler.chain, full_chain_log)
        full_chain_log.close()
    print("MCMC full chain (all temps) h/pickled and logged as " + log_path + '%s_mmcmc_full_chain...' % s_ident)
    
    blobs = None # not implemented yet
    
    # Print main-phase autocorrelation time and acceptance fraction.
    try:
        # max_acl = np.nanmax(sampler.acor)
        acor_all = sampler.acor
    except:
        # max_acl = -1.
        acor_all = np.array([-1.])
    
    if partemp:
        mc_full_output = dict(acceptance_fraction=sampler.acceptance_fraction,
                            acor=acor_all, blobs=blobs, chain=sampler.chain,
                            lnprobability=sampler.lnprobability, betas=sampler.betas,
                            nprop=sampler.nprop, nprop_accepted=sampler.nprop_accepted,
                            nswap=sampler.nswap, tswap_acceptance_fraction=sampler.tswap_acceptance_fraction,
                            pkeys_all=pkeys_all)
    else:
        mc_full_output = dict(acceptance_fraction=sampler.acceptance_fraction,
                            acor=acor_all, blobs=blobs, chain=sampler.chain,
                            lnprobability=sampler.lnprobability, betas=[None],
                            nprop=[None], nprop_accepted=[None],
                            nswap=[None], tswap_acceptance_fraction=[None],
                            pkeys_all=pkeys_all)
    
    try:
        # Delete some items from the sampler that don't h/pickle well.
        sampler_dict = sampler.__dict__.copy()
        for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
            sampler_dict.__delitem__(item)
        # with open(log_path + '%s_smcmc_full_sampler.hkl' % s_ident) as sampler_log:    
        hickle.dump(sampler_dict, log_path + '%s_mmcmc_full_sampler.hkl' % s_ident, mode='w') #, compression='gzip', compression_opts=7)
        print("Sampler logged at iteration %d." % nn)
    except:
        full_chain_log = gzip.open(log_path + '%s_mmcmc_full_chain.txt.gz' % s_ident, 'wb', 7)
        pickle.dump(sampler.chain, full_chain_log)
        full_chain_log.close()
    
    print("MCMC output (all temps) h/pickled and logged as " + log_path + '%s_mmcmc_full_sampler...' % s_ident)
    # except:
    #     print("WARNING: Failed to log full sampler.")
    
    # Print main-phase autocorrelation time and acceptance fraction.
    try:
        max_acl = np.nanmax(sampler.acor)
        if partemp:
            acor_T0 = sampler.acor[0]
        else:
            acor_T0 = sampler.acor
    except:
        max_acl = -1.
        acor_T0 = -1.
    print("\nLargest Main Autocorrelation Time = %.1f" % max_acl)
    
    # Hickle/Pickle and log select parts of the MCMC output.
    try:
        if partemp:
            lnprob_out = sampler.lnprobability[0] # zero-temp chi-squareds
            # mc_output = ['acceptance_fraction, acor, blobs, chain_T0, lnprobability, pkeys_all', sampler.acceptance_fraction[0], acor_T0, None, ch, lnprob_out, pkeys_all]
            mc_output = dict(acceptance_fraction=sampler.acceptance_fraction[0],
                             acor=np.array([acor_T0]), blobs=blobs, chain=ch,
                             lnprobability=lnprob_out, pkeys_all=pkeys_all)
            try:
                # Hickle won't compress types other than numpy arrays. May prefer pickle here.
                hickle.dump(mc_output, log_path + '%s_mmcmc_lite_sampler_T0_gzip.hkl' % s_ident, mode='w', compression='gzip', compression_opts=5)
            except:
                mc_output_log = gzip.open(log_path + '%s_mmcmc_lite_sampler_T0.txt.gz' % s_ident, 'wb', 5)
                pickle.dump(mc_output, mc_output_log)
                mc_output_log.close()
            
            # Median acceptance fraction of all walkers. 
            print("Mean, Median Acceptance Fractions (zeroth temp): %.2f, %.2f" % (np.mean(sampler.acceptance_fraction[0]), np.median(sampler.acceptance_fraction[0])))
            mcmc_log.writelines('\nMean, Median Acceptance Fractions (zeroth temp): %.2f, %.2f' % (np.mean(sampler.acceptance_fraction[0]), np.median(sampler.acceptance_fraction[0])))
            
            print("\nMCMC T0 output h/pickled and logged as " + os.path.expanduser(log_path + '%s_mmcmc_lite_sampler_T0...' % s_ident))
            
        else:
            lnprob_out = sampler.lnprobability
            mc_output = dict(acceptance_fraction=sampler.acceptance_fraction,
                             acor=np.array([acor_T0]), blobs=blobs, chain=ch,
                             lnprobability=lnprob_out, pkeys_all=pkeys_all)
            try:
                # Hickle won't compress types other than numpy arrays. May prefer pickle here.
                hickle.dump(mc_output, log_path + '%s_mmcmc_lite_sampler_T0_gzip.hkl' % s_ident, mode='w', compression='gzip', compression_opts=5)
            except:
                mc_output_log = gzip.open(log_path + '%s_mmcmc_lite_sampler_T0.txt.gz' % s_ident, 'wb', 5)
                pickle.dump(mc_output, mc_output_log)
                mc_output_log.close()
    
            # Median acceptance fraction of all walkers.
            print("Mean, Median Acceptance Fractions: %.2f, %.2f" % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))
            mcmc_log.writelines('\nMean, Median Acceptance Fractions: %.2f, %.2f' % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))
            
            print("\nMCMC T0 output pickled/hickled and logged as " + os.path.expanduser(log_path + '%s_mmcmc_sampler.txt.gz' % s_ident))
            
    except:
        print("\nWARNING! MCMC output could not be logged.")
    
    
    # Max likelihood params values.
    ind_lk_max = np.where(lnprob_out==lnprob_out.max())
    lk_max = np.e**lnprob_out.max()
    params_ml_mcmc = dict(zip(pkeys_all, ch[ind_lk_max][0]))
    params_ml_mcmc_sorted = [val for (key, val) in sorted(params_ml_mcmc.items())]
    
    # Get mean values (50th percentile) and 1-sigma (68%) confidence intervals
    # for each parameter (in order +, -).
    params_mean_mcmc = map(lambda vv: (vv[1], vv[2]-vv[1], vv[1]-vv[0]),
                        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
    print("\nMax-Likelihood Param Values:")
    mcmc_log.writelines('\n\nMAX-LIKELIHOOD PARAM VALUES:')
    for kk, key in enumerate(pkeys_all):
        #print("mu0 = %.2f +/- %.2f/%2f \nmu1 = %.2f +/- %.2f/%2f \nsig0 = %.2f +/- %.2f/%2f \nsig1 = %.2f +/- %.2f/%2f" % (mu0_mcmc[0], mu0_mcmc[1], mu0_mcmc[2], mu1_mcmc[0], mu1_mcmc[1], mu1_mcmc[2], sig0_mcmc[0], sig0_mcmc[1], sig0_mcmc[2], sig1_mcmc[0], sig1_mcmc[1], sig1_mcmc[2])
        print(key, '= %.3e' % params_ml_mcmc[key])
        mcmc_log.writelines('\n%s = %.3e' % (key, params_ml_mcmc[key]))
    
    print("\n50%-Likelihood Param Values (50th percentile +/- 1 sigma (i.e., 34%):")
    mcmc_log.writelines('\n\n50%-LIKELIHOOD PARAM VALUES (50th percentile +/- 1 sigma (i.e., 34%):')
    for kk, key in enumerate(pkeys_all):
        #print("mu0 = %.2f +/- %.2f/%2f \nmu1 = %.2f +/- %.2f/%2f \nsig0 = %.2f +/- %.2f/%2f \nsig1 = %.2f +/- %.2f/%2f" % (mu0_mcmc[0], mu0_mcmc[1], mu0_mcmc[2], mu1_mcmc[0], mu1_mcmc[1], mu1_mcmc[2], sig0_mcmc[0], sig0_mcmc[1], sig0_mcmc[2], sig1_mcmc[0], sig1_mcmc[1], sig1_mcmc[2]))
        # if key=='dust_mass':
        #     print(key, '= %.3e +/- %.3e/%.3e' % (params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        #     mcmc_log.writelines('\n%s = %.3e +/- %.3e/%.3e' % (key, params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        # else:
        #     print(key, '= %.3f +/- %.3f/%.3f' % (params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        #     mcmc_log.writelines('\n%s = %.3f +/- %.3f/%.3f' % (key, params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        print(key, '= %.3f +/- %.3f/%.3f' % (params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        mcmc_log.writelines('\n%s = %.3f +/- %.3f/%.3f' % (key, params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
    
    
    # Construct max- and mean-likelihood models.
    # try:
    print("\nConstructing 'best-fit' models...")
    mod_idents = ['maxlk', 'meanlk']
    params_50th_mcmc = np.array(params_mean_mcmc)[:,0]
    # # Normalize mass_fraction values so sum to 1.0.
    # params_50th_mcmc[2:5] /= np.sum(params_50th_mcmc.copy()[2:5])
    
    for mm, pl in enumerate([params_ml_mcmc_sorted, params_50th_mcmc]):
        pl_dict = dict()
        pl_dict.update(zip(pkeys_all, pl))
        
        # Name for model and its directory.
        fnstring = "%s_mmcmc_%s_aexp%.3f_amin%.2e_dm%.2e" % \
                    (s_ident, mod_idents[mm], pl_dict['aexp'], pl_dict['amin'], pl_dict['dust_mass'])
        
        # Make the MCFOST model.
        make_mcfmod(pkeys_all, pl_dict, mcmod.parfile, model_path, s_ident, fnstring, lam)
        
        # Calculate Chi2 for images.
        chi2s = chi2_morph(model_path+fnstring+'/data_%s' % str(lam),
                            data, uncerts, mcmod.mod_bin_factor, phi_stokes, unit_conv)
        # Calculate reduced Chi2 for images.
        # chi2_red_I = chi2_I/(np.where(np.isfinite(data_I_masked.filled(np.nan)))[0].size - ndim)
        chi2_red_Qr = chi2s/(np.where(np.isfinite(data[0].filled(np.nan)))[0].size - ndim)
        
        # chi2_red_total = chi2_red_I + chi2_red_Qr
        chi2_red_total = chi2_red_Qr
        
        if mm==0:
            lk_type = 'Max-Likelihood'
            # print('\nMax-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
            # mcmc_log.writelines('\n\nMax-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        elif mm==1:
            lk_type = '50%%-Likelihood'
            # print('50%%-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
            # mcmc_log.writelines('\n50%%-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        # print('%s total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        # mcmc_log.writelines('\n%s total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        # print('%s total chi2_red: %.3e | I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, chi2_red_I, chi2_red_Qr))
        # mcmc_log.writelines('\n%s total chi2_red: %.3e | I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, chi2_red_I, chi2_red_Qr))
        print('%s total chi2_red: %.3e | Qr chi2_red: %.3f' % (lk_type, chi2_red_total, chi2_red_Qr))
        mcmc_log.writelines('\n%s total chi2_red: %.3e | Qr chi2_red: %.3f' % (lk_type, chi2_red_total, chi2_red_Qr))
        
        # Make scattered-light and dust properties models for maxlk and meanlk.
        try:
            os.chdir(model_path + fnstring)
            # Make the dust properties at proper wavelength.
            # NOTE: This must come before the image calculation at the same
            # wavelength, otherwise MCFOST automatically deletes the image directory!
            subprocess.call('rm -rf data_'+str(lam), shell=True)
            subprocess.call('mcfost '+fnstring+'.para -dust_prop -op %s >> dustmcfostout.txt' % lam, shell=True)
            time.sleep(2)
            # # Delete the (mostly) empty data_1.647 directory after dust_prop step.
            # subprocess.call('rm -rf data_'+str(lam), shell=True)
            # Make the SED model.
            subprocess.call('mcfost '+fnstring+'.para -rt >> sedmcfostout.txt', shell=True)
            # Make the scattered-light models at proper wavelengths and PA.
            subprocess.call('mcfost '+fnstring+'.para -img '+str(lam)+' -rt2 >> imagemcfostout.txt', shell=True)
            # time.sleep(2)
            # subprocess.call('mcfost '+fnstring+'.para -img 0.565 -PA %s -rt2 -only_scatt >> imagemcfostout.txt' % skyPA, shell=True)
            subprocess.call('mcfost '+fnstring+'.para -img 0.565 -rt2 -only_scatt >> imagemcfostout.txt', shell=True)
            time.sleep(2)
            print("Made scattered-light and dust properties models.")
        except:
            print("Failed to make scattered-light and dust properties models.")
    
    # Plot and save maxlk and meanlk models.
    try:
        # Placeholder for plotting functions.
        pass
        print("Max and Mean Likelihood models made, plotted, and saved.\n")
    except:
        print("Max and Mean Likelihood models made and saved but plotting failed.\n")

    # # Make triangle plot. Will fail if ln probabilities are weird.
    # try:
    #     import corner
    #     # labels_tri = ['I', 'amin', 'M dust', 'r_in', 'r_out']
    #     labels_tri = pkeys_all
    #     
    #     samples = ch.reshape((-1, ndim)) # All zero-temp samples for every parameter.
    #     fig_tri = corner.corner(samples, labels=labels_tri, quantiles=[0.16, 0.5, 0.84],
    #                             label_kwargs={"fontsize":16},
    #                             show_titles=False, verbose=True)
    # except:
    #     print("Corner plot failed.")
    
    
    time_end_secs = time.time()
    time_elapsed_secs = time_end_secs - time_start_secs # [seconds]
    print("END TIME:", time.ctime())
    print("ELAPSED TIME: %.2f minutes = %.2f hours" % (time_elapsed_secs/60., time_elapsed_secs/3600.))
    
    mcmc_log.writelines(['\n\nSTART TIME - END TIME: ', start, ' - ', time.ctime()])
    mcmc_log.writelines(['\nELAPSED TIME: %.2f minutes = %.2f hours' % (time_elapsed_secs/60., time_elapsed_secs/3600.)])
    mcmc_log.writelines('\n\nSUCCESSFUL EXIT')
    mcmc_log.close()
    
    # pdb.set_trace()
    
    return


# Close MPI pool.
try:
    pool.close()
    print("\nMPI Pool closed")
except:
    print("\nNo MPI pools to close.")

print("diskmc.py script finished\n")

# # Pause interactively before finishing script.
# pdb.set_trace()

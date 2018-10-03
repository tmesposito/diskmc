#!/usr/bin/env python

__author__ = 'Tom Esposito'
__copyright__ = 'Copyright 2018, Tom Esposito'
__credits__ = ['Tom Esposito']
__license__ = 'GNU General Public License v3'
__version__ = '0.1.0'
__maintainer__ = 'Tom Esposito'
__email__ = 'espos13@gmail.com'
__status__ = 'Development'

import os, sys, argparse, warnings
import subprocess 
from shutil import rmtree
import pdb
import time, datetime as dt
import glob
import hickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants
from scipy.ndimage import zoom

try:
    import acor
except ImportError:
    raise ImportError('Package "acor" could not be imported; this is not crucial but no autocorrelation info will be calculated.')
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
    Class that contains data, uncertainties, and associated info.
    """
    
    def __init__(self, data, data_types, stars, uncerts, bin_facts=None, mask_params=None,
                 s_ident='no_ident'):
        """
        Initialization code for MCData.
        
        Inputs:
            data: list of data products (images, SED points, etc.) that will be used
                to compute the likelihood function (i.e. will be "fit").
            data_types: list of str labels for each data type in data (in same order);
                e.g. ['Qr', 'Q', 'U', 'I'] for radial Stokes Q, Stokes Q,
                Stokes U, and total intensity.
            stars: list of array(y,x) integer positions for the star in each data
                product; element should be None if not relevant (like for an SED).
            uncerts: list of uncertainties for data products.
            bin_facts: list of factors by which data products are spatially binned;
                only applies to images and should be None for all others.
            mask_params: not implemented.
            s_ident: str identifier for the MCMC run; no spaces allowed.
        """
        
        self.data = data
        self.data_types = data_types
        self.stars = stars
        self.uncerts = uncerts
        self.bin_facts = bin_facts
        self.mask_params = mask_params
        self.s_ident = s_ident


class MCMod:
    """
    Class that contains basic model info.
    """
    
    def __init__(self, pkeys, parfile, pmeans_lib=None, psigmas_lib=None,
                 plims_lib=None, priors=None, lam=1., unit_conv=1.,
                 mod_bin_factor=None, model_path='.', log_path='.', s_ident='no_ident'):
        """
        Initialization code for MCMod.
        
        Inputs:
            pkeys: array of str names for each model parameter being varied in the MCMC.
                These MUST match those used in diskmc.make_mcfmod().
            parfile: str path to the initial MCFOST parameter file used as a basis for
                all MCMC models created.
            pmeans_lib: dict of mean values for all model parameters if using a
                Gaussian walker initialization. See run_diskmc.py for details.
            psigmas_lib: dict of sigma values for all model parameters if using a
                Gaussian walker initialization. See run_diskmc.py for details.
            plims_lib: dict of uniform distribution limits for all model parameters
                if using a uniform walker initialization. See run_diskmc.py for details.
            priors: dict of flat prior lower and upper bounds (exclusive).
                See run_diskmc.py for details.
            lam: single wavelength at which to create the MCFOST models [microns]
            unit_conv: multiplicative conversion factor to convert MCFOST models
                into whatever unit is desired for the likelihood function.
            mod_bin_factor: factor by which models images are spatially binned;
                may need to match bin_facts set in MCData.
            model_path: str path to outer directory that will hold all diskmc
                model output.
            log_path: str path to directory for diskmc logs to go into.
            s_ident: str identifier for the MCMC run; no spaces allowed.
        """
        
        self.pkeys = pkeys
        self.parfile = parfile
        self.pmeans_lib = pmeans_lib
        self.psigmas_lib = psigmas_lib
        self.plims_lib = plims_lib
        self.priors = priors
        self.lam = lam
        self.unit_conv = unit_conv
        self.mod_bin_factor = mod_bin_factor
        self.model_path = model_path
        self.log_path = log_path
        self.s_ident = s_ident
        self.pl = None
        self.pl_dict = None


# Define your prior function here. The value that it returns will be added
# to the ln probability of each model.
def mc_lnprior(pl, pkeys, priors):
    """
    Define the flat prior boundaries.
    Takes parameter list pl and parameter keys pkeys as inputs.
    
    Inputs:
        pl: array of parameter values (must be in same order as pkeys).
        pkeys: array of sorted str pkeys (must be in same order as pl).
        priors: dict with same keys as pkeys (but order doesn't matter).
            If None, all models will pass.
    
    Returns 0 if successful, or -infinity if failure.
    """
    
    # NOTE: edge can't be more than ~1/6 of r_in or MCFOST fails (get either
    # "disk radius is smaller than stellar radius" or "r_min < 0.0" error).
    
    if priors is not None:
        for ii, key in enumerate(pkeys):
            if priors[key][0] < pl[pkeys==key] < priors[key][1]:
                continue
            else:
                return -np.inf
    
    # If get to here, all parameters pass the prior and returns 0.
    return 0.


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
        elif pkey=='disk_pa':
            par.disk_pa = pl_dict['disk_pa']
        # Handle multiple dust populations.
        elif 'dust_pop' in pkey:
            pkey_split = pkey[9:].split('_')
            pop_num = int(pkey_split[0])
            dust_key = "_".join(pkey_split[1:])
 # FIX ME!!! Only indexes correctly for 1 density zone right now. Would need to loop
 # over the [0] index here to go through multiples density zones.
            par.density_zones[0]['dust'][pop_num][dust_key] = pl_dict[pkey]
        # Must loop over all dust populations for some dust parameters.
        elif pkey in ['amax', 'aexp', 'ngrains', 'porosity']:
            for dp in range(len(par.density_zones[0]['dust'][:])):
                par.density_zones[0]['dust'][dp][pkey] = pl_dict[pkey]
        elif pkey in ['debris_disk_vertical_profile_exponent', 'edge', 'flaring_exp', 'gamma_exp', 'surface_density_exp']:
            for dz in par.density_zones:
                dz[pkey] = pl_dict[pkey]
        # Log parameters.
        elif pkey in ['dust_mass']:
            par.set_parameter('dust_mass', 10**pl_dict['dust_mass'])
        # Log parameters looped over all dust populations.
        elif pkey == 'amin':
            for dp in range(len(par.density_zones[0]['dust'][:])):
                par.density_zones[0]['dust'][dp][pkey] = 10**pl_dict[pkey]
        else:
            par.set_parameter(pkey, pl_dict[pkey])
    
    if fnstring is None:
        fnstring = "%s_mcmc_%s%.5e" % \
                   (s_ident, pkeys[0], pl_dict[pkeys[0]])
    
    modeldir = model_path + fnstring
    try:
        os.mkdir(modeldir)
    except OSError:
        time.sleep(0.5) # short pause
# FIX ME!!! This rmtree will cause an OSError if the directory doesn't exist.
# Need a better solution than this try/except block.
        rmtree(modeldir) # delete any existing directory
        time.sleep(0.5) # pause after removing directory to make sure completes
        os.mkdir(modeldir) # try to make directory again
    
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


def chi2_morph(path, data, uncerts, data_types, mod_bin_factor,
               phi_stokes, unit_conv):
    """
    Calculate simple chi-squared for each data-model pair. Only pertains to
    morphological models (i.e., images) and not SED's, photometry, etc.
    
    NOTE! Currently assumes same binning factor for every model.
    
    """
    
    # Use try/except to prevent failed MCFOST models from killing the MCMC.
    try:
        # Load latest model from file.
        model = fits.getdata(path + '/RT.fits.gz') # [W/m^2...]
        
        chi2s = []
        
        for ii, dt in enumerate(data_types):
            if dt == 'Qr':
                mod_use, model_Ur = get_radial_stokes(model[1,0,0,:,:], model[2,0,0,:,:], phi_stokes) # [W/m^2...]
            elif dt == 'I':
                mod_use = model[0,0,0,:,:]
                
 # FIX ME!!! Total intensity forward modeling is currently disabled.
                # if algo=='loci':
                #     mod_use = do_fm_loci(dataset, mod_use.copy(), c_list)
                # elif algo=='pyklip':
                #     # Forward model to match the KLIP'd data.
                #     mod_use = do_fm_pyklip(modfm, dataset, model_I.copy())
            
            elif dt == 'Q':
                mod_use = model[1,0,0,:,:]
            elif dt == 'U':
                mod_use = model[2,0,0,:,:]
            
            # Convert units of model according to unit_conv.
            mod_use *= unit_conv # [converted brightness units]
            # Bin model by mod_bin_factor and conserve flux.
            mod_use = zoom(mod_use.copy(), 1./mod_bin_factor)*mod_bin_factor
            
            # Calculate simple chi^2 data and model.
            chi2s.append(np.nansum(((data[ii] - mod_use)/uncerts[ii])**2))
            
            # # Or, calculate reduced chi^2.
            # chi2_Qr = chi_Qr/(np.where(np.isfinite(data_Qr))[0].size + len(theta))
        
        return np.array(chi2s)
    except:
        return np.array(np.inf)


def mc_lnlike(pl, pkeys, data, uncerts, data_types, mod_bin_factor, phi_stokes,
             parfile, model_path, unit_conv, priors,
             lam, partemp, ndim, write_model, s_ident):
    """
    Computes and returns the natural log of the likelihood value
    for a given model.
    
    """
    
    # Update pl_dict with new values for convenience.
    pl_dict = dict()
    pl_dict.update(zip(pkeys, pl))
    
    # For affine-invariant ensemble sampler, run the prior test here.
    if not partemp:
        if not np.isfinite(mc_lnprior(pl, pkeys, priors)):
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
    # try/except here works around some unsolved directory creation/deletion issues.
    try:
        make_mcfmod(pkeys, pl_dict, parfile, model_path, s_ident, fnstring, lam)
    except:
        return -np.inf
    
    # Calculate Chi2 for all images in data.
    chi2s = chi2_morph(model_path+fnstring+'/data_%s' % str(lam),
                        data, uncerts, data_types, mod_bin_factor,
                        phi_stokes, unit_conv)
    # # Calculate reduced Chi2 for images.
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
    
    print("\nSTART TIME: " + start)
    
    data = mcdata.data
    uncerts = mcdata.uncerts
    data_types = np.array(mcdata.data_types) # need as nd.array for later
    stars = mcdata.stars
    
    model_path = mcmod.model_path
    log_path = mcmod.log_path
    lam = mcmod.lam # [microns]
    unit_conv = mcmod.unit_conv
    
    # Sort the parameter names.
    # NOTE: this must be an array (can't be a list).
    pkeys_all = np.array(sorted(mcmod.pkeys))
    
    # Create log file.
    mcmc_log_fn = log_path + '%s_mcmc_log.txt' % s_ident
    mcmc_log = open(mcmc_log_fn, 'w')
    
 # FIX ME!!! Need to handle this specific case better.
    # Make phi map specifically for conversion of Stokes to radial Stokes.
    yy, xx = np.mgrid[:data[0].shape[0], :data[0].shape[1]]
    phi_stokes = np.arctan2(yy - stars[0][0], xx - stars[0][1])
    
    # Bin data by factors specified in mcdata.bin_facts list.
    # Do nothing if mcdata.bin_facts is None or its elements are 1.
    if mcdata.bin_facts is None:
        mcdata.bin_facts = len(data)*[1]
    
    data_orig = []
    uncerts_orig = []
    stars_orig = []
    for ii, bin_fact in enumerate(mcdata.bin_facts):
        if bin_fact not in [1, None]:
            # Store the original data as backup.
            data_orig.append(data[ii].copy())
            uncerts_orig.append(uncerts[ii].copy())
            # Bin data, uncertainties, and mask by interpolation.
            datum_binned = zoom(np.nan_to_num(data_orig[ii].data), 1./bin_fact)*bin_fact
            uncert_binned = zoom(np.nan_to_num(uncerts_orig[ii]), 1./bin_fact, order=1)*bin_fact
 # FIX ME!!! Interpolating the mask may not work perfectly. Linear interpolation (order=1)
 # is best so far.
            try:
                mask_binned = zoom(np.nan_to_num(data_orig[ii].mask), 1./bin_fact, order=1)
            except:
                mask_binned = False
            
            stars_orig.append(stars[ii].copy())
            star_binned = stars_orig[ii]/int(bin_fact)
            
            # radii_binned = make_radii(datum_binned, star_binned)
            
            # mask_fit = np.ones(datum_binned.shape).astype(bool)
            # mask_fit[star_binned[0]-int(hw_y/bin_fact):star_binned[0]+int(hw_y/bin_fact)+1, star_binned[1]-int(hw_x/bin_fact):star_binned[1]+int(hw_x/bin_fact)+1] = False
 # FIX ME!!! Need to specify this inner region mask or happens automatically?
            # mask_fit[radii_binned < r_fit/int(bin_fact)] = True
            
            data[ii] = np.ma.masked_array(datum_binned, mask=mask_binned)
            uncerts[ii] = uncert_binned
            stars[ii] = star_binned
    
    
    ####################################
    # ------ INITIALIZE WALKERS ------ #
    # This will be done using a uniform distribution drawn from
    # plims_lib (first option if not None) or a Gaussian distribution
    # drawn from pmeans_lib and psigmas_lib (if plims_lib == None).
    
    ndim = len(pkeys_all)
    
    print("\nNtemps = %d, Ndim = %d, Nwalkers = %d, Nstep = %d, Nburn = %d, Nthreads = %d" % (ntemps, ndim, nwalkers, niter, nburn, nthreads))
    
    # Sort parameters for walker initialization.
    if mcmod.plims_lib is not None:
        plims_sorted = np.array([val for (key, val) in sorted(mcmod.plims_lib.items())])
        init_type = 'uniform'
    elif mcmod.pmeans_lib is not None:
        pmeans_sorted = [val for (key, val) in sorted(mcmod.pmeans_lib.items())]
        psigmas_sorted = [val for (key, val) in sorted(mcmod.psigmas_lib.items())]
        init_type = 'gaussian'
    
    # Make the array of initial walker positions p0.
    if init_samples_fn is None:
        if partemp:
            if init_type == 'uniform':
                # Uniform initialization between priors.
                p0 = np.random.uniform(low=plims_sorted[:,0], high=plims_sorted[:,1], size=(ntemps, nwalkers, ndim))
            elif init_type == 'gaussian':
                # Gaussian ball initialization.
                p0 = np.random.normal(loc=pmeans_sorted, scale=psigmas_sorted, size=(ntemps, nwalkers, ndim))
        else:
            if init_type == 'uniform':
                # Uniform initialization between priors.
                p0 = np.random.uniform(low=plims_sorted[:,0], high=plims_sorted[:,1], size=(nwalkers, ndim))
            elif init_type == 'gaussian':
                # Gaussian ball initialization.
                p0 = np.random.normal(loc=pmeans_sorted, scale=psigmas_sorted, size=(nwalkers, ndim))
        print("Walker initialization = " + init_type)
    else:
        init_type == 'sampled'
        init_step_ind = -1
        init_samples = hickle.load(log_path + init_samples_fn)
        if partemp:
            p0 = init_samples['_chain'][:,:,init_step_ind,:]
            lnprob_init = init_samples['_lnprob'][:,:,init_step_ind]
            lnlike_init = init_samples['_lnlikelihood'][:,:,init_step_ind]
        else:
            p0 = init_samples['_chain'][:,init_step_ind,:]
            lnprob_init = init_samples['_lnprob'][:,init_step_ind]
            lnlike_init = init_samples['_lnlikelihood'][:,init_step_ind]
        print("\nLoaded init_samples from %s.\nWalkers will start from those final positions." % init_samples_fn)
    
    # Try to get the MCFOST version number being used.
    try:
        output = subprocess.check_output("mcfost -v", shell=True)
        mcf_version = output.split('\n')[0].split(' ')[-1]
    except:
        mcf_version = 'unknown'
    print("Running MCFOST version %s" % mcf_version)
    
    log_preamble = ['|---MCMC LOG---|\n\n', '%s' % s_ident,
                        '\nLOG DATE: ' + dt.date.isoformat(dt.date.today()),
                        '\nJOB START: ' + start,
                        '\nNPROCESSORS: %d\n' % nthreads,
                        '\nMCFOST version: %s' % mcf_version,
                        '\nMCFOST PARFILE: ', mcmod.parfile,
                        '\n\nMCMC PARAMS: Ndim: %d, Nwalkers: %d, Nburn: %d, Niter: %d, Nthin: %d, Nthreads: %d' % (ndim, nwalkers, nburn, niter, nthin, nthreads),
                        '\nPARALLEL-TEMPERED?: ' + str(partemp), ' , Ntemps: %d' % ntemps,
                        '\nINITIALIZATION: ', init_type,
                        '\na = %.2f' % mc_a,
                        '\nWavelength = %s microns' % str(lam),
                        '\n'
                        ]
    mcmc_log.writelines(log_preamble)
    mcmc_log.close()
    
    # Create emcee sampler instances: parallel-tempered or ensemble.
    if partemp:
        # Pass data_I, uncertainty_I, and parfile as arguments to emcee sampler.
        sampler = PTSampler(ntemps, nwalkers, ndim, mc_lnlike, mc_lnprior,
                      loglargs=[pkeys_all, data, uncerts, data_types, mcmod.mod_bin_factor,
                        phi_stokes, mcmod.parfile, model_path, unit_conv, mcmod.priors,
                        lam, partemp, ndim, write_model, s_ident],
                      logpargs=[pkeys_all, mcmod.priors],
                        threads=nthreads)
                      #pool=pool)
        
        # Insert pkeys into the sampler dict for later use.
        sampler.__dict__['pkeys_all'] = pkeys_all
        
    else:
        sampler = EnsembleSampler(nwalkers, ndim, mc_lnlike,
                      args=[pkeys_all, data, uncerts, data_types, mcmod.mod_bin_factor,
                        phi_stokes, mcmod.parfile, model_path, unit_conv, mcmod.priors,
                        lam, partemp, ndim, write_model, s_ident],
                      # logpargs=[pkeys_all],
                        threads=nthreads)


    ###############################
    # ------ BURN-IN PHASE ------ #
    
    if nburn > 0:
        print("\nBURN-IN START...\n")
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
    
    
    ############################
    # ------ MAIN PHASE ------ #
    
    print("\nMAIN-PHASE MCMC START...\n")
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
                    # Delete some items from the sampler that don't hickle well.
                    sampler_dict = sampler.__dict__.copy()
                    for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
                        try:
                            sampler_dict.__delitem__(item)
                        except:
                            continue
                    hickle.dump(sampler_dict, log_path + '%s_mcmc_full_sampler.hkl' % s_ident, mode='w')
                    print("Sampler logged at iteration %d." % nn)
                except:
                    hickle.dump(sampler.chain, log_path + '%s_mcmc_full_chain.hkl' % s_ident, mode='w')
    else:
        for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn, lnprob0=lnprob_burn, iterations=niter)):
            # Print progress every 25%.
            if nn in [niter/4, niter/2, 3*niter/4]:
                print("PROCESSING ITERATION %d; MCMC %.1f%% COMPLETE..." % (nn, 100*float(nn)/niter))
                # Log the full sampler or chain (all temperatures) every so often.
                try:
                    # Delete some items from the sampler that don't hickle well.
                    sampler_dict = sampler.__dict__.copy()
                    for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
                        try:
                            sampler_dict.__delitem__(item)
                        except:
                            continue
                    hickle.dump(sampler_dict, log_path + '%s_mcmc_full_sampler.hkl' % s_ident, mode='w')
                    print("Sampler logged at iteration %d." % nn)
                except:
                    hickle.dump(sampler.chain, log_path + '%s_mcmc_full_chain.hkl' % s_ident, mode='w')
    
    
    print('\nMCMC RUN COMPLETE!\n')
    
    
    ##################################
    # ------ RESULTS HANDLING ------ #
    # Log the sampler output and chains. Also get the max and median likelihood
    # parameter values and save models for them.
    
    # If possible, dump the whole sampler to an HDF5 log file (could be large).
    # If that fails for some reason, just dump the sampler chains.
    try:
        # Delete some items from the sampler that don't hickle well
        # and are not typically useful later.
        sampler_dict = sampler.__dict__.copy()
        for item in ['pool', 'logl', 'logp', 'logpkwargs', 'loglkwargs']:
            try:
                sampler_dict.__delitem__(item)
            except:
                continue
        hickle.dump(sampler_dict, log_path + '%s_mcmc_full_sampler.hkl' % s_ident, mode='w') #, compression='gzip', compression_opts=7)
        print("Full sampler (all temps) logged as " + log_path + '%s_mcmc_full_sampler.hkl' % s_ident)
    except:
        # As last resort, log the final full chain (all temperatures) for post-analysis.
        print("FAILED to log full sampler!! Trying to save the chains...")
        hickle.dump(sampler.chain, log_path + '%s_mcmc_full_chain_gzip.hkl' % s_ident, mode='w', compression='gzip', compression_opts=7)
        print("MCMC full chain (all temps) logged as " + log_path + '%s_mcmc_full_chain...' % s_ident)
    
    
    # Chain has shape (ntemps, nwalkers, nsteps/nthin, ndim).
    if partemp:
        assert sampler.chain.shape == (ntemps, nwalkers, niter/nthin, ndim)
    else:
        assert sampler.chain.shape == (nwalkers, niter/nthin, ndim)
    
    # Re-open the text log for additional info.
    mcmc_log = open(mcmc_log_fn, 'a')
    
    # If multiple temperatures, take zero temperature walkers because only they
    # sample the posterior distribution.
    # ch has dimensions [nwalkers, nstep, ndim] and excludes burn-in samples.
    if partemp:
        ch = sampler.chain[0,:,:,:] # zeroth temperature chain only
        samples = ch[:, :, :].reshape((-1, ndim))
        lnprob_out = sampler.lnprobability[0] # zero-temp chi-squareds
        # Median acceptance fraction of zero temp walkers. 
        print("\nMean, Median Acceptance Fractions (zeroth temp): %.2f, %.2f" % (np.mean(sampler.acceptance_fraction[0]), np.median(sampler.acceptance_fraction[0])))
        mcmc_log.writelines('\nMean, Median Acceptance Fractions (zeroth temp): %.2f, %.2f' % (np.mean(sampler.acceptance_fraction[0]), np.median(sampler.acceptance_fraction[0])))
    else:
        ch = sampler.chain
        samples = ch[:, :, :].reshape((-1, ndim))
        lnprob_out = sampler.lnprobability # all chi-squareds
        # Median acceptance fraction of all walkers.
        print("\nMean, Median Acceptance Fractions: %.2f, %.2f" % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))
        mcmc_log.writelines('\nMean, Median Acceptance Fractions: %.2f, %.2f' % (np.mean(sampler.acceptance_fraction), np.median(sampler.acceptance_fraction)))

    
    # Renormalize mass_fraction values so sum to 1.0 (more intuitive).
    wh_pops = [ind for ind, key in enumerate(pkeys_all) if 'dust_pop' in key]
    samples_orig = samples.copy()
    samples[:, wh_pops] /= np.reshape(np.sum(samples_orig[:, wh_pops], axis=1), (samples_orig.shape[0], 1))
    
    # Haven't implemented blobs handling yet.
    blobs = None
    
    # Print zero temp main-phase autocorrelation time and acceptance fraction.
    try:
        max_acl = np.nanmax(sampler.acor)
        if partemp:
            acor_T0 = sampler.acor[0]
        else:
            acor_T0 = sampler.acor
    except:
        max_acl = -1.
        acor_T0 = -1.
    print("Largest Main Autocorrelation Time = %.1f" % max_acl)
    
    
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
        print(key + ' = %.3e' % params_ml_mcmc[key])
        mcmc_log.writelines('\n%s = %.3e' % (key, params_ml_mcmc[key]))

    print("\n50%-Likelihood Param Values (50th percentile +/- 1 sigma (i.e., 34%):")
    mcmc_log.writelines('\n\n50%-LIKELIHOOD PARAM VALUES (50th percentile +/- 1 sigma (i.e., 34%):')
    for kk, key in enumerate(pkeys_all):
        print(key + ' = %.3f +/- %.3f/%.3f' % (params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
        mcmc_log.writelines('\n%s = %.3f +/- %.3f/%.3f' % (key, params_mean_mcmc[kk][0], params_mean_mcmc[kk][1], params_mean_mcmc[kk][2]))
    
    
    # Construct max- and mean-likelihood models.
    # try:
    print("\nConstructing 'best-fit' models...")
    mod_idents = ['maxlk', 'meanlk']
    params_50th_mcmc = np.array(params_mean_mcmc)[:,0]
    
    for mm, pl in enumerate([params_ml_mcmc_sorted, params_50th_mcmc]):
        pl_dict = dict()
        pl_dict.update(zip(pkeys_all, pl))
        
        # Name for model and its directory.
        try:
            fnstring = "%s_mcmc_%s_%s%.3e_%s%.3e_%s%.3e" % \
                       (s_ident, mod_idents[mm], pkeys_all[0], pl_dict[pkeys_all[0]], pkeys_all[1], pl_dict[pkeys_all[1]],
                        pkeys_all[2], pl_dict[pkeys_all[2]])
        except:
            fnstring = "%s_mcmc_%s_%s%.5e" % \
                       (s_ident, mod_idents[mm], pkeys_all[0], pl_dict[pkeys_all[0]])
        
        # Make the MCFOST model.
        make_mcfmod(pkeys_all, pl_dict, mcmod.parfile, model_path, s_ident, fnstring, lam)
        
        # Calculate Chi2 for images.
        chi2s = chi2_morph(model_path+fnstring+'/data_%s' % str(lam),
                            data, uncerts, data_types, mcmod.mod_bin_factor, phi_stokes, unit_conv)
        # Calculate reduced Chi2 for images.
        chi2_reds = np.array([chi2s[ii]/(np.where(np.isfinite(data[ii].filled(np.nan)))[0].size - ndim) for ii in range(len(data))])
        chi2_red_total = np.sum(chi2_reds)
        
        if mm==0:
            lk_type = 'Max-Likelihood'
            # print('\nMax-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
            # mcmc_log.writelines('\n\nMax-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        elif mm==1:
            lk_type = '50%-Likelihood'
            # print('50%%-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
            # mcmc_log.writelines('\n50%%-Likelihood total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        # print('%s total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        # mcmc_log.writelines('\n%s total chi2_red: %.3e | SED Cushing G: %.3e , I chi2_red: %.3f , Qr chi2_red: %.3f' % (lk_type, chi2_red_total, G_mm, chi2_red_I, chi2_red_Qr))
        print('%s total chi2_red: %.3e' % (lk_type, chi2_red_total))
        print("individual chi2_red's: " + len(chi2_reds)*"%.3e | " % tuple(chi2_reds))
        mcmc_log.writelines('\n\n%s total chi2_red: %.3e' % (lk_type, chi2_red_total))
        mcmc_log.writelines("\nindividual chi2_red's: " + len(chi2_reds)*"%.3e | " % tuple(chi2_reds))
        
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
            time.sleep(2)
            print("Saved scattered-light and dust properties models.")
        except:
            print("Failed to save scattered-light and dust properties models.")
    
    # Plot and save maxlk and meanlk models.
    try:
        # Placeholder for plotting functions.
        pass
        print("Max and Mean Likelihood models made, plotted, and saved.\n")
    except:
        print("Max and Mean Likelihood models made and saved but plotting failed.\n")
    
    
    time_end_secs = time.time()
    time_elapsed_secs = time_end_secs - time_start_secs # [seconds]
    print("END TIME: " + time.ctime())
    print("ELAPSED TIME: %.2f minutes = %.2f hours" % (time_elapsed_secs/60., time_elapsed_secs/3600.))
    
    mcmc_log.writelines(['\n\nSTART TIME - END TIME: ', start, ' - ', time.ctime()])
    mcmc_log.writelines(['\nELAPSED TIME: %.2f minutes = %.2f hours' % (time_elapsed_secs/60., time_elapsed_secs/3600.)])
    mcmc_log.writelines('\n\nSUCCESSFUL EXIT')
    mcmc_log.close()
    
    # Close MPI pool.
    try:
        pool.close()
        print("\nMPI Pool closed")
    except:
        print("\nNo MPI pools to close.")
    
    print("\nmc_main function finished\n")
    
    # # Pause interactively before finishing script.
    # pdb.set_trace()
    
    return

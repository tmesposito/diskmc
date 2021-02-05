#!/usr/bin/env python
"""
Main functions for the DiskMC package to run an MCMC using MCFOST to create
disk models.
"""

__author__ = 'Tom Esposito'
__copyright__ = 'Copyright 2018, Tom Esposito'
__credits__ = ['Tom Esposito']
__license__ = 'GNU General Public License v3'
__version__ = '0.2.0'
__maintainer__ = 'Tom Esposito'
__email__ = 'espos13@gmail.com'
__status__ = 'Development'

import os, sys, argparse, warnings
import subprocess 
from shutil import rmtree
try:
    import ipdb as pdb
except:
    import pdb
import time, datetime as dt
import glob
import hickle
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants
from scipy.ndimage import zoom

import emcee
emcee_version_major = int(emcee.__version__.split('.')[0])
try:
    import acor
except ImportError:
    raise ImportError('Package "acor" could not be imported; this is not crucial but no autocorrelation info will be calculated.')
try:
    from emcee import PTSampler, EnsembleSampler
except ImportError:
    from emcee import EnsembleSampler
#from emcee.utils import MPIPool
#from mpi4py import MPI

if sys.version_info.major >= 3:
    from diskmc_tools import get_ann_stdmap, make_radii, get_radial_stokes
    from paramfiles import Paramfile
else:
    from diskmc_tools import get_ann_stdmap, make_radii, get_radial_stokes
    from paramfiles import Paramfile


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
                 s_ident='no_ident', algo_I=None):
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
            algo_I: total intensity PSF subtraction algorithm used, to set correct
                forward modeling method; 'pyklip' is only option right now.
        """
        
        self.data = data
        self.data_types = data_types
        self.stars = stars
        self.uncerts = uncerts
        self.bin_facts = bin_facts
        self.mask_params = mask_params
        self.s_ident = s_ident
        self.algo_I = algo_I


class MCMod:
    """
    Class that contains basic model info.
    """
    
    def __init__(self, pkeys, parfile, pmeans_lib=None, psigmas_lib=None,
                 plims_lib=None, priors=None, scatlight=True, fullimg=False,
                 sed=False, dustprops=False, lam=1., unit_conv=1.,
                 mod_bin_factor=None, model_path='.', log_path='.', s_ident='no_ident',
                 modfm=None, fm_params=None):
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
            scatlight: bool, True to make a scattered-light-only model at wavelength lam (default).
            fullimg: bool, True to compute the temperature profile, SED, and an image
                at wavelength lam that combines thermal and scattered-light emission.
            sed: bool, True to make an SED model from the new .para file.
                Ignored if fullimg==True.
            dustprops: bool, True to output the dust properties like phase function.
            lam: single wavelength at which to create the MCFOST models [microns]
            unit_conv: multiplicative conversion factor to convert MCFOST models
                into whatever unit is desired for the likelihood function.
            mod_bin_factor: factor by which models images are spatially binned;
                may need to match bin_facts set in MCData.
            model_path: str path to outer directory that will hold all diskmc
                model output.
            log_path: str path to directory for diskmc logs to go into.
            s_ident: str identifier for the MCMC run; no spaces allowed.
            modfm: for total intensity forward modeling, either a DiskFM object
                already prepared or True to build a DiskFM object based on fm_params.
            fm_params: library of 
        """
        
        self.pkeys = pkeys
        self.parfile = parfile
        self.pmeans_lib = pmeans_lib
        self.psigmas_lib = psigmas_lib
        self.plims_lib = plims_lib
        self.priors = priors
        self.scatlight = scatlight
        self.fullimg = fullimg
        self.sed = sed
        self.dustprops = dustprops
        self.lam = lam
        self.unit_conv = unit_conv
        self.mod_bin_factor = mod_bin_factor
        self.model_path = model_path
        self.log_path = log_path
        self.s_ident = s_ident
        self.pl = None
        self.pl_dict = None
        self.modfm = modfm
        self.fm_parmas = fm_params


def log_sampler(sampler, sampler_keys_trim, log_path, s_ident, nn):
    """
    Function to save important sampler items to log file for later use.
    
    Returns a str message about the logging results.
    """
    log_message = 'No log message'

    sampler_dict = sampler.__dict__.copy()

    if emcee_version_major >= 3:
        sampler_dict['_chain'] = sampler.chain.copy()
        sampler_dict['_lnprob'] = sampler.lnprobability.copy()

    try:
        # Delete some items from the sampler that may not hickle well.
        for item in sampler_keys_trim:
            try:
                sampler_dict.__delitem__(item)
            except:
                continue
        log_name = os.path.join(log_path, '{}_mcmc_full_sampler.hkl'.format(s_ident))
        hickle.dump(sampler_dict, log_name, mode='w')
    # If hickle fails, try to pickle the full sampler.
    except NotImplementedError as ee:
        log_name = os.path.join(log_path, '{}_mcmc_full_sampler.pkl'.format(s_ident))
        log_message = "WARNING: Sampler pickled (hickle FAILED) at iteration {0!s} as {1}\nHickle error was: {2}".format(nn, log_name, ee)
        with open(log_name, 'w+') as pickle_log:
            pickle.dump(sampler_dict, pickle_log, protocol=2)
        # print("Hickle error was: %s" % ee)
    # If hickle and pickle both fail, just log the chain array only.
    except Exception as ee:
        # print("WARNING: Logging sampler FAILED at iteration {!s}!".format(nn))
        # print("Error was: %s" % ee)
        log_name = os.path.join(log_path, '{}_mcmc_full_chain_gzip.hkl'.format(s_ident))
        log_message = "WARNING: Logging sampler FAILED at iteration {0!s}!\nError was: {1}\nLogged the MCMC CHAIN ONLY (all temps) as {2}".format(nn, ee, log_name)
        hickle.dump(sampler.chain, log_name, mode='w', compression='gzip', compression_opts=7)
    else:
        log_message = "Sampler logged at iteration {0!s} as {1}".format(nn, log_name)

    print(log_message)

    return log_message


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


def make_mcfmod(pkeys, pl_dict, parfile, model_path, s_ident='', fnstring=None,
                lam=1.6, scatlight=True, fullimg=False, sed=False, dustprops=False):
    """
    Make an MCFOST model by writing a .para parameter file and passing it to MCFOST.
    This function requires an initial .para file that will be copied and updated
    with the user's desired parameter values to make a new model. This function fails
    silently (by design) if MCFOST model creation is unsuccessful.
    
    OVERWRITE WARNING! Any existing directory at the new model's location,
        e.g., model_path/fnstring, will be SILENTLY OVERWRITTEN! Period.
    
    Inputs:
        pkeys: array of str names for each model parameter being edited in the *.para
            parameter file. Names MUST match those used in mcfost.paramfiles.Paramfile
            (from the mcfost-python package) and the keys in pl_dict.
        pl_dict: dict of parameter name and value pairs to be used in the new model.
        parfile: str path to a (*.para) MCFOST parameter file that will be used
            as a basis for the new model's parameter file. Any parameters not
            included in pl_dict will remain unchanged.
        model_path: str path to parent directory for new model.
        s_ident: str identifier for the new model. Only used here to name models
            in the fashion "[s_ident]_mcmc_..." if fnstring is None.
        fnstring: str name for the new model. If None, a generic name will be assigned
            using s_ident and the first parameter's name and value from pl_dict.
        lam: float wavelength [microns] at which to compute the MCFOST
             scattered-light model.
        scatlight: bool, True to make a scattered-light-only model at wavelength lam (default).
        fullimg: bool, True to compute the temperature profile, SED, and an image
            at wavelength lam that combines thermal and scattered-light emission.
        sed: bool, True to make an SED model from the new .para file.
            Ignored if fullimg==True.
        dustprops: bool, True to output the dust properties like phase function.
    
    Outputs:
        Technically nothing, but writes MCFOST .para parameter files and models to disk.
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
        fnstring = "%s_mcmc_%s%.5e" % (s_ident, pkeys[0], pl_dict[pkeys[0]])
    
    modeldir = os.path.join(model_path, fnstring)
    try:
        os.mkdir(modeldir)
    except OSError:
# FIX ME!!! This rmtree will cause an OSError if the directory doesn't exist.
# Need a better solution than this try/except block.
        time.sleep(0.5) # short pause
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
        if fullimg:
            subprocess.call('mcfost '+fnstring+'.para >> mcfostout.txt', shell=True)
            subprocess.call('mcfost '+fnstring+'.para -img '+str(lam)+' >> imagemcfostout.txt', shell=True)
        if sed and not fullimg:
            subprocess.call('mcfost '+fnstring+'.para >> sedmcfostout.txt', shell=True)
        if scatlight:
            subprocess.call('mcfost '+fnstring+'.para -img '+str(lam)+' -rt2 -only_scatt >> SLimagemcfostout.txt', shell=True)
        if dustprops:
            subprocess.call('mcfost '+fnstring+'.para -dust_prop -op '+str(lam)+' >> dustmcfostout.txt', shell=True)
        if not (fullimg or scatlight or sed or dustprops):
            print("No model created: neither 'fullimg', 'scatlight', 'sed', nor 'dustprops' flags set to True.")
            pass
    except:
        pass
    
    # cd back into the outer MCMC directory.
    os.chdir(model_path)
    
    return


def chi2_morph(path, data, uncerts, data_types, mod_bin_factor,
               phi_stokes, unit_conv, algo_I=None, modfm=None):
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
            # Total intensity; includes thermal + scattered-light images.
            if dt == 'I':
                mod_use = model[0,0,0,:,:]  # [W/m^2...]
 # FIX ME!!! Total intensity ADI forward modeling is currently disabled.
                if algo_I == 'loci':
                    mod_use = do_fm_loci(dataset, mod_use.copy(), c_list)
                elif algo_I == 'pyklip':
                    # Forward model to match the KLIP'd data.
                    mod_use = do_fm_pyklip(modfm, dataset, model_I.copy())
            
            # Radial Stokes Q polarized intensity (i.e., Q_phi or Qr).
            elif dt == 'Qr':
                mod_use, model_Ur = get_radial_stokes(model[1,0,0,:,:], model[2,0,0,:,:], phi_stokes) # [W/m^2...]
            # Stokes Q polarized intensity.
            elif dt == 'Q':
                mod_use = model[1,0,0,:,:] # [W/m^2...]
            # Stokes U polarized intensity.
            elif dt == 'U':
                mod_use = model[2,0,0,:,:] # [W/m^2...]
            
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
             parfile, model_path, unit_conv, priors, scatlight, fullimg, sed, dustprops,
             lam, partemp, ndim, write_model, s_ident, algo_I=None, modfm=None):
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
        make_mcfmod(pkeys, pl_dict, parfile, model_path, s_ident, fnstring,
                    lam=lam, scatlight=scatlight, fullimg=fullimg)
    except:
        return -np.inf
    
    # Calculate Chi2 for all images in data.
    chi2s = chi2_morph(os.path.join(model_path, fnstring, 'data_%s' % str(lam)),
                        data, uncerts, data_types, mod_bin_factor,
                        phi_stokes, unit_conv, algo_I=algo_I, modfm=modfm)
    # # Calculate reduced Chi2 for images.
    # chi2_red_Qr = chi2_Qr/(np.where(np.isfinite(data_Qr))[0].size - ndim)
    
    if not write_model:
        try:
            rmtree(os.path.join(model_path, fnstring))
        except:
            time.sleep(0.5)
            subprocess.call('rm -rf %s' % os.path.join(model_path, fnstring), shell=True)
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
    
# TEMP!!!
    # For now, emcee v3.* offers only an EnsembleSampler, so force that mode
    # if such a version is detected.
    if (ntemps > 1) and (emcee_version_major >= 3):
        ntemps = 1
        partemp = False
        emcee_v3_msg = "WARNING! FORCED to use EnsembleSampler because emcee v3+ detected. Setting ntemps=1 and partemp=False."
        print("\n" + emcee_v3_msg)
        print("To use a PTSampler, try the ptemcee package (NOT currently compatible with diskmc) or using emcee v2.2.1.")
    else:
        emcee_v3_msg = None
    
    data = mcdata.data
    uncerts = mcdata.uncerts
    data_types = np.array(mcdata.data_types) # need as nd.array for later
    stars = mcdata.stars
    
    model_path = os.path.join(os.path.abspath(os.path.expanduser(mcmod.model_path)), '')
    log_path = os.path.join(os.path.abspath(os.path.expanduser(mcmod.log_path)), '')
    lam = mcmod.lam # [microns]
    unit_conv = mcmod.unit_conv
    
    # Sort the parameter names.
    # NOTE: this must be an array (can't be a list).
    pkeys_all = np.array(sorted(mcmod.pkeys))
    
    # Create log file.
    mcmc_log_fn = os.path.join(log_path, '%s_mcmc_log.txt' % s_ident)
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
        init_samples = hickle.load(os.path.join(log_path, init_samples_fn))
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
    print("Running diskmc version %s" % __version__)
    
    log_preamble = ['|---MCMC LOG---|\n\n', '%s' % s_ident,
                        '\nLOG DATE: ' + dt.date.isoformat(dt.date.today()),
                        '\nJOB START: ' + start,
                        '\nNPROCESSORS: %d\n' % nthreads,
                        '\ndiskmc version: %s' % __version__,
                        '\nMCFOST version: %s' % mcf_version,
                        '\nMCFOST PARFILE: ', mcmod.parfile,
                        '\n\nMCMC PARAMS: Ndim: %d, Nwalkers: %d, Nburn: %d, Niter: %d, Nthin: %d, Nthreads: %d' % (ndim, nwalkers, nburn, niter, nthin, nthreads),
                        '\nPARALLEL-TEMPERED?: ' + str(partemp), ' , Ntemps: %d' % ntemps,
                        '\nINITIALIZATION: ', init_type,
                        '\na = %.2f' % mc_a,
                        '\nWavelength = %s microns' % str(lam),
                        '\n',
                        ]
    mcmc_log.writelines(log_preamble)
    if emcee_v3_msg is not None:
        mcmc_log.writelines('\n{}\n'.format(emcee_v3_msg))
    mcmc_log.close()

    # Create emcee sampler instances: parallel-tempered or ensemble.
    if partemp:
        # Instantiate parallel-tempered sampler.
        # Pass data_I, uncertainty_I, and parfile as arguments to emcee sampler.
        sampler = PTSampler(ntemps, nwalkers, ndim, mc_lnlike, mc_lnprior, a=mc_a,
                      loglargs=[pkeys_all, data, uncerts, data_types, mcmod.mod_bin_factor,
                        phi_stokes, mcmod.parfile, model_path, unit_conv, mcmod.priors,
                        mcmod.scatlight, mcmod.fullimg, mcmod.sed, mcmod.dustprops,
                        lam, partemp, ndim, write_model, s_ident,
                        mcdata.algo_I, mcmod.modfm],
                      logpargs=[pkeys_all, mcmod.priors],
                        threads=nthreads)
                      #pool=pool)
    
    else:
        # Instantiate ensemble sampler.
        if emcee_version_major >= 3:
            # Put backend setup here for some future use.
            # backend_log_name = os.path.join(log_path, '{}_mcmc_full_sampler.h5'.format(s_ident))
            # backend = emcee.backends.HDFBackend(backend_log_name, name='mcmc')
            # backend.reset(nwalkers, ndim)
            # 
            # vars(backend)['pkeys_all'] = pkeys_all
            backend = None
            
            from multiprocessing import Pool
            pool = Pool()
            
            sampler = EnsembleSampler(nwalkers, ndim, mc_lnlike, a=mc_a,
                        args=[pkeys_all, data, uncerts, data_types, mcmod.mod_bin_factor,
                        phi_stokes, mcmod.parfile, model_path, unit_conv, mcmod.priors,
                        mcmod.scatlight, mcmod.fullimg, mcmod.sed, mcmod.dustprops,
                        lam, partemp, ndim, write_model, s_ident,
                        mcdata.algo_I, mcmod.modfm],
                        pool=pool,
                        backend=backend)
            # Add some items to sampler that don't exist in emcee >2.2.1.
            vars(sampler)['_chain'] = np.array([])
            vars(sampler)['_lnprob'] = np.array([])
        else:
            sampler = EnsembleSampler(nwalkers, ndim, mc_lnlike, a=mc_a,
                        args=[pkeys_all, data, uncerts, data_types, mcmod.mod_bin_factor,
                        phi_stokes, mcmod.parfile, model_path, unit_conv, mcmod.priors,
                        mcmod.scatlight, mcmod.fullimg, mcmod.sed, mcmod.dustprops,
                        lam, partemp, ndim, write_model, s_ident,
                        mcdata.algo_I, mcmod.modfm],
                        threads=nthreads)

    # Insert pkeys and priors into the sampler dict for later use.
    # Force '|S' string dtype to avoid unicode (which doesn't hickle well).
    vars(sampler)['pkeys_all'] = pkeys_all.astype('S')
    vars(sampler)['priors'] = mcmod.priors.copy()
    
    # List of items to delete from the sampler dict that may not hickle well during
    # logging. Mix of emcee v2 and v3, Ensemble, and PTSampler items.
    sampler_keys_trim = ['pool', 'lnprobfn', 'log_prob_fn', 'runtime_sortingfn',
                        'logl', 'logp', 'logpkwargs', 'loglkwargs', 'args',
                        'kwargs', '_random', '_moves', '_previous_state', 'backend']


    ###############################
    # ------ BURN-IN PHASE ------ #
    if nburn > 0:
        print("\nBURN-IN START...\n")
        for bb, (pburn, lnprob_burn, lnlike_burn) in enumerate(sampler.sample(p0, iterations=nburn)):
            # Print progress every 25%.
            if bb in [nburn//4, nburn//2, 3*nburn//4]:
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

        # Pause interactively between burn-in and main phase.
        # Comment this out if you don't want the script to pause here.
        # pdb.set_trace()

        # Reset the sampler chains and lnprobability after burn-in.
        sampler.reset()

        print("BURN-IN COMPLETE!")

    elif (nburn==0) & (init_samples_fn is not None):
        print("\nWalkers initialized from file and no burn-in samples requested.")
        sampler.reset()
        pburn = p0
        lnprob_burn = None #lnprob_init 
        lnlike_burn = None #lnlike_init
    else:
        print("\nNo burn-in samples requested.")
        pburn = p0
        lnprob_burn = None
        lnlike_burn = None
    
    
    ############################
    # ------ MAIN PHASE ------ #
    
    print("\nMAIN-PHASE MCMC START...\n")
    if partemp:
        for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn, lnprob0=lnprob_burn, lnlike0=lnlike_burn, iterations=niter)):
            # Print progress every 25%.
            if nn in [niter//4, niter//2, 3*niter//4]:
                print("Processing iteration %d; MCMC %.1f%% complete..." % (nn, 100*float(nn)/niter))
                if emcee_version_major < 3:
                    # Log the full sampler or chain (all temperatures) every so often.
                    log_message = log_sampler(sampler, sampler_keys_trim, log_path, s_ident, nn)
    else:
        for nn, (pp, lnprob, lnlike) in enumerate(sampler.sample(pburn, lnprob_burn, iterations=niter)):
            # Print progress every 25%.
            if nn in [niter//4, niter//2, 3*niter//4]:
                print("Processing iteration %d; MCMC %.1f%% complete..." % (nn, 100*float(nn)/niter))
                # Log the full sampler or chain (all temperatures) every so often.
                log_message = log_sampler(sampler, sampler_keys_trim, log_path, s_ident, nn)
    
    
    print('\nMCMC RUN COMPLETE!\n')
    
    
    ##################################
    # ------ RESULTS HANDLING ------ #
    # Log the sampler output and chains. Also get the max and median likelihood
    # parameter values and save models for them.
    
    # If possible, save the whole sampler to an HDF5 log file (could be large).
    # If that fails because hickle won't handle something in the sampler,
    # try to pickle it instead. If that still fails, just log the sampler chains.
    # if emcee_version_major < 3:
    log_message = log_sampler(sampler, sampler_keys_trim, log_path, s_ident, 'FINAL')


    # Chain has shape (ntemps, nwalkers, nsteps/nthin, ndim).
    if partemp:
        assert sampler.chain.shape == (ntemps, nwalkers, niter/nthin, ndim)
    else:
        assert sampler.chain.shape == (nwalkers, niter/nthin, ndim)
    
    # Re-open the text log for additional info.
    mcmc_log = open(mcmc_log_fn, 'a')
    mcmc_log.writelines(['\n' + log_message,
                        '\n\n|--- RESULTS FOR ALL ITERATIONS (NO BURN-IN EXCLUDED) ---|',
                        '\n'])
    
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
    
    # Get median values (50th percentile) and 1-sigma (68%) confidence intervals
    # for each parameter (in order +, -).
    params_med_mcmc = list(map(lambda vv: (vv[1], vv[2]-vv[1], vv[1]-vv[0]),
                        zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    
    print("\nMax-Likelihood Param Values:")
    mcmc_log.writelines('\n\nMAX-LIKELIHOOD PARAM VALUES:')
    for kk, key in enumerate(pkeys_all):
        print(key + ' = %.3e' % params_ml_mcmc[key])
        mcmc_log.writelines('\n%s = %.3e' % (key, params_ml_mcmc[key]))

    print("\n50%-Likelihood Param Values (50th percentile +/- 1 sigma (i.e., 34%):")
    mcmc_log.writelines('\n\n50%-LIKELIHOOD PARAM VALUES (50th percentile +/- 1 sigma (i.e., 34%):')
    for kk, key in enumerate(pkeys_all):
        print(key + ' = %.3f +/- %.3f/%.3f' % (params_med_mcmc[kk][0], params_med_mcmc[kk][1], params_med_mcmc[kk][2]))
        mcmc_log.writelines('\n%s = %.3f +/- %.3f/%.3f' % (key, params_med_mcmc[kk][0], params_med_mcmc[kk][1], params_med_mcmc[kk][2]))
    
    
    # Construct max- and med-likelihood models.
    print("\nConstructing 'best-fit' models...")
    mod_idents = ['maxlk', 'medlk']
    params_50th_mcmc = np.array(params_med_mcmc)[:,0]
    
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
        make_mcfmod(pkeys_all, pl_dict, mcmod.parfile, model_path, s_ident,
                    fnstring, lam=lam, scatlight=mcmod.scatlight, fullimg=mcmod.fullimg)
        
        # Calculate Chi2 for images.
        chi2s = chi2_morph(os.path.join(model_path, fnstring, 'data_%s' % str(lam)),
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
        
        # Make image, sed, and/or dust properties models for maxlk and medlk.
        try:
            os.chdir(os.path.join(model_path, fnstring))
            # Make the dust properties at proper wavelength.
            # NOTE: This must come before the image calculation at the same
            # wavelength, otherwise MCFOST automatically deletes the image directory!
            subprocess.call('rm -rf data_'+str(lam), shell=True)
            subprocess.call('mcfost '+fnstring+'.para -dust_prop -op %s >> dustmcfostout.txt' % lam, shell=True)
            time.sleep(1)
            # # Delete the (mostly) empty data_[lam] directory after dust_prop step.
            # subprocess.call('rm -rf data_'+str(lam), shell=True)
            # Make the SED model.
            subprocess.call('mcfost '+fnstring+'.para -rt >> sedmcfostout.txt', shell=True)
            # Make the image models (thermal + scattered-light) at demanded wavelength.
            subprocess.call('mcfost '+fnstring+'.para -img '+str(lam)+' -rt2 >> imagemcfostout.txt', shell=True)
            time.sleep(1)
            print("Saved image and dust properties models.")
        except:
            print("Failed to save image and dust properties models.")
    
    # Plot and save maxlk and medlk models.
    try:
        # Placeholder for plotting functions.
        pass
        print("Max and Median Likelihood models made, plotted, and saved.\n")
    except:
        print("Max and Median Likelihood models made and saved but plotting failed.\n")
    
    
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
        print("\nPool closed")
    except:
        print("\nNo Pool to close.")
    
    print("\nmc_main function finished\n")
    
    # # Pause interactively before finishing script.
    # pdb.set_trace()
    
    return

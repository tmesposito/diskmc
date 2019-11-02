#!/usr/bin/env python

import os
import argparse
import warnings
try:
    import ipdb as pdb
except:
    import pdb
#import time, datetime as dt
import gzip
import numpy as np
from astropy.io import fits
from astropy import constants

from diskmc_tools import make_radii, get_ann_stdmap
from diskmc import mc_main, MCMod, MCData

# !!!WARNING!!! Ignoring some annoying warnings about FITS files.
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


# |----- COMMAND LINE ARGUMENTS -----| #

# Passing in arguments like this allows you to make most of the common
# changes to the MCMC straight from the command line Python call.
# Alternatively, you could remove these arguments and just feed the required
# variables straight into diskmc.mc_main; however, this would mean editing
# hard-coded values every time you want to change the number of walkers,
# number of iterations, etc.

# Call script as: python run_diskmc.py s_ident ntemps nwalkers niter nburn [nthin] [mc_a] [--init_samples_fn]
parser = argparse.ArgumentParser(description='MCMC script for disk modeling with MCFOST')
# Parse MCMC settings variables.
parser.add_argument('s_ident')
parser.add_argument('ntemps', type=int, default=10)
parser.add_argument('nwalkers', type=int, default=100) # MUST BE EVEN; 100 is recommended minimum
parser.add_argument('niter', type=int, default=500)
parser.add_argument('nburn', type=int, default=150) # 0 for no burn-in
parser.add_argument('nthin', type=int, nargs='?', default=1) # no thinning by default
parser.add_argument('mc_a', type=float, nargs='?', default=2.0) # 2.0 is emcee default
parser.add_argument('--init_samples_fn', type=str, nargs='?', default=None) # str filename for optional full_sampler log to use to initialize walkers

args = parser.parse_args()

# Turn namespace items into local variables.
locals().update(vars(args))

# Set flag to use the Parallel-Tempered sampler if ntemps > 1.
# Otherwise, use the Ensemble sampler.
if ntemps > 1:
    partemp = True
else:
    partemp = False


# |----- PATHS -----| #

# CUSTOMIZE these paths to match your personal directory setup.

model_path = os.path.expanduser('~/Desktop/test_dir/diskmc_%s/' % s_ident)
log_path = os.path.expanduser('~/Desktop/test_dir/diskmc_logs/')

# Path to the MCFOST parameter file that will serve as a template for all
# MCMC models to be built from.
parfile = os.path.expanduser('~/Desktop/test_dir/diskmc_init_%s.para' % s_ident)

# Make a subdirectory for the mcfost models if they will be saved.
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if not os.path.isdir(model_path + "plots/"):
    os.makedirs(model_path + "plots/")
if not os.path.isdir(log_path):
    os.makedirs(log_path)


# |----- DATASET PARAMETERS -----| #

# CUSTOMIZE these parameters to your data.

# skyPA = 74*np.pi/180. # [rad]
pa_list = []
pscale = 0.014166 # [arcsec/pix]
dist = 102.9 # [pc]

# Coordinates of the star in the data.
star = np.array([140, 140]) # [pix] (y,x)
# algo = 'loci'
# algo = 'pyklip'

# Define region of data-model residuals used for goodness-of-fit measurement.
# hw_y is the distance considered on either side of star along y-axis.
hw_y = 80
hw_x = 40
# r_fit defines the inner radial distance from the star considered.
r_fit = 14

# Wavelength at which models are computed, for unit conversion.
lam = 1.647 # [microns]
# Conversion factor from MCFOST's output W/m/Hz to Jy/arcsec^2.
nu = constants.c.value/(1e-6*lam) # [Hz]
conv_WtoJy = (1e3/pscale**2)*1.e26/nu # [(mJy arcsec^-2) / (W m^-2)]

# Option to spatially bin data and models by some factor.
# Typically, this factor should be chosen such that one resolution element
# is transformed into one pixel in the binned image.
bin_factor = 2.


# |----- LOAD DATA -----| #

# CUSTOMIZE this section to your data.

# Load radial Stokes Q data from FITS file.
data_rstokes = fits.getdata(os.path.expanduser('~/path/to/data/example_gpi_rstokes_datacube.fits'))
data_Qr = data_rstokes[1] # radial Stokes Q channel
data_Ur = data_rstokes[2] # radial Stokes U channel

# Array of radial distances from star in data_Qr.
radii = make_radii(data_Qr, star)

# Calculate noise map for data_Qr from the Ur channel.
# This assumes an optically thin disk in which only single scattering
# events occur. As such, Ur should contain only noise equivalent to the noise in Qr.
uncertainty_Qr = get_ann_stdmap(data_Ur, star, radii, r_max=135)
uncertainty_Qr[uncertainty_Qr==0] = np.nan

# Make mask for data-model comparison, excluding large empty regions and edges.
mask_fit = np.ones(data_Qr.shape).astype(bool)
mask_fit[star[0]-hw_y:star[0]+hw_y+1, star[1]-hw_x:star[1]+hw_x+1] = False
mask_fit[radii < r_fit] = True # mask out the pixels at edge of mask b/c noisy
# Mask the data in a new masked array.
data_Qr_masked = np.ma.masked_array(data_Qr, mask_fit)


# Create a handy MCData object to hold all data and associated variables.
data_info = MCData([data_Qr_masked], ['Qr'], [star], [uncertainty_Qr], [bin_factor],
                    [(hw_y, hw_x, r_fit)], s_ident)


# |----- MCMC SETUP -----| #

# Flag to delete (False) or save (True) every MCFOST model after sampled.
write_model = False

# Factor by which to bin the model images.
mod_bin_factor = bin_factor

# Set number of parallel threads to which emcee will assign walkers.
nthreads = 4
# Set number of threads that each MCFOST process will use.
omp_nthreads = 2
try:
    os.environ["OMP_NUM_THREADS"] = str(omp_nthreads) # force MCFOST to use omp_nthreads threads per process
    print("You've set OMP_NUM_THREADS = %d for MCFOST (not persistent after MCMC)" % omp_nthreads)
except:
    print("Unable to set OMP_NUM_THREADS = %d ; defaulting to the current shell value." % omp_nthreads)


# |----- MCMC INITIALIZATION PARAMETERS & PRIORS -----| #

# Set types of models to create at each iteration.
scatlight = True  # Make scattered-light-only image models?
fullimg = False  # Make thermal emission + scattered-light image models? Also makes the SED.
sed = False  # Make the SED? Ignored if fullimg==True.
dustprops = False  # Make the dust properties?

# NOTE: Parameter key names MUST match those used in diskmc.make_mcfmod().
# amin and dust_mass are log parameters, so state them here as np.log10(amin), ...

# # For Gaussian initialization, set the mean value for each parameter.
# pmeans_lib = dict(aexp=3.6, amin=0.6, #debris_disk_vertical_profile_exponent=0.92,
#                 dust_mass=-6.8,
#                 # dust_pop_0_mass_fraction=0.33, dust_pop_1_mass_fraction=0.33,
#                 # dust_pop_2_mass_fraction=0.33, gamma_exp=1.0, inc=82.1,
#                 # porosity=0.01, r_in=44.0, scale_height=4.0, surface_density_exp=0.7)
#                 )
# 
# # For Gaussian initialization, set the sigma value for each parameter.
# psigmas_lib = dict(aexp=1.0, amin=0.2, #debris_disk_vertical_profile_exponent=0.4,
#                    dust_mass=0.4,
#                 # dust_pop_0_mass_fraction=0.2, dust_pop_1_mass_fraction=0.2,
#                 # dust_pop_2_mass_fraction=0.2, gamma_exp=0.5, inc=2.,
#                 # porosity=0.3, r_in=10.0, scale_height=1.0, surface_density_exp=0.5)
#                 )
pmeans_lib = None
psigmas_lib = None

# For a uniform initialization, set Min/Max value limits for parameters.
# These MUST fall within the prior boundaries, or walkers will become
# zombies that never leave a lnprob == NaN condition.
plims_lib = dict(aexp=(2.1, 6.4), amin=(-0.9, 1.5),
                # debris_disk_vertical_profile_exponent=(0.11, 3.),
                dust_mass=(-8.7, -5.9), #dust_pop_0_mass_fraction=(0.002, 1.),
                # dust_pop_1_mass_fraction=(0.002, 1.), dust_pop_2_mass_fraction=(0.002, 1.),
                #  gamma_exp=(-2.99, 3.0), inc=(76., 86.),
                # porosity=(0.002, 0.95), r_in=(10.1, 53.),
                # scale_height=(0.31, 15.), surface_density_exp=(-2.99, 3.0))
                )

# Set flat prior bounds for parameters (lnprob = -inf outside of bounds).
# Tuple pairs are exclusive lower and upper bounds (i.e. 0.1 < amin < 10.).
# NOTE: The priors should be inclusive of any initialization range above;
# any walker initialized and rejected by the prior may become a zombie that
# never leaves a lnprob == NaN condition.
priors = dict(aexp=(2.0, 6.5), amin=(-1.0, 1.6),
                # debris_disk_vertical_profile_exponent=(0.11, 3.),
                dust_mass=(-8.8, -6.0), #dust_pop_0_mass_fraction=(0.002, 1.),
                # dust_pop_1_mass_fraction=(0.002, 1.), dust_pop_2_mass_fraction=(0.002, 1.),
                #  gamma_exp=(-2.99, 3.0), inc=(76., 86.),
                # porosity=(0.002, 0.95), r_in=(10.1, 53.),
                # scale_height=(0.31, 15.), surface_density_exp=(-2.99, 3.0))
                )


# Create a handy MCMod instance to hold basic model info.
mod_info = MCMod(plims_lib.keys(), parfile, pmeans_lib, psigmas_lib, plims_lib, priors,
                scatlight, fullimg, sed, dustprops,
                lam, conv_WtoJy, mod_bin_factor, model_path, log_path, s_ident)


# |----- RUN THE MCMC -----| #
if __name__=='__main__':
    mc_main(s_ident, ntemps, nwalkers, niter, nburn, nthin, nthreads,
            mcdata=data_info, mcmod=mod_info, partemp=partemp, mc_a=mc_a,
            init_samples_fn=init_samples_fn, write_model=write_model, plot=False,
            save=False)


# print("run_diskmc.py script finished\n")

# # Pause interactively before finishing script.
# pdb.set_trace()

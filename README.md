# DiskMC

A Python framework for running a Markov Chain Monte Carlo (MCMC) with models of circumstellar disks produced by the MCFOST Monte Carlo and ray-tracing radiative transfer code.

Note that this package does not include MCFOST itself (Pinte et al. 2006, 2009) and will not run without it.

DiskMC is partly based on [`mcfost-python`](https://github.com/mperrin/mcfost-python). In particular, it directly incoporates `paramfiles.py` from that package to write parameter files (credit for that bit to the mcfost-python developers). A newer version of MCFOST<->Python interaction tools is being developed at https://github.com/cpinte/pymcfost.

## Documentation ##

**_Python 2.7 and 3.x compatible._**
To run a parallel-tempered MCMC, you will need `emcee` v2.2.1. Single temperature Ensemble samplers will work with v2.2.1 or 3.0.

_More extensive documentation is to come. The source code is heavily commented and is the best resource at the moment._

The core of DiskMC is in `diskmc.py`, where `diskmc.mc_main` is the function to call the MCMC itself. You can call diskmc.py directly as a script to initiate an MCMC, ideally with a preparatory wrapper script like the `run_diskmc.py` example. It is there that you can specify input data, MCFOST variables to vary, MCMC priors, data masks, and more.

### MCMC Visualization ###

Some basic visualization tools are provided in `diskmc_plot.py` to analyze the MCMC output. With `diskmc_plot.mc_analyze` you can get posterior distributions, plot walker chains, plot time-step histograms, and make corner plots.

### Test on Example Data ###

Example input files for testing are included in the `test_dir` directory. An example script to load those data, set MCMC parameters, and start a short run is provided in `run_diskmc.py`. The example command to start an MCMC from the command line is, to be run from within the `diskmc/` directory, is:

`python run_diskmc.py example0 2 20 10 5`

This will start an MCMC named "example0" that uses 2 parallel temperatures, 20 walkers, 10 iterations, and 5 burn-in iterations. It varies 3 MCFOST model parameters: minimum grain size "amin", grain size distribution power law index "aexp", and log10 dust mass "dust_mass". This is an extremely short run as far as MCMC's are considered, and just serves to get you started. See the `run_diskmc.py` source code for explanations of the input arguments and other options not shown here.

Output from the test should be a .txt log file and an .hkl file of the HDF5 compressed emcee sampler components in `diskmc_logs/`. There should also be a new directory named `diskmc_example0` that contains subdirectories with FITS cubes ("RT.fits.gz") of the maximum-likelihood ("maxlk") and median-likelihood ("medlk") models resulting from the MCMC.

To visualize the results for the zeroth temperature walkers, the function call from an interactive Python session is:

```
from diskmc.diskmc_plot import mc_analyze
mc_analyze(s_ident='example0', path='test_dir/diskmc_logs/', ntemp_view=0, partemp=True)
```

NOTE: hickle will usually only load .hkl files that were created with a similar version; e.g., using hickle+Python3 to load a sampler saved with hickle+Python2.7 often fails. The example sampler in test_dir was created with Python2.7, so you will likely need to use that to open it.

## Attribution ##

For now, please cite Esposito et al. (2018) if you find this code useful in your research, as that is the first paper to use this specific framework. The BibTeX entry for the paper is:

    @article{esposito2018,
        Adsurl = {http://adsabs.harvard.edu/abs/2018AJ....156...47E},
        Author = {{Esposito}, T.~M. and {Duch{\^e}ne}, G. and {Kalas}, P. and {Rice}, M. and {Choquet}, {\'E}. and {Ren}, B. and {Perrin}, M.~D. and {Chen}, C.~H. and {Arriaga}, P. and {Chiang}, E. and {Nielsen}, E.~L. and {Graham}, J.~R. and {Wang}, J.~J. and {De Rosa}, R.~J. and {Follette}, K.~B. and {Ammons}, S.~M. and {Ansdell}, M. and {Bailey}, V.~P. and {Barman}, T. and {Sebasti{\'a}n Bruzzone}, J. and {Bulger}, J. and {Chilcote}, J. and {Cotten}, T. and {Doyon}, R. and {Fitzgerald}, M.~P. and {Goodsell}, S.~J. and {Greenbaum}, A.~Z. and {Hibon}, P. and {Hung}, L.-W. and {Ingraham}, P. and {Konopacky}, Q. and {Larkin}, J.~E. and {Macintosh}, B. and {Maire}, J. and {Marchis}, F. and {Marois}, C. and {Mazoyer}, J. and {Metchev}, S. and {Millar-Blanchaer}, M.~A. and {Oppenheimer}, R. and {Palmer}, D. and {Patience}, J. and {Poyneer}, L. and {Pueyo}, L. and {Rajan}, A. and {Rameau}, J. and {Rantakyr{\"o}}, F.~T. and {Ryan}, D. and {Savransky}, D. and {Schneider}, A.~C. and {Sivaramakrishnan}, A. and {Song}, I. and {Soummer}, R. and {Thomas}, S. and {Wallace}, J.~K. and {Ward-Duong}, K. and {Wiktorowicz}, S. and {Wolff}, S.},
        Doi = {10.3847/1538-3881/aacbc9},
        Eid = {47},
        Journal = {\aj},
        Keywords = {circumstellar matter, infrared: planetary systems, stars: individual: HD 35841, techniques: high angular resolution},
        Month = aug,
        Pages = {47},
        Title = {{Direct Imaging of the HD 35841 Debris Disk: A Polarized Dust Ring from Gemini Planet Imager and an Outer Halo from HST/STIS}},
        Volume = 156,
        Year = 2018,
        Bdsk-Url-1 = {https://doi.org/10.3847/1538-3881/aacbc9}
	}

## License ##

Copyright 2018 Tom Esposito.

DiskMC is free software made available under the GNU General Public License. For details see the LICENSE file.

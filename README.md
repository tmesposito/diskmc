# DiskMC

A Python framework for running a Markov Chain Monte Carlo (MCMC) with models of circumstellar disks produced by the MCFOST Monte Carlo and ray-tracing radiative transfer code.

Note that this package does not include MCFOST itself (Pinte et al. 2006, 2009) and will not run without it.

DiskMC is partly based on [`mcfost-python`](https://github.com/cpinte/mcfost-python) and is still dependent on some of its components.

## Documentation ##

**_Only Python 2.7 compatible._** Python 3 support is on its way.

_More extensive documentation is to come. The source code is heavily commented and is the best resource at the moment._

The core of DiskMC is in `diskmc.py`, where `diskmc.mc_main` is the function to call the MCMC itself. An example script to load data, set MCMC parameters, and start a run is provided in `run_diskmc.py`. Currently, that script loads external data files that are not provided here, so those (two or three) specific paths will need to be edited for it to run on your system. Nevertheless, an example command to start an MCMC from the command line would be:

`python run_diskmc.py example0 2 20 10 5`

This will start an MCMC named "example0" that uses 2 parallel temperatures, 20 walkers, 10 iterations, and 5 burn-in iterations. This is an extremely short run as far as MCMC's are considered, and just serves to get you started. See the `run_diskmc.py` source code for explanations of the input arguments and other options not shown here.

### MCMC Visualization ###

Some basic visualization tools are provided in `diskmc_plot.py` to analyze the MCMC output. With `diskmc_plot.mc_analyze` you can get posterior distributions, plot walker chains, plot time-step histograms, and make corner plots. An example function call to do this for the MCMC above would be:

```
from diskmc_plot import mc_analyze
mc_analyze(s_ident='example0', path='~/Desktop/test_dir/diskmc_logs/', ntemp_view=0, partemp=True)
```

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

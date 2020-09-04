#!/usr/bin/env python3
"""
Generate simulations of a slightly non-redundant array using hera_sim.
"""
from mpi4py import MPI
import numpy as np

import uvtools
import hera_cal as hc
import pyuvdata
from pyuvdata import UVData

from hera_sim.visibilities import VisCPU, conversions
from hera_sim.beams import PolyBeam,PerturbedPolyBeam

import utils
import time, copy, sys


def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation specification
    cfg_spec = dict( nfreq=20,
                     start_freq=1.0e+8,
                     bandwidth=0.2e+8,
                     start_time=2458902.33333,
                     integration_time=40.,
                     ntimes=40,
                     cat_name="gleamegc.dat",
                     apply_gains=True,
                     apply_noise=True,
                     ant_pert=False,
                     seed=None,
                     ant_pert_sigma=0.0)
                        
    # Beam model parameters
    cfg_beam = dict( ref_freq=1.e+8,
                     spindex=-0.6975,
                     seed=None,
                     perturb_scale=0.0,
                     mainlobe_scale_mean=1.0,
                     mainlobe_scale_sigma=0.0,
                     xstretch_mean=1.0,
                     xstretch_sigma=0.0,
                     ystretch_mean=1.0,
                     ystretch_sigma=0.0,
                     xystretch_same=True,
                     xystretch_dist=None,
                     rotation_dist=None,
                     rotation_mean=0.0,
                     rotation_sigma=0.0,
                     mainlobe_width=0.3, 
                     nmodes=8,
                     beam_coeffs=[0.29778665, -0.44821433,  0.27338272, -0.10030698, -0.01195859,
                                  0.06063853, -0.04593295,  0.0107879 ,  0.01390283, -0.01881641,
                                 -0.00177106,  0.01265177, -0.00568299, -0.00333975,  0.00452368,
                                  0.00151808, -0.00593812,  0.00351559
                                ] )
    
    # Fluctuating gain model parameters
    cfg_gain = dict(nmodes=8, seed=None)
    
    # Noise parameters
    cfg_noise = dict(nsamp=1.0, seed=None, noise_file=None)
    
    # Combine into single dict
    cfg = { 'sim_beam':   cfg_beam,
            'sim_spec':   cfg_spec,
            'sim_noise':  cfg_noise,
            'sim_gain':   cfg_gain,
           }
    return cfg


if __name__ == '__main__':
    # Run simulations (MPI-enabled)    

    # Get config file name from args
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
    else:
        print("Usage: analyse_sims.py config_file")
        sys.exit(1)
    
    # Begin MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    
    # Load config file
    cfg = utils.load_config(config_file, default_cfg())
    cfg_spec = cfg['sim_spec']
    cfg_out = cfg['sim_output']
    cfg_beam = cfg['sim_beam']
    cfg_gain = cfg['sim_gain']
    cfg_noise = cfg['sim_noise']
    
    
    # Construct array layout to simulate
    ants = utils.build_array()
    Nant = len(ants)
    ant_index = list(ants.keys())

    #perturb antenna position
    if cfg_spec['ant_pert']:
        np.random.seed(cfg_spec['seed'])
        for i in range(Nant):
            ants[i] = tuple(list(ants[i]) + (cfg_spec['ant_pert_sigma'] * np.random.randn(3))) #3 for x,y,z
    
    # Build empty UVData object with correct dimensions
    uvd = utils.empty_uvdata(ants=ants, **cfg_spec)

    # Create frequency array
    freq0 = 100e6
    freqs = np.unique(uvd.freq_array)
    ra_dec, flux = utils.load_ptsrc_catalog(cfg_spec['cat_name'], 
                                            freq0=freq0, freqs=freqs, usecols=(0,1,2,3))

    # Build list of beams using Best fit coeffcients for Chebyshev polynomials
    if cfg_beam['perturb']:

        mainlobe_scale_mean = cfg_beam['mainlobe_scale_mean']
        mainlobe_scale_sigma = cfg_beam['mainlobe_scale_sigma']
        xstretch_mean = cfg_beam['xstretch_mean']
        xstretch_sigma = cfg_beam['xstretch_sigma']
        ystretch_mean = cfg_beam['ystretch_mean']
        ystretch_sigma = cfg_beam['ystretch_sigma']
        rotation_mean = cfg_beam['rotation_mean']
        rotation_sigma = cfg_beam['rotation_sigma']

        np.random.seed(cfg_beam['seed'])

        #to perturb mainlobe
        mainlobe_scale = mainlobe_scale_sigma * np.random.randn(Nant) + mainlobe_scale_mean

        #to perturb xstretch and ystretch
        xstretch = np.full(Nant,xstretch_mean)
        ystretch = np.full(Nant,ystretch_mean)
        
        if cfg_beam['xystretch_dist']=='Gaussian':
            xstretch = xstretch_sigma * np.random.randn(Nant) + xstretch_mean
            if cfg_beam['xystretch_same']==True:
                ystretch = xstretch
            else:
                ystretch = ystretch_sigma * np.random.randn(Nant) + ystretch_mean
    
        if cfg_beam['xystretch_dist']=='Uniform': 
            xstretch = np.random.uniform(-2.*xstretch_sigma,2.*xstretch_sigma,Nant) + xstretch_mean
            if cfg_beam['xystretch_same']==True:
                ystretch = xstretch
            else:
                ystretch = np.random.uniform(-2.*ystretch_sigma,2.*ystretch_sigma,Nant) + ystretch_mean

        if cfg_beam['xystretch_dist']=='Outlier':
            xstretch[cfg_beam['outlier_ant_id']] = cfg_beam['outlier_xstretch']
            ystretch = xstretch

        #to perturb rotation
        rotation = np.zeros(Nant)
        if cfg_beam['rotation_dist']=='Gaussian': 
            rotation = rotation_sigma * np.random.randn(Nant) + rotation_mean

        if cfg_beam['rotation_dist']=='Uniform':
            rotation = np.random.uniform(0.,360.,Nant)

        #to perturb sidelobe and other perturbation
        beam_list = [PerturbedPolyBeam(np.random.randn(cfg_beam['nmodes']),
                                       mainlobe_scale= mainlobe_scale[i],
                                       xstretch=xstretch[i], ystretch=ystretch[i],
                                       rotation=rotation[i],
                                        **cfg_beam) for i in range(Nant)]
    else:
        beam_list = [PolyBeam(**cfg_beam) for i in range(Nant)]


    # Create VisCPU visibility simulator object (MPI-enabled)
    simulator = VisCPU(
        uvdata=uvd,
        beams=beam_list,
        beam_ids=ant_index,
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        precision=2,
        use_pixel_beams=False, # Do not use pixel beams
        bm_pix=10,
        mpi_comm=comm
    )
        
    # Run simulation
    tstart = time.time()
    simulator.simulate()
    print("Simulation took %2.1f sec" % (time.time() - tstart))
    
    if myid != 0:
        # Wait for root worker to finish IO before quitting
        comm.Barrier()
        sys.exit(0)

    # Write simulated data to file
    uvd = simulator.uvdata
    if cfg_out['datafile_true'] != '':
        uvd.write_uvh5(cfg_out['datafile_true'], clobber=cfg_out['clobber'])
    
    # Add noise
    if cfg_spec['apply_noise']:
        uvd = utils.add_noise_from_autos(uvd, input_noise=cfg_noise['noise_file'], 
                                         nsamp=cfg_noise['nsamp'], 
                                         seed=cfg_noise['seed'], inplace=True)
        if cfg_out['datafile_post_noise'] != '':
            uvd.write_uvh5(cfg_out['datafile_post_noise'], 
                           clobber=cfg_out['clobber'])
    
    # Add fluctuating gain model if requested
    if cfg_spec['apply_gains']:
        
        # Generate fluctuating gain model
        nfreqs = cfg_spec['nfreq']
        gg = utils.generate_gains(Nant, nfreqs, **cfg['sim_gain'])
        if cfg_out['gain_file'] != '':
            utils.save_simulated_gains(uvd, gg, 
                                       outfile=cfg_out['gain_file'], 
                                       overwrite=cfg_out['clobber'])
        
        # Loop over all baselines and apply gain factor
        for bl in np.unique(uvd.baseline_array):
            
            # Calculate product of gain factors (time-indep. for now)
            ant1, ant2 = uvd.baseline_to_antnums(bl)
            gigj = gg[ant1] * gg[ant2].conj()
            
            # Get index in data array
            idxs = uvd.antpair2ind((ant1, ant2))
            dtype = uvd.data_array.dtype
            uvd.data_array[idxs,0,:,0] *= np.atleast_2d(gigj.astype(dtype))
        
        # Output gain-multiplied data if requested
        uvd.write_uvh5(cfg_out['datafile_post_gains'], 
                       clobber=cfg_out['clobber'])
    
    # Sync with other workers and finalise
    comm.Barrier()
    sys.exit(0)

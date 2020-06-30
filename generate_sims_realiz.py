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
                     apply_noise=True )
                        
    # Beam model parameters
    cfg_beam = dict( ref_freq=1.e+8,
                     spindex=-0.6975,
                     seed=None,
                     sigma=0.05,
                     mainlobe_scale=1.0,
                     mainlobe_width=0.3, 
                     nmodes=8,
                     beam_coeffs=[
                               2.35088101e-01, -4.20162599e-01,  2.99189140e-01, 
                              -1.54189057e-01,  3.38651457e-02,  3.46936067e-02, 
                              -4.98838130e-02,  3.23054464e-02, -7.56006552e-03, 
                              -7.24620596e-03,  7.99563166e-03, -2.78125602e-03,
                              -8.19945835e-04,  1.13791191e-03, -1.24301372e-04, 
                              -3.74808752e-04,  1.93997376e-04, -1.72012040e-05 
                                ] )
    
    # Fluctuating gain model parameters
    cfg_gain = dict(nmodes=8, seed=None)
    
    # Noise parameters
    cfg_noise = dict(nsamp=1.0, seed=None, noise_file=None)
    
    #relization parameters
    cfg_realiz = dict(nsample=1)
    
    # Combine into single dict
    cfg = { 'sim_beam':   cfg_beam,
            'sim_spec':   cfg_spec,
            'sim_noise':  cfg_noise,
            'sim_gain':   cfg_gain,
            'sim_realiz': cfg_realiz,
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
    cfg_realiz = cfg['sim_realiz']

    # Construct array layout to simulate
    ants = utils.build_array()
    Nant = len(ants)
    ant_index = list(ants.keys())


    true_data = cfg['sim_realiz']['true_data']
    uvd_in = UVData()
    uvd_in.read_uvh5(true_data)

    for i in range(cfg_realiz['nsample']):
        
        cfg_noise['seed'] += 1
        cfg['sim_gain']['seed'] += 1
        str_gain_file = cfg_out['gain_file']+'_'+str(i)
        str_datafile_post_gains = cfg_out['datafile_post_gains']+'_'+str(i)

        # Add noise
        if cfg_spec['apply_noise']:
            uvd = utils.add_noise_from_autos(uvd_in, input_noise=cfg_noise['noise_file'], 
                                             nsamp=cfg_noise['nsamp'], 
                                             seed=cfg_noise['seed'], inplace=False)

        # Add fluctuating gain model if requested
        if cfg_spec['apply_gains']:
        
            # Generate fluctuating gain model
            nfreqs = cfg_spec['nfreq']
            gg = utils.generate_gains(Nant, nfreqs, **cfg['sim_gain'])
            if cfg_out['gain_file'] != '':
                utils.save_simulated_gains(uvd, gg, 
                                               outfile=str_gain_file, 
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
        uvd.write_uvh5(str_datafile_post_gains, 
                       clobber=cfg_out['clobber'])

    # Sync with other workers and finalise
    comm.Barrier()
    sys.exit(0)
        

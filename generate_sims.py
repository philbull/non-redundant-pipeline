#!/usr/bin/env python3
"""
Generate simulations of a non-redundant array using hera_sim.
"""
from mpi4py import MPI
import numpy as np

import uvtools
import hera_cal as hc
import pyuvdata

from hera_sim.visibilities import VisCPU, conversions
from hera_sim.beams import PolyBeam,PerturbedPolyBeam

import utils
import time, copy, sys

import matplotlib.pyplot as plt
def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation parameters
    cfg_sim = dict( ref_freq = 1.e+8,
                    beam_coeffs=[
                              2.35088101e-01, -4.20162599e-01,  2.99189140e-01, 
                             -1.54189057e-01,  3.38651457e-02,  3.46936067e-02, 
                             -4.98838130e-02,  3.23054464e-02, -7.56006552e-03, 
                             -7.24620596e-03,  7.99563166e-03, -2.78125602e-03,
                             -8.19945835e-04,  1.13791191e-03, -1.24301372e-04, 
                             -3.74808752e-04,  1.93997376e-04, -1.72012040e-05 
                               ],
                    spindex=-0.6975,
                    seed=11,
                    sigma=0.05,
                    mainlobe_scale=1.0,
                    mainlobe_width=0.3, 
                    nmodes=8,)

    cfg_sim_spec = dict(nfreq=20,
                        start_freq=1.0e+8,
                        bandwidth=0.2e+8,
                        start_time=2458902.33333,
                        integration_time=40.,
                        ntimes=40,)

    cfg_sim_gain = dict(nmodes=8,
                         seed=10,)

    # Combine into single dict
    cfg = { 'simulation': cfg_sim,
            'sim_spec':   cfg_sim_spec,
            'sim_gain':   cfg_sim_gain,
           }
    return cfg

if __name__ == '__main__':
    # Run simulations
    

    # Get config file name
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
    else:
        print("Usage: analyse_sims.py config_file")
        sys.exit(1)
        
    # Load config file
    cfg = utils.load_config(config_file, default_cfg())
   
    # Begin MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    
    ants = utils.build_array()
    Nant = len(ants)
    ant_index = list(ants.keys())
 
    uvd = utils.empty_uvdata(ants=ants,**cfg['sim_spec'])

    # Create frequency array
    freq0 = 1.e8
    freqs = np.unique(uvd.freq_array)
    ra_dec, flux = utils.load_ptsrc_catalog(cfg['cat_name'], freq0=freq0, 
                                            freqs=freqs)

    # Build list of beams using Best fit coeffcients for Chebyshev polynomials
    ref_freq = cfg['simulation']['ref_freq']
    beam_coeffs = cfg['simulation']['beam_coeffs']
    spectral_index = cfg['simulation']['spectral_index']
    perturb = cfg['simulation']['perturb']
    seed = cfg['simulation']['seed']
    sigma = cfg['simulation']['sigma']
    mainlobe_scale = cfg['simulation']['mainlobe_scale']
    mainlobe_width = cfg['simulation']['mainlobe_width']
    nmodes = cfg['simulation']['nmodes']

    if perturb:
        np.random.seed(seed)
        pcoeffs = np.random.randn(nmodes)
        beam_list = [PerturbedPolyBeam(pcoeffs, perturb_scale=sigma,
                        mainlobe_scale=mainlobe_scale,
                        mainlobe_width=mainlobe_width,beam_coeffs=beam_coeffs,
                       spectral_index=spectral_index, ref_freq=ref_freq) 
                     for i in range(Nant)]

    else:
        beam_list = [ PolyBeam(beam_coeffs=beam_coeffs, 
                           spectral_index=spectral_index,
                           ref_freq=ref_freq) 
                  for i in range(Nant)]

    # Create VisCPU visibility simulator object (MPI-enabled)
    simulator = VisCPU(
        uvdata=uvd,
        beams=beam_list,
        beam_ids=ant_index,
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        real_dtype=np.float64,
        complex_dtype=np.complex128,
        use_pixel_beams=False,
        bm_pix = 20,
        mpi_comm=comm
    )
        
    # Run simulation
    tstart = time.time()
    simulator.simulate()
    print("Run took %2.1f sec" % (time.time() - tstart))

    if myid != 0:
        # Wait for root worker to finish IO before quitting
        comm.Barrier()
        comm.Finalize() # FIXME
        sys.exit(0)

    # Write simulated data to file
    uvd = simulator.uvdata
    red_grps, vecs, bl_lens, = uvd.get_redundancies()
    uvd.write_uvh5("calibration/test_sim.uvh5", clobber=True)
    
    # Generate gain model
    nfreqs = cfg['sim_spec']['nfreq']
    gg = utils.generate_gains(Nant, nfreqs, **cfg['sim_gain'])
    utils.save_simulated_gains(uvd, gg, outfile='test.calfits', overwrite=False)
    
    # Loop over all baselines and apply gain factor
    uvd_g = copy.deepcopy(uvd)
    for bl in np.unique(uvd_g.baseline_array):
        
        # Calculate product of gain factors (time-indep. for now)
        ant1, ant2 = uvd_g.baseline_to_antnums(bl)
        gigj = gg[ant1] * gg[ant2].conj()
        
        # Get index in data array
        idxs = uvd_g.antpair2ind((ant1, ant2))
        uvd_g.data_array[idxs,0,:,0] *= np.atleast_2d(gigj.astype(uvd_g.data_array.dtype))
    uvd_g.write_uvh5("calibration/test_sim_g.uvh5", clobber=True)
    
    # Add noise
    uvd_n = utils.add_noise_from_autos(uvd_g, nsamp=1., seed=10, inplace=False)    
    uvd_n.write_uvh5("calibration/test_sim_n.uvh5", clobber=True)

    # Sync with other workers and finalise
    comm.Barrier()
    comm.Finalize()
    sys.exit(0)

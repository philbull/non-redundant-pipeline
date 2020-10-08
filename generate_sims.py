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
from hera_sim.beams import PolyBeam, PerturbedPolyBeam

# Default setting for use_mpi (can be overridden by commandline arg)
USE_MPI_DEFAULT = True

try:
    import healpy
    import healvis
except:
    print("Unable to import healpy and/or healvis; diffuse mode unavailable")

import utils
import time, copy, sys


def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation specification
    cfg_spec = dict( nfreq=20,
                     start_freq=1.e8,
                     bandwidth=0.2e8,
                     start_time=2458902.33333,
                     integration_time=40.,
                     ntimes=40,
                     cat_name="gleamegc.dat",
                     apply_gains=True,
                     apply_noise=True,
                     ant_pert=False,
                     seed=None,
                     ant_pert_sigma=0.0,
                     use_legacy_array=False,
                     hex_spec=(3,4), 
                     hex_ants_per_row=None, 
                     hex_ant_sep=14.6,
                     use_ptsrc=True )
                        
    # Diffuse model specification
    cfg_diffuse = dict( use_diffuse=False,
                        nside=64,
                        obs_latitude=-30.7215277777,
                        obs_longitude = 21.4283055554,
                        beam_pol='XX',
                        nprocs=1 )
    
    # Beam model parameters
    cfg_beam = dict( ref_freq=1.e8,
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
                     rotation_dist='',
                     rotation_mean=0.0,
                     rotation_sigma=0.0,
                     mainlobe_width=0.3, 
                     nmodes=8,
                     beam_coeffs=[ 0.29778665, -0.44821433, 0.27338272, 
                                  -0.10030698, -0.01195859, 0.06063853, 
                                  -0.04593295,  0.0107879,  0.01390283, 
                                  -0.01881641, -0.00177106, 0.01265177, 
                                  -0.00568299, -0.00333975, 0.00452368,
                                   0.00151808, -0.00593812, 0.00351559
                                 ] )
    
    # Fluctuating gain model parameters
    cfg_gain = dict(nmodes=8, seed=None)
    
    # Noise parameters
    cfg_noise = dict(nsamp=1., seed=None, noise_file=None)
    
    # Combine into single dict
    cfg = { 'sim_beam':     cfg_beam,
            'sim_spec':     cfg_spec,
            'sim_diffuse':  cfg_diffuse,
            'sim_noise':    cfg_noise,
            'sim_gain':     cfg_gain,
           }
    return cfg


if __name__ == '__main__':
    # Run simulations (MPI-enabled)    

    # Get config file name from args
    use_mpi = USE_MPI_DEFAULT
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        if len(sys.argv) > 2:
            use_mpi = bool(sys.argv[2])
            print("use_mpi:", use_mpi)
    else:
        print("Usage: analyse_sims.py config_file [use_mpi]")
        sys.exit(1)
    
    # Begin MPI
    if use_mpi:
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
    else:
        comm = None
        myid = 0
    
    # Load config file
    cfg = utils.load_config(config_file, default_cfg())
    cfg_spec = cfg['sim_spec']
    cfg_diffuse = cfg['sim_diffuse']
    cfg_out = cfg['sim_output']
    cfg_beam = cfg['sim_beam']
    cfg_gain = cfg['sim_gain']
    cfg_noise = cfg['sim_noise']
    
    # Construct array layout to simulate
    if cfg_spec['use_legacy_array']:
        # This is the deprecated legacy function 
        ants = utils.build_array()
    else:
        ants = utils.build_hex_array(hex_spec=cfg_spec['hex_spec'], 
                                     ants_per_row=cfg_spec['hex_ants_per_row'], 
                                     d=cfg_spec['hex_ant_sep'])
    Nant = len(ants)
    ant_index = list(ants.keys())

    # Perturb antenna positions
    if cfg_spec['ant_pert']:
        np.random.seed(cfg_spec['seed'])
        for i in range(Nant):
            ants[i] = tuple(list(ants[i]) + 
                            (cfg_spec['ant_pert_sigma']*np.random.randn(3))) #3 for x,y,z
    
    # Build empty UVData object with correct dimensions
    uvd = utils.empty_uvdata(ants=ants, **cfg_spec)

    # Create frequency array
    freq0 = 100e6
    freqs = np.unique(uvd.freq_array)
    
    # Load point source catalogue
    if cfg_spec['use_ptsrc']:
        ra_dec, flux = utils.load_ptsrc_catalog(cfg_spec['cat_name'], 
                                                freq0=freq0, freqs=freqs, 
                                                usecols=(0,1,2,3))

    # Build list of beams using best-fit coefficients for Chebyshev polynomials
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

        # Perturb mainlobe
        mainlobe_scale = mainlobe_scale_sigma * np.random.randn(Nant) \
                       + mainlobe_scale_mean

        # Perturb xstretch and ystretch
        xstretch = np.full(Nant,xstretch_mean)
        ystretch = np.full(Nant,ystretch_mean)
        
        if cfg_beam['xystretch_dist'] == 'Gaussian':
            xstretch = xstretch_sigma * np.random.randn(Nant) + xstretch_mean
            if cfg_beam['xystretch_same']:
                ystretch = xstretch
            else:
                ystretch = ystretch_sigma * np.random.randn(Nant) + ystretch_mean
    
        if cfg_beam['xystretch_dist'] == 'Uniform': 
            xstretch = xstretch_mean + np.random.uniform(-2.*xstretch_sigma, 
                                                         2.*xstretch_sigma, 
                                                         Nant)
            if cfg_beam['xystretch_same']:
                ystretch = xstretch
            else:
                ystretch = ystretch_mean + np.random.uniform(-2.*ystretch_sigma, 
                                                             2.*ystretch_sigma, 
                                                             Nant)
        if cfg_beam['xystretch_dist'] == 'Outlier':
            xstretch[cfg_beam['outlier_ant_id']] = cfg_beam['outlier_xstretch']
            ystretch = xstretch

        # Perturb rotation
        rotation = np.zeros(Nant)
        if cfg_beam['rotation_dist'] == 'Gaussian': 
            rotation = rotation_sigma * np.random.randn(Nant) + rotation_mean
        elif cfg_beam['rotation_dist'] == 'Uniform':
            rotation = np.random.uniform(0.,360.,Nant)
        else:
            raise ValueError("rotation_dist '%s' not recognized" \
                             % cfg_beam['rotation_dist'])

        # Perturb sidelobe and other perturbation
        beam_list = [PerturbedPolyBeam(np.random.randn(cfg_beam['nmodes']),
                                       mainlobe_scale= mainlobe_scale[i],
                                       xstretch=xstretch[i], 
                                       ystretch=ystretch[i],
                                       rotation=rotation[i],
                                       **cfg_beam) for i in range(Nant)]
    else:
        beam_list = [PolyBeam(**cfg_beam) for i in range(Nant)]
    
    # Use VisCPU to create point source sim, or load a template file with 
    # correct data structures instead
    if cfg_spec['use_ptsrc']:
        
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
            # Wait for root worker to finish IO before ending all other worker procs
            comm.Barrier()
            sys.exit(0)
    
        # Write simulated data to file
        uvd = simulator.uvdata
        if cfg_out['datafile_true'] != '':
            uvd.write_uvh5(cfg_out['datafile_true'], clobber=cfg_out['clobber'])
    
    
    # Simulate diffuse model using healvis (multi-threaded)
    if cfg_diffuse['use_diffuse']:
        
        # Create healvis baseline spec
        healvis_bls = []
        for i in range(len(ants)):
            for j in range(i, len(ants)):
                _bl = healvis.observatory.Baseline(ants[i], ants[j], i, j)
                healvis_bls.append(_bl)

        # Set times
        times = np.unique(uvd.time_array)
        Ntimes = times.size

        # Create Observatory object
        fov = 360. # deg
        obs = healvis.observatory.Observatory(cfg_diffuse['obs_latitude'], 
                                              cfg_diffuse['obs_longitude'], 
                                              array=healvis_bls, 
                                              freqs=freqs)
        obs.set_pointings(times)
        obs.set_fov(fov)
        obs.set_beam(beam_list) # beam list
        
        # Create GSM sky model
        gsm = healvis.sky_model.construct_skymodel('gsm', freqs=freqs, 
                                                   ref_chan=0,
                                                   Nside=cfg_diffuse['nside'])

        # Compute visibilities
        gsm_vis, _times, _bls = obs.make_visibilities(gsm,
                                              beam_pol=cfg_diffuse['beam_pol'], 
                                              Nprocs=cfg_diffuse['nprocs'])
        
        # Check that ordering of healvis output matches existing uvd object
        antpairs_hvs = [(healvis_bls[i].ant1, healvis_bls[i].ant2) for i in _bls]
        antpairs_uvd = [uvd.baseline_to_antnums(_b) for _b in uvd.baseline_array]
        
        assert antpairs_hvs == antpairs_uvd, \
                                 "healvis 'bls' array does not match the " \
                                 "ordering of existing UVData.baseline_array"
        assert np.all(_times == uvd.time_array), \
                                 "healvis 'times' array does not match the " \
                                 "ordering of existing UVData.time_array"
        
        # Add diffuse data to UVData object
        uvd.data_array[:,:,:,0] += gsm_vis
        
        # Output diffuse-added data if requested
        if cfg_out['datafile_post_diffuse'] != '':
            uvd.write_uvh5(cfg_out['datafile_post_diffuse'], 
                           clobber=cfg_out['clobber'])
    
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
    if use_mpi:
        comm.Barrier()
    sys.exit(0)
    

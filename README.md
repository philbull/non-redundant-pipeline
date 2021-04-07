
# Non-redundant analysis pipeline
Pipeline for simulating, calibrating, and analysing data from non-redundant arrays, based on the HERA stack.

## How to use this cluster version

It is currently only usable on ILIFU (because the batch script headers are set for SLURM).

There is another layer of configuration on top of the yaml files. This is implemented using shell scripts and Python scripts. 
* Edit the file `set_run` which establishes this other layer of configuration. 
* Place the desired catalog files (catall.txt, catBC.txt etc.) in the current directory and set NR_CATALOG in `set_run` to one of these. All catalogues cannot be run at the same time, they have to be done separately. The catalog file will be used to name sub-directories in the output data heirarchy.

Then run `sh run_non`. This will create many temporary batch scripts and yaml files in the current directory. It will queue jobs on ILIFU to run all the non-redundant cases defined by the files yaml_files/generate_sims\*.yaml, in parallel.

The ordering of runs is important.
1. Run simulations for catalog catall.txt first, because this creates the files that will be used to determine the noise to add to simulations using other catalogues.
	* Run all the point source simulations (NR_WHICH="points" in `set_run`) first, and wait til they are finished.
	* Then run the diffuse simulations (NR_WHICH="diffuse" in `set_run`). 
2. Run the other catalogues.

Because of the way things are parallelized, the point source and diffuse simulations cannot be done in a single ILIFU job. That is the reason for the separation of points a. and b. The diffuse simulation will load data created by the point source simulation. Run the point source sims, then change NR_WHICH to diffuse, and run the diffuse sims.

The output will be in the place you specified as NR_OUTPUT_ROOT in `set_run`.

"""
Reconfigure the yaml files based on settings in set_run.
"""

import yaml, sys, os

# Walk the params
def change(d):
      for k, v in d.items():
        if isinstance(v, dict):
          v = change(v)
        else:
          if d[k] == "None": d[k] = None
      return d

def setup_yaml_gen(setup, in_gen, out_gen):
    # Retrieve environment


    CATALOG = setup["catalog"]
    OUTPUT_ROOT = setup["output_root"]
    NUM_TIMES = int(setup["num_times"])
    NUM_FREQS = int(setup["num_freqs"])
    APPLY_NOISE = setup["apply_noise"]
    APPLY_GAINS = setup["apply_gains"]
    APPLY_REFLECTION = setup["apply_reflection"]
    APPLY_XTALK = setup["apply_xtalk"]
    CAL_AS_HICKLE = setup["cal_as_hickle"]
    USE_EOR = setup["use_eor"]
    USE_DIFFUSE = setup["use_diffuse"]
    USE_HEALVIS = setup["use_healvis"]

    if APPLY_NOISE not in [ False, True ]:
        raise ValueError("APPLY_NOISE has an invalid value: "+str(APPLY_NOISE))
    if APPLY_GAINS not in [ False, True ]:
        raise ValueError("APPLY_GAINS has an invalid value: "+str(APPLY_GAINS))
    if APPLY_REFLECTION not in [ False, True ]:
        raise ValueError("APPLY_REFLECTION has an invalid value: "+str(APPLY_REFLECTION))
    if APPLY_XTALK not in [ False, True ]:
        raise ValueError("APPLY_XTALK has an invalid value: "+str(APPLY_GAINS))
    if CAL_AS_HICKLE not in [ False, True ]:
        raise ValueError("CAL_AS_HICKLE has an invalid value: "+str(CAL_AS_HICKLE))
    if USE_EOR not in [ False, True ]:
        raise ValueError("USE_EOR has an invalid value: "+str(USE_EOR))
    if USE_DIFFUSE not in [ False, True ]:
        raise ValueError("USE_EOR has an invalid value: "+str(USE_DIFFUSE))
    if USE_HEALVIS not in [ False, True ]:
        raise ValueError("USE_EOR has an invalid value: "+str(USE_HEALVIS))


    NSIDE = int(setup["nside"])
    DIFFUSE_MODEL = setup["diffuse_model"]
    if DIFFUSE_MODEL not in [ "EOR", "GSM" ]:
        raise ValueError("DIFFUSE_MODEL has an invalid value: "+str(DIFFUSE_MODEL))
    HEX_SPEC = setup["hex_spec"]

    with open(in_gen) as f:
      generate_sims = yaml.load(f, Loader=yaml.FullLoader)

 
    nf = os.path.basename(generate_sims["sim_output"]["datafile_true"])
    generate_sims["sim_spec"]["cat_name"] = CATALOG+".txt"
    generate_sims["sim_spec"]["ntimes"] = NUM_TIMES
    generate_sims["sim_spec"]["nfreq"] = NUM_FREQS
    generate_sims["sim_spec"]["bandwidth"] = NUM_FREQS*97e3
    generate_sims["sim_spec"]["hex_spec"] = HEX_SPEC
    generate_sims["sim_spec"]["apply_noise"] = APPLY_NOISE
    generate_sims["sim_spec"]["apply_gains"] = APPLY_GAINS
    generate_sims["sim_spec"]["apply_reflection"] = APPLY_REFLECTION
    generate_sims["sim_spec"]["apply_xtalk"] = APPLY_XTALK
    generate_sims["sim_spec"]["output_root"] = OUTPUT_ROOT



    # Add the parameters for use_diffuse
    generate_sims["sim_diffuse"]["use_diffuse"] = USE_DIFFUSE
    generate_sims["sim_diffuse"]["use_healvis"] = USE_HEALVIS
    generate_sims["sim_diffuse"]["nside"] = NSIDE  
    

    # Fix output file path
    for f in [ "datafile_true", "datafile_post_gains", "datafile_post_noise", "datafile_post_diffuse", "gain_file" ]:
      nf = os.path.basename(generate_sims["sim_output"][f])
      generate_sims["sim_output"][f] = OUTPUT_ROOT+"/"+CATALOG+"/"+nf

    # use_eor
    generate_sims["sim_eor"] = { "use_eor" : USE_EOR }
    generate_sims["sim_output"]["datafile_post_eor"] = generate_sims["sim_output"]["datafile_true"][:-5]+"_e.uvh5"


    # Add noise dump_file
    nf = os.path.basename(generate_sims["sim_output"]["datafile_post_noise"])[:-5]
    generate_sims["sim_output"]["noise_post_noise"] = OUTPUT_ROOT+"/"+CATALOG+"/"+nf+"n.npz"


    # Walk the parameters and change any instance of "None" to None
    generate_sims = change(generate_sims)
    generate_sims["orig_yaml"] = in_gen

    # Dump new sims parameters
    stream = open(out_gen, "w")
    yaml.dump(generate_sims, stream, default_flow_style=False)  
    stream.close()

def setup_yaml_analyse(setup, in_gen, in_analyse, out_analyse):
    CATALOG = setup["catalog"]
    OUTPUT_ROOT = setup["output_root"]
    NUM_FREQS = int(setup["num_freqs"])
    CAL_AS_HICKLE = setup["cal_as_hickle"]

    if CAL_AS_HICKLE not in [ False, True ]:
        raise ValueError("CAL_AS_HICKLE has an invalid value: "+str(CAL_AS_HICKLE))

    with open(in_gen) as f:
        generate_sims = yaml.load(f, Loader=yaml.FullLoader)

    for f in [ "datafile_post_gains", "gain_file" ]:
        nf = os.path.basename(generate_sims["sim_output"][f])
        generate_sims["sim_output"][f] = OUTPUT_ROOT+"/"+CATALOG+"/"+nf


    with open(in_analyse) as f:
        analyse_sims = yaml.load(f, Loader=yaml.FullLoader)

    # Change datafile true to be what you want. All of the cases add gains and noise,
    # so the final file to use is post_noise.
    analyse_sims["analysis"]["input_data"] = generate_sims["sim_output"]["datafile_post_gains"]
    analyse_sims["analysis"]["input_truegain"] = generate_sims["sim_output"]["gain_file"]
    analyse_sims["analysis"]["output_data"] = analyse_sims["analysis"]["input_data"][:-5]+"_cal.uvh5"
    analyse_sims["analysis"]["cal_as_hickle"] = CAL_AS_HICKLE
    analyse_sims["pspec_run"]["spw_ranges"] = "(0,"+str(NUM_FREQS)+")"
    analyse_sims["orig_yaml"] = in_analyse
    del analyse_sims["redcal"]["ant_metrics_file"]

    # Dump new sims parameters
    stream = open(out_analyse, "w")
    yaml.dump(analyse_sims, stream, default_flow_style=False)
    stream.close()

if __name__ == "__main__":
    import sys
    if sys.argv[1] not in [ "generate", "analyse" ]:
        raise ValueError("Invalid specification for generate or analyse")


    with open("globals.yaml") as f:
        global_setup = yaml.safe_load(f)

    if sys.argv[1] == "generate":
        setup_yaml_gen(global_setup, sys.argv[2], sys.argv[3])
    else:
        setup_yaml_analyse(global_setup, sys.argv[2], sys.argv[3], sys.argv[4])

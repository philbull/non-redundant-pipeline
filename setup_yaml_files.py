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

# Retrieve environment
AZ_ZA_CORRECTIONS = os.environ.get("NR_AZ_ZA_CORRECTIONS")
WHICH = os.environ.get("NR_WHICH")      # points or points+diffuse "diffuse"
CATALOG = os.environ.get("NR_CATALOG").rstrip(".txt")
OUTPUT_ROOT = os.environ.get("NR_OUTPUT_ROOT")
NPROC = int(os.environ.get("NR_NP"))
NUM_TIMES = int(os.environ.get("NR_NUM_TIMES"))
NUM_FREQS = int(os.environ.get("NR_NUM_FREQS"))
APPLY_NOISE = os.environ.get("NR_APPLY_NOISE")
APPLY_GAINS = os.environ.get("NR_APPLY_GAINS")
CAL_AS_HICKLE = os.environ.get("NR_CAL_AS_HICKLE")
if APPLY_NOISE not in [ "False", "True" ]:
    raise ValuesError("APPLY_NOISE has an invalid value: "+APPLY_NOISE)
if APPLY_GAINS not in [ "False", "True" ]:
    raise ValuesError("APPLY_GAINS has an invalid value: "+APPLY_GAINS)
if CAL_AS_HICKLE not in [ "False", "True" ]:
    raise ValuesError("CAL_AS_HICKLE has an invalid value: "+CAL_AS_HICKLE)
NSIDE = int(os.environ.get("NR_NSIDE"))
DIFFUSE_MODEL = os.environ.get("NR_DIFFUSE_MODEL")
if DIFFUSE_MODEL not in [ "EOR", "GSM" ]:
    raise ValuesError("DIFFUSE_MODEL has an invalid value: "+DIFFUSE_MODEL)
DUMMY_SOURCE = os.environ.get("NR_DUMMY_SOURCE")
if DUMMY_SOURCE not in [ "False", "True" ]:
    raise ValuesError("DUMMY_SOURCE has an invalid value: "+DUMMY_SOURCE)
HEX_SPEC = os.environ.get("NR_HEX_SPEC")

GENERATE_SIMS_YAML = sys.argv[1]
ANALYSE_SIMS_YAML = sys.argv[2]

if WHICH not in [ "points", "diffuse" ]:
    raise ValueError(WHICH+" is invalid")

with open(GENERATE_SIMS_YAML) as f:
  generate_sims = yaml.load(f, Loader=yaml.FullLoader)
stream = open("g_orig.yaml", "w")	# Save original params
yaml.dump(generate_sims, stream, default_flow_style=False)
stream.close()

nf = os.path.basename(generate_sims["sim_output"]["datafile_true"])
if WHICH == "points": generate_sims["sim_spec"]["load_points_sim"] = None
else: generate_sims["sim_spec"]["load_points_sim"] = OUTPUT_ROOT+"/"+CATALOG+"/calibration_points/"+nf
generate_sims["sim_spec"]["az_za_corrections"] = AZ_ZA_CORRECTIONS
generate_sims["sim_spec"]["cat_name"] = CATALOG+".txt"
generate_sims["sim_spec"]["ntimes"] = NUM_TIMES
generate_sims["sim_spec"]["nfreq"] = NUM_FREQS
generate_sims["sim_spec"]["hex_spec"] = HEX_SPEC
generate_sims["sim_spec"]["apply_noise"] = (APPLY_NOISE == "True")
generate_sims["sim_spec"]["apply_gains"] = (APPLY_GAINS == "True")
generate_sims["sim_spec"]["dummy_source"] = (DUMMY_SOURCE == "True")
generate_sims["sim_spec"]["output_root"] = OUTPUT_ROOT


if CATALOG == "catall":
  generate_sims["sim_noise"]["noise_file"] = None
elif APPLY_NOISE == "True": 
    nf = os.path.basename(generate_sims["sim_output"]["datafile_true"])
    if WHICH == "diffuse": nf = nf[:-5]+"_d"+nf[-5:]
    generate_sims["sim_noise"]["noise_file"] = OUTPUT_ROOT+"/catall/calibration_"+WHICH+"/"+nf
else:
    generate_sims["sim_noise"]["noise_file"] = None

# Add the parameters for use_diffuse
generate_sims["sim_output"].update(datafile_post_diffuse = \
        generate_sims["sim_output"]["datafile_post_gains"].replace("_g", "_d"))
generate_sims["sim_diffuse"] = { "use_diffuse" : (WHICH=="diffuse"), "nside" : NSIDE, "beam_pol" : "XX", "nprocs" : NPROC, "diffuse_model" : DIFFUSE_MODEL }

# Fix output file path
for f in [ "datafile_true", "datafile_post_gains", "datafile_post_noise", "datafile_post_diffuse", "gain_file" ]:
  nf = os.path.basename(generate_sims["sim_output"][f])
  generate_sims["sim_output"][f] = OUTPUT_ROOT+"/"+CATALOG+"/calibration_"+WHICH+"/"+nf

# Walk the parameters and change any instance of "None" to None
generate_sims = change(generate_sims)
generate_sims["orig_yaml"] = GENERATE_SIMS_YAML

# Dump new sims parameters
stream = open("g.yaml", "w")
yaml.dump(generate_sims, stream, default_flow_style=False)  
stream.close()


with open(ANALYSE_SIMS_YAML) as f:
  analyse_sims = yaml.load(f, Loader=yaml.FullLoader)
stream = open("a_orig.yaml", "w")       # Save original params
yaml.dump(analyse_sims, stream, default_flow_style=False)
stream.close()

# Change datafile true to be what you want. All of the cases add gains and noise,
# so the final file to use is post_noise.
analyse_sims["analysis"]["input_data"] = generate_sims["sim_output"]["datafile_post_gains"]
analyse_sims["analysis"]["input_truegain"] = generate_sims["sim_output"]["gain_file"]
analyse_sims["analysis"]["output_data"] = analyse_sims["analysis"]["input_data"][:-5]+"_cal.uvh5"
analyse_sims["analysis"]["cal_as_hickle"] = (CAL_AS_HICKLE == "True")
analyse_sims["pspec_run"]["spw_ranges"] = "(0,"+str(NUM_FREQS)+")"


# Dump new sims parameters
stream = open("a.yaml", "w")
yaml.dump(analyse_sims, stream, default_flow_style=False)
stream.close()

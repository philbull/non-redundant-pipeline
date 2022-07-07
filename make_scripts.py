import numpy as np
import os, yaml

gen_to_analyse = {
    "generate_sims_4a_0.01.yaml":  "analyse_sims_eor10.yaml",
    "generate_sims_4a_0.02.yaml":  "analyse_sims_eor11.yaml",
    "generate_sims_4b_0.01.yaml":  "analyse_sims_eor12.yaml",
    "generate_sims_4b_0.02.yaml":  "analyse_sims_eor13.yaml",
    "generate_sims_4c_a.yaml":  "analyse_sims_eor14.yaml",
    "generate_sims_4c_b.yaml":  "analyse_sims_eor15.yaml",
    #"generate_sims_main001.yaml":  "analyse_sims_eor3.yaml",        # Get beam interpolation erro
    #"generate_sims_main002.yaml":  "analyse_sims_eor4.yaml",
    "generate_sims_outlier2_1.1.yaml":  "analyse_sims_eor8.yaml",
    "generate_sims_outlier7_1.1.yaml":  "analyse_sims_eor9.yaml",
    "generate_sims_side005.yaml":  "analyse_sims_eor1.yaml",
    "generate_sims_side02.yaml":  "analyse_sims_eor2.yaml",
    "generate_sims_unixystretch0.01.yaml":  "analyse_sims_eor7.yaml",
    "generate_sims_xystretch0.01.yaml":  "analyse_sims_eor5.yaml",
    "generate_sims_xystretch0.02.yaml":  "analyse_sims_eor6.yaml",
    "generate_sims.yaml":  "analyse_sims_eor.yaml",
    "generate_sims_one_bright.yaml":  "analyse_sims_eor.yaml",       # Based on generate_sims.yaml
    "generate_sims_high_diffuse.yaml":  "analyse_sims_eor.yaml"       # Based on generate_sims.yaml
}

LOCAL = ""

def create_sim_script(runtime, mem, script_to_run, gen_or_analyse, in_gen_yaml, in_analyse_yaml, script_name, output):

    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

# {script}, {gen_or_analyse}, {in_gen_yaml}, {in_analyse_yaml}

sleep `od -An -N1 -i /dev/random`

module load hdf5

source ~/.bashrc
conda activate sampler
date
python setup_yaml_files.py {gen_or_analyse} {in_gen_yaml} {in_analyse_yaml} {gen_or_analyse}$$.yaml
if ! python {script} {gen_or_analyse}$$.yaml
then
  echo {output} failed
  rm {output} 2> /dev/null
fi
rm {gen_or_analyse}$$.yaml
date
""".format(runtime=runtime, mem=mem, script=script_to_run, gen_or_analyse=gen_or_analyse, in_gen_yaml=in_gen_yaml, 
		in_analyse_yaml=in_analyse_yaml, output=output)

    with open(script_name, "w") as f:
        f.write(script)

def create_sampler_script(runtime, mem, file_root, in_sampler_yaml, global_yaml, script_name, output):

    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

# {file_root}, {in_sampler_yaml}, {global_yaml}

sleep `od -An -N1 -i /dev/random`

module load hdf5

source ~/.bashrc
conda activate sampler
cd ../gain_sampler
date
python setup_yaml_file.py {file_root} {global_yaml} {in_sampler_yaml} samp$$.yaml
if ! python run_sampler.py samp$$.yaml
then
  echo {output} failed
  rm {output} 2> /dev/null
fi
rm samp$$.yaml
date
""".format(runtime=runtime, mem=mem, file_root=file_root, in_sampler_yaml=in_sampler_yaml,
          global_yaml=global_yaml, output=output)

    with open(script_name, "w") as f:
        f.write(script)

def create_plots_script(runtime, mem, file_root, case, script_name, output):
    
    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

# {file_root}, {case}

sleep `od -An -N1 -i /dev/random`

module load hdf5

source ~/.bashrc
conda activate sampler
cd ../gain_sampler
date
sed s/SAMPLER_DIR/'{file_root}'/ paper_plots.ipynb > paper_plots_{case}.ipynb
if papermill paper_plots_{case}.ipynb {out_dir}/paper_plots_{case}_out.ipynb
then
  cp {output} .
else
  echo {output} failed
  rm {output} `basename {output}` 2> /dev/null
fi  

rm paper_plots_{case}.ipynb 
date
""".format(runtime=runtime, mem=mem, file_root=file_root.replace("/", "\\/"), out_dir=os.path.dirname(file_root), case=case, script_name=script_name, output=output)

    with open(script_name, "w") as f:
        f.write(script)
        
def write_makefile_rule(output, dependencies, comment, command):
    with open("Makefile", "a") as f:
        f.write("# "+comment+"\n")
        f.write(output+":")
        for dep in dependencies:
            f.write(" "+dep)
        f.write("\n")
        f.write("\t"+command+"\n\n")    

def cl(case):
    if case == "": return "vanilla"
    else: return case

def cs(case):
    if case == "": return ""
    else: return "_"+case
    
file_counter = 1

with open("globals.yaml") as f:
    global_setup = yaml.safe_load(f)

with open("Makefile", "w") as f:
    f.write("ROOT := "+global_setup["output_root"]+"\n\n")



all = ""
phony = []
for gen in gen_to_analyse:
    # Make the case match the actual output files
    with open("yaml_files/"+gen) as f:
        generate_sims = yaml.safe_load(f)
    datafile = generate_sims["sim_output"]["datafile_true"]
    case = os.path.basename(datafile)[9:-5]
    if generate_sims["sim_spec"]["cat_name"] is None:
        generate_sims["sim_spec"]["cat_name"] = global_setup["catalog_dir"]+".txt"

    CATALOG_DIR = generate_sims["sim_spec"]["cat_name"][:-4]
    
    ## Generate

    output = "${ROOT}/"+CATALOG_DIR+"/viscatBC"+cs(case)+"_g.uvh5"
    # runtime, mem, script_to_run, gen_or_analyse, in_gen_yaml, in_analyse_yaml, script_name
    create_sim_script(global_setup["sim_time_gen"], 
                      int(np.ceil(float(global_setup["sim_mem"])/8)), 
                      "generate_sims.py", 
                      "generate", 
                      "yaml_files/"+gen, 
                      "", 
                      "run_generate_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))
    
    
    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ "yaml_files/"+gen, 
                         "globals.yaml",
                         "generate_sims.py"
                          ],
                        "Generate",
                        "sh run_generate_"+str(file_counter))
    previous_output = output

    
    ## Analyse
    
    output = "${ROOT}/"+CATALOG_DIR+"/viscatBC"+cs(case)+"_g_cal.uvh5"
    create_sim_script(global_setup["sim_time_analyse"], 
                      int(np.ceil(float(global_setup["sim_mem"])/8)), 
                      "analyse_sims.py", 
                      "analyse", 
                      "yaml_files/"+gen, 
                      "yaml_files/"+gen_to_analyse[gen], 
                      "run_analyse_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))
    
    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ previous_output,
                           "yaml_files/"+gen_to_analyse[gen], 
                           "analyse_sims.py"
                         ],
                        "Analyse",
                        "sh run_analyse_"+str(file_counter))
    previous_output = output

    
    ## Sample
    
    
    output = "${ROOT}/"+CATALOG_DIR+"/sampled_viscatBC"+cs(case)+"/sampler.hkl"
    # runtime, mem, file_root, in_sampler_yaml, global_yaml, script_name
    create_sampler_script(global_setup["sampler_time"], 
                          int(np.ceil(float(global_setup["sampler_mem"])/8)), 
                          global_setup["output_root"]+"/"+CATALOG_DIR+"/viscatBC"+cs(case), 
                          "yaml_files/sampler.yaml", 
                          "../non-redundant-pipeline/globals.yaml",
                          "run_sample_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))
    
    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ previous_output,
                           "../gain_sampler/run_sampler.py",
                         "../gain_sampler/yaml_files/sampler.yaml",
                         ],
                        "Sampler",
                        "sh run_sample_"+str(file_counter))
    previous_output = output

    
    ## Plot
    
    output = "${ROOT}/"+CATALOG_DIR+"/paper_plots_"+cl(case)+"_out.ipynb"
    # runtime, mem, file_root, case, script_name
    create_plots_script(global_setup["sampler_time"], 
                        int(np.ceil(float(global_setup["sampler_mem"])/8)), 
                        global_setup["output_root"]+"/"+CATALOG_DIR+"/sampled_viscatBC"+cs(case), 
                        cl(case), 
                        "run_plot_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))
    
    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ previous_output,
                          "../gain_sampler/paper_plots.ipynb",
                          ],
                        "Plot",
                        "sh run_plot_"+str(file_counter))
    all += CATALOG_DIR+"/paper_plots_"+cl(case)+"_out.ipynb"+" "
    phony.append( ( CATALOG_DIR+"/paper_plots_"+cl(case)+"_out.ipynb", output ) )
    previous_output = output
    file_counter += 1
    
    
def create_varied_sampler_run(case):
    global all, phony, file_counter
    
    # Always based on vanilla sim. Just different sampler yaml file, then need different plot output.
    # Must be a sampler yaml file called sampler_<case>.yaml

    # runtime, mem, file_root, in_sampler_yaml, global_yaml, script_name
    CATALOG_DIR = "catall_nobright"
    output = "${ROOT}/"+CATALOG_DIR+"/sampled_viscatBC_"+case+"/sampler.hkl"
    create_sampler_script(global_setup["sampler_time"], 
                          int(np.ceil(float(global_setup["sampler_mem"])/8)), 
                          global_setup["output_root"]+"/"+CATALOG_DIR+"/viscatBC", 
                          "yaml_files/sampler_"+case+".yaml", 
                          "../non-redundant-pipeline/globals.yaml",
                          "run_sample_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))

    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ global_setup["output_root"]+"/"+CATALOG_DIR+"/viscatBC_g_cal.uvh5",
                           "../gain_sampler/run_sampler.py",
                         "../gain_sampler/yaml_files/sampler_"+case+".yaml",
                          ],
                        "Sampler",
                        "sh run_sample_"+str(file_counter))
    previous_output = output

     # runtime, mem, file_root, case, script_name
    output = "${ROOT}/"+CATALOG_DIR+"/paper_plots_vanilla_"+case+"_out.ipynb"
    create_plots_script(global_setup["sampler_time"], 
                        int(np.ceil(float(global_setup["sampler_mem"])/8)), 
                        global_setup["output_root"]+"/"+CATALOG_DIR+"/sampled_viscatBC_"+case, 
                        "vanilla_"+case, 
                        "run_plot_"+str(file_counter), output.replace("${ROOT}", global_setup["output_root"]))

    # output, dependencies, comment, command
    write_makefile_rule(output,
                        [ previous_output,
                          "../gain_sampler/paper_plots.ipynb",
                          ],
                        "Plot",
                        "sh run_plot_"+str(file_counter))

    all += CATALOG_DIR+"/paper_plots_vanilla_"+case+"_out.ipynb"+" "
    phony.append( ( CATALOG_DIR+"/paper_plots_vanilla_"+case+"_out.ipynb", output ) )
    file_counter += 1

for variation in [ "4_modes_gauss",  "all_modes_gauss",  "flat_prior", "huge_prior",  "wide_prior", "Vprior_offset" ]: 
    create_varied_sampler_run(variation)
    
"""
## Small sim ======
# It has a different generate sims AND a different sampler yaml

# Generate
output = "$ROOT"+"/catall_nobright/viscatBC_small_g.uvh5"
write_makefile_rule(output,
                    [ "yaml_files/generate_sims_small.yaml",
                      ],
                    "Generate",
                    "sh run_generate_small")
previous_output = output


# Sample
output = "$ROOT"+"/catall_nobright/sampled_viscatBC_small_small"
write_makefile_rule(output,
                    [ previous_output,
                     "../gain_sampler/yaml_files/sampler_small.yaml",
                      ],
                    "Sample",
                    "sh run_sample_small")
previous_output = output

# Plot
output = "$ROOT"+"/catall_nobright/paper_plots_sim_small.ipynb"
write_makefile_rule(output,
                    [ previous_output,
                     "../gain_sampler/paper_plots_simple.ipynb",
                      ],
                    "Plot",
                    "sh run_plot_small")

all += "catall_nobright/paper_plots_sim_small.ipynb "
phony.append( ("catall_nobright/paper_plots_sim_small.ipynb" , output ) )
"""

# Dump the whole list for make all

with open("Makefile", "a") as f:
    f.write(".PHONY: "+" ".join([ p[0] for p in phony])+"\n\n")
with open("Makefile", "a") as f:
    for p in phony:
        f.write(p[0]+": "+p[1]+"\n\n")

        
with open("Makefile", "a") as f:
    f.write("all: "+all+"\n")
    f.write("\techo Finished all\n") 

    


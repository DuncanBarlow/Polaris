import numpy as np
import utils_deck_generation as idg
import healpy_pointings as hpoint
import netcdf_read_write as nrw
import utils_intensity_map as uim
import os
import subprocess
import sys
from scipy.stats import qmc


def define_system_params(root_dir):
    sys_params = {}
    sys_params["num_parallel_ifriits"] = 1
    sys_params["num_openmp_parallel"] = 4
    sys_params["num_ex_checkpoint"] = 1

    sys_params["run_gen_deck"] = True
    sys_params["run_sims"] = True
    sys_params["run_checkpoint"] = True
    sys_params["run_clean"] = True

    sys_params["root_dir"] = root_dir
    sys_params["config_dir"] = "config_"
    sys_params["sim_dir"] = "time_"
    sys_params["trainingdata_filename"] = "training_data_and_labels.nc"
    sys_params["ifriit_ouput_name"] = "p_in_z1z2_beam_all.nc"
    sys_params["figure_location"] = "plots"
    sys_params["plot_file_type"] = ".pdf"
    sys_params["plasma_profile_dir"] = "plasma_profiles"
    sys_params["ifriit_input_name"] = "ifriit_inputs_base.txt"
    sys_params["plasma_profile_nc"] = "ifriit_1davg_input.nc"
    sys_params["heat_source_nc"] = "heat_source_all_beams.nc"
    sys_params["dataset_params_filename"] = "dataset_params.nc"
    sys_params["facility_spec_filename"] = "facility_spec.nc"
    sys_params["deck_gen_params_filename"] = "deck_gen_params.nc"
    sys_params["ifriit_binary_filename"] = "main"

    return sys_params



def define_dataset_params(num_examples,
                          random_sampling=0,
                          random_seed=12345):
    dataset_params = {}
    target_radius = 850.0

    dataset_params["num_examples"] = num_examples
    dataset_params["random_seed"] = random_seed
    dataset_params["random_sampling"] = random_sampling
    dataset_params["hemisphere_symmetric"] = True
    dataset_params["imap_nside"] = 256
    dataset_params["run_plasma_profile"] = False
    dataset_params["run_with_cbet"] = False
    dataset_params["num_profiles_per_config"] = 1

    num_variables_per_beam = 0
    # pointings
    dataset_params["surface_cover_radians"] = np.radians(30.0)
    dataset_params["theta_index"] = num_variables_per_beam
    num_variables_per_beam += 1
    dataset_params["phi_index"] = num_variables_per_beam
    num_variables_per_beam += 1
    # defocus
    dataset_params["defocus_default"] = 0.0
    dataset_params["defocus_range"] = 35.0 # mm
    dataset_params["defocus_bool"] = True
    if dataset_params["defocus_bool"]:
        dataset_params["defocus_index"] = num_variables_per_beam
        num_variables_per_beam += 1
    # quad splitting
    dataset_params["quad_split_range"] = 3.0 # multiples of angular beam seperation within port
    dataset_params["quad_split_bool"] = True
    dataset_params["quad_split_skew_bool"] = True
    if dataset_params["quad_split_bool"]:
        dataset_params["quad_split_index"] = num_variables_per_beam
        num_variables_per_beam += 1
        if dataset_params["quad_split_skew_bool"]:
            dataset_params["quad_split_skew_index"] = num_variables_per_beam
            num_variables_per_beam += 1
    # power (time-varying?)
    dataset_params["min_power"] = 0.5 # fraction of full power
    dataset_params["power_index"] = num_variables_per_beam
    dataset_params["time_varying_pulse"] = False
    if dataset_params["time_varying_pulse"]:
        dataset_params["num_powers_per_cone"] = dataset_params["num_profiles_per_config"]
        num_variables_per_beam += dataset_params["num_profiles_per_config"]
    else:
        dataset_params["num_powers_per_cone"] = 1
        num_variables_per_beam += 1

    dataset_params["num_variables_per_beam"] = num_variables_per_beam
    dataset_params["run_type"] = "lmj" #"test" #"lmj" #"nif"
    if dataset_params["run_type"] == "nif":
        facility_spec = idg.import_nif_config()
    elif (dataset_params["run_type"] == "lmj") or (dataset_params["run_type"] == "test"):
        if dataset_params["quad_split_bool"]:
            print("LMJ not designed to work with quad splitting yet!")
        facility_spec = idg.import_lmj_config(dataset_params["quad_split_bool"])
    facility_spec['target_radius'] = target_radius

    dataset_params["LMAX"] = 30
    dataset_params["num_coeff"] = int(((dataset_params["LMAX"] + 2) * (dataset_params["LMAX"] + 1))/2.0)
    # Assume symmetry
    dataset_params["num_input_params"] = int(facility_spec['num_cones']/2) * dataset_params["num_variables_per_beam"]

    return dataset_params, facility_spec


def populate_dataset_random_inputs(dataset_params, dataset):

    random_generator=np.random.default_rng(dataset_params["random_seed"])
    if dataset_params["random_sampling"] == 1:
        print("Random Sampling!")
        sample = random_generator.random((dataset_params["num_examples"], dataset_params["num_input_params"]))
    else:
        sampler = qmc.LatinHypercube(d=dataset_params["num_input_params"],
                                     strength=1, seed=random_generator, optimization="random-cd")
        sample = sampler.random(n=dataset_params["num_examples"])
    dataset["input_parameters"] = sample

    return dataset



def define_dataset(dataset_params):
    dataset = {}
    dataset["non_expand_keys"] = ["non_expand_keys","num_evaluated"]
    dataset["num_evaluated"] = 0

    dataset["input_parameters"] = np.zeros((dataset_params["num_examples"], dataset_params["num_input_params"]))
    dataset["real_modes"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"], dataset_params["num_coeff"]))
    dataset["imag_modes"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"], dataset_params["num_coeff"]))
    dataset["avg_flux"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    dataset["rms"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    return dataset



def generate_training_data(dataset, dataset_params, sys_params, facility_spec):

    nrw.save_general_netcdf(dataset_params, sys_params["root_dir"] + "/" + sys_params["dataset_params_filename"])
    nrw.save_general_netcdf(facility_spec, sys_params["root_dir"] + "/" + sys_params["facility_spec_filename"])

    max_parallel = dataset["num_evaluated"]-1
    chkp_marker = 1.0
    filename_trainingdata = sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"]
    if sys_params["run_sims"]:
        # int is a floor round
        num_parallel_runs = int((dataset_params["num_examples"] - dataset["num_evaluated"]) / sys_params["num_parallel_ifriits"])
        if num_parallel_runs > 0:
            for ir in range(num_parallel_runs):
                min_parallel = max_parallel + 1
                max_parallel = min_parallel + sys_params["num_parallel_ifriits"] - 1
                dataset = run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)

                if sys_params["run_checkpoint"]:
                    if ((max_parallel + 1) >= (chkp_marker * sys_params["num_ex_checkpoint"])):
                        print("Save training data checkpoint at run: " + str(max_parallel))
                        dataset["num_evaluated"] = max_parallel + 1
                        nrw.save_general_netcdf(dataset, filename_trainingdata)
                        chkp_marker +=1

        if max_parallel != (dataset_params["num_examples"] - 1):
            min_parallel = max_parallel + 1
            max_parallel = dataset_params["num_examples"] - 1
            dataset = run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)

    if sys_params["run_checkpoint"]:
        dataset["num_evaluated"] = max_parallel + 1
        nrw.save_general_netcdf(dataset, filename_trainingdata)



def run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec):
    config_location = sys_params["root_dir"] + "/" + sys_params["config_dir"]
    for tind in range(dataset_params["num_profiles_per_config"]):
        sim_dir = "/" + sys_params["sim_dir"] + str(tind)

        if dataset_params["run_plasma_profile"] and tind!=0: # this ensures the first run will be a solid sphere
            num_mpi_parallel = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
        else:
            num_mpi_parallel = 1

        subprocess.check_call(["./bash_parallel_ifriit", config_location, sim_dir, str(min_parallel), str(max_parallel), str(num_mpi_parallel), str(sys_params["num_openmp_parallel"])])

    dataset = nrw.retrieve_xtrain_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)
    return dataset



def main(argv):
    sys_params = define_system_params(argv[1])

    print(argv[3])
    run_type = str(argv[3]).split("=")[1]

    if (run_type=="init") or (run_type=="full"):
        dataset_params, facility_spec = define_dataset_params(int(argv[2]))

        dataset = define_dataset(dataset_params)
        dataset = populate_dataset_random_inputs(dataset_params, dataset)

        deck_gen_params = idg.define_deck_generation_params(dataset_params, facility_spec)
        deck_gen_params = idg.create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)
        idg.save_data_dicts_to_file(sys_params, dataset, dataset_params, deck_gen_params, facility_spec)

    if (run_type=="restart") or (run_type=="full"):
        dataset, dataset_params, deck_gen_params, facility_spec = idg.load_data_dicts_from_file(sys_params)
        generate_training_data(dataset, dataset_params, sys_params, facility_spec)

    return dataset, dataset_params, sys_params, facility_spec



if __name__ == "__main__":
    _, _, _, _ = main(sys.argv)

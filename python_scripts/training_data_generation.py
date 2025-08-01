import numpy as np
import utils_deck_generation as idg
import healpy_pointings as hpoint
import netcdf_read_write as nrw
import utils_intensity_map as uim
import os
import subprocess
import sys
from scipy.stats import qmc
import shutil
import glob
import stat


def define_system_params(data_dir):
    sys_params = {}
    sys_params["num_parallel_ifriits"] = 4
    sys_params["num_openmp_parallel"] = 1
    sys_params["num_ex_checkpoint"] = 1

    sys_params["run_sims"] = True
    sys_params["run_checkpoint"] = True
    sys_params["run_clean"] = False

    sys_params["root_dir"] = ".."
    sys_params["data_dir"] = data_dir
    sys_params["config_dir"] = "config_"
    sys_params["sim_dir"] = "time_"
    sys_params["figure_location"] = "plots"
    sys_params["plot_file_type"] = ".pdf"
    sys_params["bash_parallel_ifriit"] = "bash_parallel_ifriit"
    sys_params["plasma_profile_dir"] = "plasma_profiles"
    sys_params["facility_config_files_dir"] = "facility_config_files"
    sys_params["python_dir"] = "python_scripts"

    sys_params["trainingdata_filename"] = "training_data_and_labels.nc"
    sys_params["dataset_params_filename"] = "dataset_params.nc"
    sys_params["facility_spec_filename"] = "facility_spec.nc"
    sys_params["deck_gen_params_filename"] = "deck_gen_params.nc"
    sys_params["ifriit_binary_filename"] = "main"

    sys_params["ifriit_run_files_dir"] = "ifriit_run_files"
    sys_params["ifriit_input_name"] = "ifriit_inputs_base.txt"
    sys_params["plasma_profile_nc"] = "ifriit_1davg_input.nc"
    sys_params["ifriit_ouput_name"] = "p_in_z1z2_beam_all.nc"
    sys_params["heat_source_nc"] = "heat_source_all_beams.nc"
    sys_params["ifriit_pulse_name"] = "pulse_per_beam.txt"

    sys_params["multi_dir"] = "multi_data"
    sys_params["multi_output_ascii_filename"] = "multi_output.txt"
    sys_params["multi_input_filename"] = "multi_input.txt"
    sys_params["multi_pulse_name"] = "laser_pulse.txt"

    return sys_params


def define_dataset_params(num_examples, sys_params,
                          random_seed=12345):
    dataset_params = {}
    dataset_params["facility"] = "omega" #"custom_facility" #"nif" #"lmj" # "omega"
    dataset_params["num_examples"] = num_examples
    dataset_params["random_seed"] = random_seed
    dataset_params["sampling_method"] = "random" #"random", "lhs", "linear"
    dataset_params["run_with_cbet"] = False
    dataset_params["run_plasma_profile"] = False
    dataset_params["bool_group_beams_by_cone"] = False

    dataset_params['target_radius'] = 2307.0

    dataset_params["plasma_profile_source"] = "default" #"multi" # "default"
    dataset_params['laser_wavelength_nm'] = 351.0 # multi inputs over-ride this
    dataset_params["num_profiles_per_config"] = 1
    dataset_params['default_power'] = np.zeros((dataset_params["num_profiles_per_config"])) + 1.0 # default power per beam TW
    dataset_params["plasma_profile_times"] = np.linspace(0.5,14.,int(dataset_params["num_profiles_per_config"]))
    dataset_params['illumination_evaluation_radii'] = np.zeros((dataset_params["num_profiles_per_config"])) \
                                                     + dataset_params['target_radius']

    dataset_params["imap_nside"] = 256
    dataset_params["LMAX"] = 30
    dataset_params["num_coeff"] = int(((dataset_params["LMAX"] + 2) * (dataset_params["LMAX"] + 1))/2.0)

    dataset_params = define_scan_parameters(dataset_params)

    # facility specifications
    if dataset_params["facility"] == "nif":
        facility_spec, dataset_params = idg.import_nif_config(sys_params, dataset_params)
    elif (dataset_params["facility"] == "lmj") or (dataset_params["facility"] == "test"):
        facility_spec, dataset_params = idg.import_lmj_config(sys_params, dataset_params)
    elif (dataset_params["facility"]=="custom_facility") or (dataset_params["facility"]=="omega"):
        facility_spec = idg.import_direct_drive_config(sys_params, dataset_params)
    elif (dataset_params["facility"] == "omega"):
        facility_spec = idg.import_direct_drive_config(sys_params)

    dataset_params["num_input_params"] = dataset_params['num_beam_groups'] * dataset_params["num_variables_per_beam"]

    return dataset_params, facility_spec


def define_scan_parameters(dataset_params):
    dataset_params["hemisphere_symmetric"] = False

    num_variables_per_beam = 0
    # pointings
    dataset_params["theta_bool"] = False
    dataset_params["pointing_bool"] = False
    dataset_params["surface_cover_radians"] = np.radians(30.0)
    if dataset_params["theta_bool"]:
        dataset_params["theta_index"] = num_variables_per_beam
        num_variables_per_beam += 1
    elif dataset_params["pointing_bool"]:
        dataset_params["theta_index"] = num_variables_per_beam
        num_variables_per_beam += 1
        dataset_params["phi_index"] = num_variables_per_beam
        num_variables_per_beam += 1
    # defocus
    dataset_params["defocus_default"] = 0.0
    dataset_params["defocus_range"] = 35.0 # mm
    dataset_params["defocus_bool"] = False
    if dataset_params["defocus_bool"]:
        dataset_params["defocus_index"] = num_variables_per_beam
        num_variables_per_beam += 1
    # quad splitting
    dataset_params["quad_split_range"] = 3.0 # multiples of angular beam seperation within port
    dataset_params["quad_split_bool"] = False
    dataset_params["quad_split_skew_bool"] = False
    if dataset_params["quad_split_bool"]:
        dataset_params["quad_split_index"] = num_variables_per_beam
        num_variables_per_beam += 1
        if dataset_params["quad_split_skew_bool"]:
            dataset_params["quad_split_skew_index"] = num_variables_per_beam
            num_variables_per_beam += 1
    # power (time-varying?)
    dataset_params["min_power"] = 0.5 # fraction of full power
    dataset_params["power_bool"] = False
    dataset_params["time_varying_pulse"] = False
    if dataset_params["power_bool"]:
        dataset_params["power_index"] = num_variables_per_beam
        if dataset_params["time_varying_pulse"]:
            num_variables_per_beam += dataset_params["num_profiles_per_config"]
        else:
            num_variables_per_beam += 1
    # beamspot
    dataset_params["select_beamspot_bool"] = False
    dataset_params["beamspot_order_default"] = 3.0
    dataset_params["beamspot_radius_default"] = dataset_params['target_radius']
    dataset_params["scan_beamspot_bool"] = False
    dataset_params["beamspot_order_max"] = dataset_params["beamspot_order_default"] * 2.0
    dataset_params["beamspot_radius_min"] = dataset_params["beamspot_radius_default"] * 0.25
    dataset_params["beamspot_radius_max"] = dataset_params["beamspot_radius_default"] * 1.2
    if (dataset_params["select_beamspot_bool"] or dataset_params["scan_beamspot_bool"]):
        dataset_params["beamspot_bool"] = True
    else:
        dataset_params["beamspot_bool"] = False
    if (dataset_params["select_beamspot_bool"] and dataset_params["scan_beamspot_bool"]):
        sys.exit("Either scan or select a specific beamspot. Can't do both.")
    if dataset_params["scan_beamspot_bool"]:
        dataset_params["beamspot_order_index"] = num_variables_per_beam
        num_variables_per_beam += 1
        dataset_params["beamspot_radius_index"] = num_variables_per_beam
        num_variables_per_beam += 1
    # bandwidth
    dataset_params["select_bandwidth_bool"] = False
    dataset_params["bandwidth_num_spectral_lines_default"] = 20
    dataset_params["bandwidth_percentage_width_default"] = 1.
    dataset_params["scan_bandwidth_bool"] = False
    dataset_params["bandwidth_num_spectral_lines_max"] = dataset_params["bandwidth_num_spectral_lines_default"] * 2
    dataset_params["bandwidth_percentage_width_max"] = dataset_params["bandwidth_percentage_width_default"] * 10.0
    if (dataset_params["scan_bandwidth_bool"] or dataset_params["select_bandwidth_bool"]):
        dataset_params["bandwidth_bool"] = True
    else:
        dataset_params["bandwidth_bool"] = False
    if (dataset_params["bandwidth_bool"] and not dataset_params["run_with_cbet"]):
        sys.exit("Trying to run bandwidth without CBET?!")
    if (dataset_params["scan_bandwidth_bool"] and dataset_params["select_bandwidth_bool"]):
        sys.exit("Either scan or select a specific bandwidth. Can't do both.")
    if dataset_params["scan_bandwidth_bool"]:
        dataset_params["bandwidth_lines_index"] = num_variables_per_beam
        num_variables_per_beam += 1
        dataset_params["bandwidth_percentage_index"] = num_variables_per_beam
        num_variables_per_beam += 1

    dataset_params["num_variables_per_beam"] = num_variables_per_beam
    return dataset_params



def populate_dataset_random_inputs(dataset_params, dataset):
    num_examples = dataset_params["num_examples"]
    num_input_params = dataset_params["num_input_params"]

    random_generator=np.random.default_rng(dataset_params["random_seed"])
    if dataset_params["sampling_method"] == "random":
        print("Random Sampling!")
        sample = random_generator.random((dataset_params["num_examples"], dataset_params["num_input_params"]))
    elif dataset_params["sampling_method"] == "lhs":
        sampler = qmc.LatinHypercube(d=dataset_params["num_input_params"],
                                     strength=1, seed=random_generator, optimization="random-cd")
        sample = sampler.random(n=dataset_params["num_examples"])
    elif (dataset_params["sampling_method"] == "linear") and (num_input_params==2) \
          and (int((num_examples)**(1.0/num_input_params))**num_input_params==int(num_examples)):
        sample = np.zeros((dataset_params["num_examples"], dataset_params["num_input_params"]))
        num_samples_per_param = int((dataset_params["num_examples"])**(1.0/dataset_params["num_input_params"]))
        val_samples_per_param = np.linspace(0.,1.,num_samples_per_param)
        iconfig = 0
        for val_param1 in val_samples_per_param:
            for val_param2 in val_samples_per_param:
                sample[iconfig, 0] = val_param1
                sample[iconfig, 1] = val_param2
                iconfig+=1
    else:
        sys.exit("dataset_params['sampling_method'] not recognised or see source code for type 'linear'")

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

    nrw.save_general_netcdf(dataset_params, sys_params["data_dir"] + "/" + sys_params["dataset_params_filename"])
    nrw.save_general_netcdf(facility_spec, sys_params["data_dir"] + "/" + sys_params["facility_spec_filename"])

    max_parallel = dataset["num_evaluated"]-1
    chkp_marker = 1.0
    filename_trainingdata = sys_params["data_dir"] + "/" + sys_params["trainingdata_filename"]
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
    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"]
    for tind in range(dataset_params["num_profiles_per_config"]):
        sim_dir = "/" + sys_params["sim_dir"] + str(tind)

        if dataset_params["run_plasma_profile"]:
            num_mpi_parallel = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
        else:
            num_mpi_parallel = 1

        loc_bash_parallel_ifriit = sys_params["root_dir"] + "/" + sys_params["bash_parallel_ifriit"]
        subprocess.check_call(["./" + loc_bash_parallel_ifriit, config_location, sim_dir, str(min_parallel), str(max_parallel), str(num_mpi_parallel), str(sys_params["num_openmp_parallel"])])

    dataset = nrw.retrieve_xtrain_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)
    return dataset



def copy_python_files(sys_params):
    path_bash_file = sys_params["data_dir"]+"/"+sys_params["bash_parallel_ifriit"]
    file_exists = os.path.exists(sys_params["data_dir"]+"/"+sys_params["bash_parallel_ifriit"])
    if not file_exists:
        shutil.copy2(sys_params["root_dir"]+"/"+sys_params["bash_parallel_ifriit"],
                     path_bash_file)
    st = os.stat(path_bash_file)
    os.chmod(path_bash_file, st.st_mode | stat.S_IEXEC)

    shutil.copytree(sys_params["root_dir"]+"/"+sys_params["ifriit_run_files_dir"],
                    sys_params["data_dir"]+"/"+sys_params["ifriit_run_files_dir"], dirs_exist_ok=True)

    files = glob.iglob(os.path.join(sys_params["root_dir"]+"/"+sys_params["python_dir"], "*.py"))

    file_exists = os.path.exists(sys_params["data_dir"]+"/"+sys_params["python_dir"])
    if not file_exists:
        os.makedirs( sys_params["data_dir"]+"/"+sys_params["python_dir"])

    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, sys_params["data_dir"]+"/"+sys_params["python_dir"])



def main(argv):
    sys_params = define_system_params(argv[1])

    print(argv[3])
    run_type = str(argv[3]).split("=")[1]

    if (run_type=="init") or (run_type=="full"):
        dataset_params, facility_spec = define_dataset_params(int(argv[2]), sys_params)

        dataset = define_dataset(dataset_params)
        dataset = populate_dataset_random_inputs(dataset_params, dataset)

        deck_gen_params = idg.define_deck_generation_params(dataset_params, facility_spec)
        deck_gen_params = idg.create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)
        idg.save_data_dicts_to_file(sys_params, dataset, dataset_params, deck_gen_params, facility_spec)
        copy_python_files(sys_params)

    if (run_type=="restart") or (run_type=="full"):
        dataset, dataset_params, deck_gen_params, facility_spec = idg.load_data_dicts_from_file(sys_params)
        generate_training_data(dataset, dataset_params, sys_params, facility_spec)

    if (run_type=="reload"):
        dataset, dataset_params, deck_gen_params, facility_spec = idg.load_data_dicts_from_file(sys_params)
        min_parallel = int(argv[2])
        max_parallel = dataset_params["num_examples"]
        dataset = nrw.retrieve_xtrain_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)
        idg.save_data_dicts_to_file(sys_params, dataset, dataset_params, deck_gen_params, facility_spec)


    return dataset, dataset_params, sys_params, facility_spec



if __name__ == "__main__":
    _, _, _, _ = main(sys.argv)

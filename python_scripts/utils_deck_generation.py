import os
import shutil
import numpy as np
import csv
import sys
import healpy_pointings as hpoint
import netcdf_read_write as nrw
import utils_multi as um
import utils_intensity_map as uim


def create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec):
    if (dataset_params["facility"]=="lmj") or (dataset_params["facility"]=="nif"):
        deck_gen_params = create_run_files_pdd(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)
    elif (dataset_params["facility"]=="omega") or (dataset_params['facility']=="custom_facility"):
        deck_gen_params = create_run_files_direct_drive(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)

    generate_run_files(dataset, dataset_params, facility_spec, sys_params, deck_gen_params)
    nrw.save_general_netcdf(deck_gen_params, sys_params["data_dir"] + "/" + sys_params["deck_gen_params_filename"])
    return deck_gen_params

def create_run_files_direct_drive(dataset, deck_gen_params, dataset_params, sys_params, facility_spec):
    num_input_params = dataset_params["num_input_params"]
    num_examples = dataset_params["num_examples"]
    num_vars = dataset_params["num_variables_per_beam"]

    deck_gen_params["power_multiplier"][:,:,:] = 1.0#facility_spec['beams_per_ifriit_beam']
    deck_gen_params['pointings'][:,:] = np.zeros(3)

    for iconfig in range(dataset["num_evaluated"], num_examples):
        ex_params = dataset["input_parameters"][iconfig,:]

        if dataset_params["scan_beamspot_bool"]:
            deck_gen_params["beamspot_order"][iconfig,:] = (dataset_params["beamspot_order_max"] - 1.0) \
                                                           * ex_params[dataset_params["beamspot_order_index"]] + 1.0
            deck_gen_params["beamspot_major_radius"][iconfig,:] = (dataset_params["beamspot_radius_max"]
                                                                  - dataset_params["beamspot_radius_min"]) \
                                                                  * ex_params[dataset_params["beamspot_radius_index"]] \
                                                                  + dataset_params["beamspot_radius_min"]
            deck_gen_params["beamspot_minor_radius"][iconfig,:] = deck_gen_params["beamspot_major_radius"][iconfig,:]
        elif dataset_params["select_beamspot_bool"]:
            deck_gen_params["beamspot_order"][iconfig,:] = dataset_params["beamspot_order_default"]
            deck_gen_params["beamspot_major_radius"][iconfig,:] = dataset_params["beamspot_radius_default"]
            deck_gen_params["beamspot_minor_radius"][iconfig,:] = dataset_params["beamspot_radius_default"]

        if dataset_params["scan_bandwidth_bool"]:
            deck_gen_params["bandwidth_num_spectral_lines"][iconfig] = int((dataset_params["bandwidth_num_spectral_lines_max"] - 2)
                                                                      * ex_params[dataset_params["bandwidth_lines_index"]] + 2)
            deck_gen_params["bandwidth_percentage_width"][iconfig,:] = dataset_params["bandwidth_percentage_width_max"] \
                                                                       ** (ex_params[dataset_params["bandwidth_percentage_index"]]
                                                                       * 2.0 - 1.0)
        elif dataset_params["select_bandwidth_bool"]:
            deck_gen_params["bandwidth_num_spectral_lines"][iconfig] = dataset_params["bandwidth_num_spectral_lines_default"]
            deck_gen_params["bandwidth_percentage_width"][iconfig,:] = dataset_params["bandwidth_percentage_width_default"]

        coord_o = np.zeros(3)
        coord_o[2] = dataset_params['target_radius']
        if dataset_params['custom_beam_groups_bool']:
            num_groups = dataset_params['beams_per_group'][0]
            group_name = dataset_params['beam_group_name_list'][0]
            deck_gen_params['power_multiplier'][:,:,:] = 0.0
            ibeams_in_group = np.where(dataset_params['beamgroup_name'] == group_name)[0]
            print("Group name: '", group_name, "', Beams: ", ibeams_in_group, " with length ", len(ibeams_in_group))
            deck_gen_params['power_multiplier'][:,ibeams_in_group,:] = 1.0
        elif dataset_params["pointing_per_beam_bool"]:
            num_groups = facility_spec['nbeams']
        else:
            num_groups = dataset_params['num_beam_groups']

        if dataset_params["pointings_zooming_bool"] == True :
            pointings_zooming_20_40(iconfig, facility_spec, deck_gen_params)

        if (dataset_params["pointing_bool"] or dataset_params["theta_bool"]):
            for igroup in range(num_groups):
                if dataset_params["pointing_per_beam_bool"] and dataset_params['custom_beam_groups_bool']:
                    group_name = facility_spec['Beam'][igroup]
                    group_slice = [ibeams_in_group[igroup]]
                elif dataset_params["pointing_per_beam_bool"]:
                    group_name = facility_spec['Beam'][igroup]
                    group_slice = [igroup]
                else:
                    group_name = dataset_params['beamgroup_name'][igroup]
                    group_slice = np.where(dataset_params['beamgroup_name'] == group_name)[0]

                il = (igroup*num_vars) % num_input_params
                iu = ((igroup+1)*num_vars-1) % num_input_params + 1
                cone_params = ex_params[il:iu]

                x = cone_params[dataset_params["theta_index"]] * 2.0 - 1.0
                y = cone_params[dataset_params["phi_index"]] * 2.0 - 1.0
                r, offset_phi_group = hpoint.square2disk(x, y)
                offset_theta_group = r * dataset_params["surface_cover_radians"]
                deck_gen_params["sim_params"][iconfig,igroup*num_vars+dataset_params["theta_index"]] = offset_theta_group
                deck_gen_params["sim_params"][iconfig,igroup*num_vars+dataset_params["phi_index"]] = offset_phi_group

                for ibeam in group_slice:
                    port_theta = facility_spec["Theta"][ibeam]
                    port_phi = facility_spec["Phi"][ibeam] - 0.0000001 # issue with port_phi = 270°

                    rotation_matrix = np.matmul(np.matmul(hpoint.rot_mat(port_phi, "z"),
                                                          hpoint.rot_mat(port_theta, "y")),
                                                np.matmul(hpoint.rot_mat(offset_phi_group, "z"),
                                                          hpoint.rot_mat(offset_theta_group, "y")))
                    coord_n = np.matmul(rotation_matrix, coord_o)

                    deck_gen_params["theta_pointings"][iconfig,ibeam] = np.arccos(coord_n[2] / coord_o[2])
                    phi_p = np.arctan2(coord_n[1], coord_n[0])
                    deck_gen_params["phi_pointings"][iconfig,ibeam] = np.where(phi_p < 0.0,  2 * np.pi + phi_p, phi_p)
                    deck_gen_params['pointings'][iconfig,ibeam] = np.array(coord_n)
    return deck_gen_params


def create_run_files_pdd(dataset, deck_gen_params, dataset_params, sys_params, facility_spec):
    num_input_params = dataset_params["num_input_params"]
    num_examples = dataset_params["num_examples"]
    num_vars = dataset_params["num_variables_per_beam"]

    coord_o = np.zeros(3)
    coord_o[2] = dataset_params['target_radius']

    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])

    for iex in range(dataset["num_evaluated"], num_examples):
        ex_params = dataset["input_parameters"][iex,:]
        for igroup in range(dataset_params['num_beam_groups']):
            quad_name = dataset_params['quad_from_each_group'][igroup]
            quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
            quad_start_ind = quad_slice[0]
            group_name = dataset_params['beamgroup_name'][quad_start_ind]
            group_slice = np.where(dataset_params['beamgroup_name'] == group_name)[0]
            quad_list_in_group = list(set(facility_spec["Quad"][group_slice]))

            il = (igroup*num_vars) % num_input_params
            iu = ((igroup+1)*num_vars-1) % num_input_params + 1
            cone_params = ex_params[il:iu]

            if dataset_params["theta_bool"]:
                offset_phi_group = 0.0
                offset_theta_group = (cone_params[dataset_params["theta_index"]] * 2.0 - 1.0) * dataset_params["surface_cover_radians"]
                deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["theta_index"]] = offset_theta_group
            elif dataset_params["pointing_bool"]:
                x = cone_params[dataset_params["theta_index"]] * 2.0 - 1.0
                y = cone_params[dataset_params["phi_index"]] * 2.0 - 1.0
                r, offset_phi_group = hpoint.square2disk(x, y)
                offset_theta_group = r * dataset_params["surface_cover_radians"]
                deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["theta_index"]] = offset_theta_group
                deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["phi_index"]] = offset_phi_group
            else:
                offset_phi_group = 0.0
                offset_theta_group = 0.0

            if dataset_params["defocus_bool"]:
                cone_defocus = cone_params[dataset_params["defocus_index"]] * dataset_params["defocus_range"]
                deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["defocus_index"]] = cone_defocus
            else:
                cone_defocus = dataset_params["defocus_default"]
            deck_gen_params["defocus"][iex,group_slice] = cone_defocus

            cone_power = np.zeros((dataset_params["num_profiles_per_config"]))
            if dataset_params["power_bool"]:
                for tind in range(dataset_params["num_profiles_per_config"]):
                    pind = dataset_params["power_index"] + tind
                    cone_power[tind] = (cone_params[pind] * (1.0 - dataset_params["min_power"]) + dataset_params["min_power"])
                    deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["power_index"] + tind] = cone_power[tind] * facility_spec['beams_per_ifriit_beam']
                    deck_gen_params["power_multiplier"][iex,group_slice,tind] = cone_power[tind] * facility_spec['beams_per_ifriit_beam']
            else:
                deck_gen_params["power_multiplier"][iex,group_slice,:] = facility_spec['beams_per_ifriit_beam']

            for quad_name in quad_list_in_group:
                quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
                beam_names = facility_spec['Beam'][quad_slice]

                deck_gen_params["port_centre_theta"][quad_slice] = np.mean(facility_spec["Theta"][quad_slice])
                deck_gen_params["port_centre_phi"][quad_slice] = np.mean(facility_spec["Phi"][quad_slice])

                offset_theta = offset_theta_group
                offset_phi = offset_phi_group
                if (deck_gen_params["port_centre_theta"][quad_slice[0]] < (np.pi / 2.)):
                    if dataset_params["theta_bool"]:
                        offset_theta = - offset_theta_group
                    elif dataset_params["pointing_bool"]:
                        if dataset_params["hemisphere_symmetric"]:
                            offset_phi = np.pi - offset_phi_group # Symmetric
                        else:
                            offset_phi = (offset_phi_group + np.pi) % (2.0 * np.pi) # anti-symmetric

                if dataset_params["quad_split_bool"]:
                    quad_split_default = (np.sqrt(np.var(facility_spec["Phi"][quad_slice])) + np.sqrt(np.var(facility_spec["Theta"][quad_slice])))
                    quad_split_magnitude = cone_params[dataset_params["quad_split_index"]] * dataset_params["quad_split_range"] * quad_split_default
                    deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["quad_split_index"]] = quad_split_magnitude
                    if dataset_params["quad_split_skew_bool"]:
                        skew_factor = np.pi / 4.0 - np.pi / 2.0 * cone_params[dataset_params["quad_split_skew_index"]]
                        deck_gen_params["sim_params"][iex,igroup*num_vars+dataset_params["quad_split_skew_index"]] = skew_factor
                    else:
                        skew_factor = 0.0

                for ind in quad_slice:
                    port_theta = deck_gen_params["port_centre_theta"][ind]
                    port_phi = deck_gen_params["port_centre_phi"][ind]

                    if dataset_params["quad_split_bool"]:
                        theta_split = facility_spec["Theta"][ind] - port_theta
                        phi_split = facility_spec["Phi"][ind] - port_phi

                        quad_skew_angle = np.arctan2(theta_split, phi_split) + skew_factor

                        beam_theta = port_theta + quad_split_magnitude * np.sin(quad_skew_angle)
                        beam_phi = port_phi + quad_split_magnitude * np.cos(quad_skew_angle)
                    else:
                        beam_theta = port_theta
                        beam_phi = port_phi

                    rotation_matrix = np.matmul(np.matmul(hpoint.rot_mat(beam_phi, "z"),
                                                          hpoint.rot_mat(beam_theta, "y")),
                                                np.matmul(hpoint.rot_mat(offset_phi, "z"),
                                                          hpoint.rot_mat(offset_theta, "y")))
                    coord_n = np.matmul(rotation_matrix, coord_o)

                    deck_gen_params["theta_pointings"][iex,ind] = np.arccos(coord_n[2] / coord_o[2])
                    phi_p = np.arctan2(coord_n[1], coord_n[0])
                    deck_gen_params["phi_pointings"][iex,ind] = np.where(phi_p < 0.0,  2 * np.pi + phi_p, phi_p)
                    deck_gen_params['pointings'][iex,ind] = np.array(coord_n)

    return deck_gen_params



def load_data_dicts_from_file(sys_params):

    data_dir = sys_params["data_dir"]
    dataset_params = nrw.read_general_netcdf(data_dir + "/" + sys_params["dataset_params_filename"])
    facility_spec = nrw.read_general_netcdf(data_dir + "/" + sys_params["facility_spec_filename"])
    dataset = nrw.read_general_netcdf(data_dir + "/" + sys_params["trainingdata_filename"])
    deck_gen_params = nrw.read_general_netcdf(data_dir + "/" + sys_params["deck_gen_params_filename"])

    return dataset, dataset_params, deck_gen_params, facility_spec



def save_data_dicts_to_file(sys_params, dataset, dataset_params, deck_gen_params, facility_spec):

    data_dir = sys_params["data_dir"]
    nrw.save_general_netcdf(dataset, data_dir + "/" + sys_params["trainingdata_filename"])
    nrw.save_general_netcdf(dataset_params, data_dir + "/" + sys_params["dataset_params_filename"])
    nrw.save_general_netcdf(facility_spec, data_dir + "/" + sys_params["facility_spec_filename"])
    nrw.save_general_netcdf(deck_gen_params, data_dir + "/" + sys_params["deck_gen_params_filename"])

    return



def define_deck_generation_params(dataset_params, facility_spec):
    num_examples = dataset_params["num_examples"]
    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])

    deck_gen_params = dict()
    deck_gen_params["non_expand_keys"] = ["non_expand_keys", "port_centre_theta", "port_centre_phi"]

    deck_gen_params["port_centre_theta"] = np.zeros(num_ifriit_beams)
    deck_gen_params["port_centre_phi"] = np.zeros(num_ifriit_beams)

    deck_gen_params['pointings'] = np.zeros((num_examples, num_ifriit_beams, 3))
    deck_gen_params["theta_pointings"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["phi_pointings"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["defocus"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["power_multiplier"] = np.zeros((num_examples, num_ifriit_beams, dataset_params["num_profiles_per_config"]))
    deck_gen_params["sim_params"] = np.zeros((num_examples, dataset_params["num_input_params"]*2))
    deck_gen_params["beamspot_order"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["beamspot_major_radius"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["beamspot_minor_radius"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["bandwidth_num_spectral_lines"] = np.zeros(num_examples, dtype=np.uint32)
    deck_gen_params["bandwidth_percentage_width"] = np.zeros((num_examples, num_ifriit_beams))

    return deck_gen_params

def import_direct_drive_config(sys_params, dataset_params):
    facility_spec = dict()

    facility_spec['num_quads'] = 0
    facility_spec['num_cones'] = 0
    facility_spec['beams_per_ifriit_beam'] = 1
    dataset_params['num_beam_groups'] = 1

    if (dataset_params['facility']=="custom_facility"):
        facility_spec['ifriit_facility_name'] = "cpm48" #"cpm48" #"cpm72" #"t11_b72" #"ico80"
        if facility_spec['ifriit_facility_name'] == "cpm48":
            facility_spec = generate_facility_cpm48(facility_spec)
        else:
            facility_spec = load_facility_csv(sys_params, facility_spec)
        facility_spec["focal_length_metres"] = 10.0
        facility_spec['Beam'] = list(map(str, np.arange(facility_spec['nbeams'])))
    else:
        facility_spec['nbeams'] = 60
        facility_spec['ifriit_facility_name'] = "OMEGA60"
        facility_spec = load_facility_csv(sys_params, facility_spec)
        facility_spec["Theta"] = np.radians(facility_spec["Theta"])
        facility_spec["Phi"] = np.radians(facility_spec["Phi"])

        facility_spec['Beam'] = [None] * facility_spec['nbeams']
        for ibeam in range(facility_spec['nbeams']):
            facility_spec['Beam'][ibeam] = str(int(10 + ibeam))
        dataset_params['beamgroup_name'] = np.array(np.zeros(int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"])), dtype='<U20')
        #inconvenient en le laissant ici : on risque d'avoir des problèmes avec le "num?
        dataset_params["beamspot_predef_beam_name"] = np.array(np.zeros(int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"])), dtype='<U20')
        dataset_params["beamspot_predef_beam_name"][:] = "SG5"
        if dataset_params['custom_beam_groups_bool']==True:
            facility_spec, dataset_params  = group_beams_by_group(facility_spec,dataset_params)
        else:
            dataset_params['beamgroup_name'][:] = "group1"
    return facility_spec


def group_beams_by_group(facility_spec, dataset_params):
    dataset_params['beamgroup_name'][:] = "None"
    if dataset_params["custom_beam_groups_name"] == "24+36":
        set_group_36 = [30,45,19,50,62,54,39,23,69,51,38,21,27,18,53,42,49,34,37,24,59,60,41,15,20,66,58,65,52,33,31,11,26,17,56,61]
        set_group_24 = [10,12,13,14,16,22,25,28,29,32,35,36,40,43,44,46,47,48,55,57,63,64,67,68]
        for idx in range(10, 10 + facility_spec['nbeams']):
            if idx in set_group_36:
                dataset_params['beamgroup_name'][idx - 10] = "group_36"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "PD-R1"
            else:
                dataset_params['beamgroup_name'][idx - 10] = "group_24"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "SG5"

    set_group_40 = np.concatenate((np.linspace(10,29,20), np.linspace(40,59,20)))
    set_group_20 = np.concatenate((np.linspace(30,39,10), np.linspace(60,69,10)))
    if dataset_params["custom_beam_groups_name"] == "20+40":
        for idx in range(10, 10 + facility_spec['nbeams']):
            if idx in set_group_40:
                dataset_params['beamgroup_name'][idx - 10] = "group_40"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "SG5"
            else:
                dataset_params['beamgroup_name'][idx - 10] = "group_20"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "PD-R1"
    if dataset_params["custom_beam_groups_name"] == "just_40":
        for idx in range(10, 10 + facility_spec['nbeams']):
            if idx in set_group_40:
                dataset_params['beamgroup_name'][idx - 10] = "group_40"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "SG5"
    if dataset_params["custom_beam_groups_name"] == "just_20":
        for idx in range(10, 10 + facility_spec['nbeams']):
            if idx in set_group_20:
                dataset_params['beamgroup_name'][idx - 10] = "group_20"
                dataset_params["beamspot_predef_beam_name"][idx - 10] = "PD-R1"

    beam_group_name_list = list(set(dataset_params['beamgroup_name']))
    beam_group_name_list = [i for i in beam_group_name_list if i != "None"]
    dataset_params['beam_group_name_list'] = np.array(beam_group_name_list, dtype='<U20')

    dataset_params['num_beam_groups'] = len(dataset_params['beam_group_name_list'])
    dataset_params['beams_per_group'] = np.array(np.zeros((int(dataset_params['num_beam_groups']))), dtype='int8')

    for igroup in range(dataset_params['num_beam_groups']):
        group_name = dataset_params['beam_group_name_list'][igroup]
        dataset_params['beams_per_group'][igroup] = len(np.where(dataset_params['beamgroup_name'] == group_name)[0])
    print(dataset_params['beam_group_name_list'], dataset_params['num_beam_groups'],  dataset_params['beams_per_group'])
    return facility_spec, dataset_params

def pointings_zooming_20_40(iconfig,facility_spec,deck_gen_params):
    """moi j'importe un r pointing, un theta pointing et un phi pointing"""

    path = "../facility_config_files/pointing_zooming_20_40_energy.csv"
    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])

    phi = []
    theta = []
    r = []
    power = []
    with open(path) as f:
        reader = csv.DictReader(f,delimiter=";")
        for row in reader:
            row_ptg_phi = row['ptg_phi'].replace(',', '.')
            phi.append(float(row_ptg_phi))
            row_ptg_theta = row['ptg_theta'].replace(',', '.')
            theta.append(float(row_ptg_theta))
            row_ptg_radius = row['ptg_radius'].replace(',', '.')
            r.append(float(row_ptg_radius))
            row_energy = row['energy'].replace(',', '.')
            power.append(float(row_energy))

    phi = np.array(phi)
    theta = np.array(theta)
    r = np.array(r)
    power = np.array(power)
    power = power/np.sum(power)

    phi = np.deg2rad(phi)
    theta  = np.deg2rad(theta)

    for j in range(num_ifriit_beams):
        deck_gen_params['pointings'][iconfig,j][0] = r[j]*np.cos(phi[j])*np.sin(theta[j])
        deck_gen_params['pointings'][iconfig,j][1] = r[j]*np.sin(phi[j])*np.sin(theta[j])
        deck_gen_params['pointings'][iconfig,j][2] = r[j]*np.cos(theta[j])
        deck_gen_params['power_multiplier'][iconfig,j,:] = deck_gen_params['power_multiplier'][iconfig,j,:] * power[j]
    return deck_gen_params

def generate_facility_cpm48(facility_spec):
    facility_spec['nbeams'] = 48
    facility_spec["Theta"] = np.zeros((facility_spec['nbeams']))
    facility_spec["Phi"] = np.zeros((facility_spec['nbeams']))

    theta_i = [21.24302, 43.64296, 51.30717, 69.91959, 72.99624, 83.35323]
    phi_i = [0.0, 43.69981, 86.62623, 25.70934, 59.45306, 88.40802]
    num_i = 6
    num_m = 2
    num_l = 4
    j = 0
    for i in range(num_i):
        for m in range(num_m):
            for l in range(num_l):
                facility_spec["Theta"][j] = (-1.0)**m * theta_i[i] + 180.0 * m
                facility_spec["Phi"][j] = (-1.0)**m * phi_i[i] + 37.2604 * m + 90.0 * l
                j+=1
    facility_spec["Theta"] = np.radians(facility_spec["Theta"])
    facility_spec["Phi"] = np.radians(facility_spec["Phi"])
    return facility_spec


def load_facility_csv(sys_params, facility_spec):
    path_facility_configs = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/"
    f=open(path_facility_configs + facility_spec['ifriit_facility_name']+"_theta_phi_rad.txt", "r")
    reader = csv.reader(f, delimiter=' ')

    facility_spec["Theta"] = []
    facility_spec["Phi"] = []
    j = 0
    for row in reader:
        facility_spec["Theta"].append(float(row[0]))
        facility_spec["Phi"].append(float(row[1]))
        j=j+1
    f.close()
    facility_spec["Theta"] = np.array(facility_spec["Theta"])
    facility_spec["Phi"] = np.array(facility_spec["Phi"])
    facility_spec['nbeams'] = np.shape(facility_spec["Theta"])[0]
    return facility_spec


def config_formatting(facility_spec):
    facility_spec["PR"] = np.array(facility_spec["PR"], dtype='i')
    condition = ((facility_spec['ifriit_facility_name'] == "LMJ") and (len(facility_spec["Beam"]) == 160))
    if condition :
        facility_spec["Beam"] = np.array(facility_spec["Beam"], dtype='<U5')
    else :
        facility_spec["Beam"] = np.array(facility_spec["Beam"], dtype='<U4')
    facility_spec["Quad"] = np.array(facility_spec["Quad"], dtype='<U4')
    facility_spec["Cone"] = np.array(facility_spec["Cone"])
    facility_spec["Theta"] = np.radians(facility_spec["Theta"])
    facility_spec["Phi"] = np.radians(facility_spec["Phi"])

    return facility_spec


def import_nif_config(sys_params, dataset_params):
    facility_spec = dict()

    facility_spec['nbeams'] = 192
    facility_spec['ifriit_facility_name'] = "NIF"
    facility_spec['num_quads'] = 48
    facility_spec['num_cones'] = 8
    facility_spec["beams_per_ifriit_beam"] = 1 # fuse quads?

    filename1 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/NIF_UpperBeams.txt"
    filename2 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/NIF_LowerBeams.txt"
    facility_spec = config_read_csv(facility_spec, filename1, filename2)
    facility_spec = config_formatting(facility_spec)

    if dataset_params["bool_group_beams_by_cone"]:
        dataset_params['quad_from_each_group'] = np.array(('Q15T', 'Q13T', 'Q14T', 'Q11T'), dtype='<U4')
        facility_spec, dataset_params = group_beams_by_cone(facility_spec, dataset_params)
    else:
        sys.exit("Must 'use_group_by_cone' for NIF, no other option written yet")

    return facility_spec, dataset_params


def import_lmj_config(sys_params, dataset_params):
    facility_spec = dict()

    facility_spec['nbeams'] = 160
    facility_spec['ifriit_facility_name'] = "LMJ"
    facility_spec['num_quads'] = 40
    facility_spec['num_cones'] = 4

    facility_spec["beams_per_ifriit_beam"] = 1
    filename1 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/LMJ_UpperBeams.txt"
    filename2 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/LMJ_LowerBeams.txt"
    facility_spec = config_read_csv(facility_spec, filename1, filename2)
    facility_spec = config_formatting(facility_spec)

    if dataset_params["bool_group_beams_by_cone"]:
        dataset_params['quad_from_each_group'] = np.array(('Q28H', 'Q10H'), dtype='<U4')
        facility_spec, dataset_params = group_beams_by_cone(facility_spec, dataset_params)
    else:
        facility_spec, dataset_params = group_beams_subcones_lmj(facility_spec, dataset_params)

    return facility_spec, dataset_params


def group_beams_by_cone(facility_spec, dataset_params):
    dataset_params['num_beam_groups'] = int(facility_spec['num_cones'] / 2) # hemisphere symmetry
    dataset_params['beams_per_group'] = np.array(np.zeros((int(dataset_params['num_beam_groups']))), dtype='int8')
    dataset_params['beamgroup_name'] = np.array(np.zeros((int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"]))), dtype='<U20')

    for igroup in range(dataset_params['num_beam_groups']):
        quad_name = dataset_params['quad_from_each_group'][igroup]
        quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
        quad_start_ind = quad_slice[0]
        cone_name = facility_spec['Cone'][quad_start_ind]
        for ibeam in range(int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"])):
            check_cone_name = facility_spec['Cone'][ibeam]
            if (abs(cone_name - check_cone_name) < 1.0) or (abs(cone_name + check_cone_name - 180.) < 1.0):
                dataset_params['beamgroup_name'][ibeam] = cone_name
        dataset_params['beams_per_group'][igroup] = int(np.count_nonzero(dataset_params['beamgroup_name'] == str(cone_name)) * facility_spec["beams_per_ifriit_beam"])

    return facility_spec, dataset_params


def group_beams_subcones_lmj(facility_spec, dataset_params):
    # hemisphere symmetry
    dataset_params['num_beam_groups'] = 4
    sub_cone_list = np.array(np.zeros((4, 5)), dtype='<U4')
    dataset_params['beams_per_group'] = np.array(np.zeros((int(dataset_params['num_beam_groups']))), dtype='int8')
    dataset_params['beamgroup_name'] = np.array(np.zeros((int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"]))), dtype='<U20')
    sub_cone_list[0,:] = ["06", "11", "17", "24", "28"]
    sub_cone_list[1,:] = ["03", "07", "14", "20", "25"]
    sub_cone_list[2,:] = ["02", "09", "13", "21", "26"]
    sub_cone_list[3,:] = ["05", "10", "18", "22", "29"]
    cone_list = list(set(facility_spec['Cone']))
    cone_index_upper = np.where(np.array(cone_list) < 90)[0]

    for igroup in range(dataset_params['num_beam_groups']):
        cone_name = cone_list[cone_index_upper[igroup%(int(facility_spec['num_cones']/2))]]
        group_label = str(cone_name) + "_subcone" + str(igroup)
        for ibeam in range(int(facility_spec['nbeams'] / facility_spec["beams_per_ifriit_beam"])):
            check_cone_name = facility_spec['Cone'][ibeam]
            for iquad_name in range(len(sub_cone_list[igroup])):
                if (abs(cone_name - check_cone_name) < 1.0) and (sub_cone_list[igroup][iquad_name] in facility_spec['Quad'][ibeam]):
                    dataset_params['beamgroup_name'][ibeam] = group_label
                elif (abs(cone_name + check_cone_name - 180.) < 1.0) and (sub_cone_list[(igroup+1)%dataset_params['num_beam_groups']][iquad_name] in facility_spec['Quad'][ibeam]):
                    dataset_params['beamgroup_name'][ibeam] = group_label
        dataset_params['beams_per_group'][igroup] = int(np.count_nonzero(dataset_params['beamgroup_name'] == group_label) * facility_spec["beams_per_ifriit_beam"])

    subcone_list_unique_names = list(set(dataset_params['beamgroup_name']))
    dataset_params['quad_from_each_group'] = np.array(np.zeros(len(subcone_list_unique_names)), dtype='<U4')
    for isubcone in range(len(subcone_list_unique_names)):
        iquad = np.where(dataset_params['beamgroup_name'] == subcone_list_unique_names[isubcone])[0][0]
        dataset_params['quad_from_each_group'][isubcone] = facility_spec["Quad"][iquad]
    return facility_spec, dataset_params


def config_read_csv(facility_spec, filename1, filename2):
    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
    j = -1
    f=open(filename1, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==-1:
            key = row
            for i in range(len(row)):
                facility_spec[row[i]] = [None] * int(num_ifriit_beams)
        else:
            for i in range(len(row)):
                if i < 2:
                    facility_spec[key[i]][j] = row[i]
                elif i < 5:
                    facility_spec[key[i]][j] = float(row[i])
                else:
                    facility_spec[key[i]][j] = int(row[i])
        j=j+1
    f.close()
    f=open(filename2, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==int(num_ifriit_beams/2.0):
            key = row
        else:
            for i in range(len(row)):
                if i < 2:
                    facility_spec[key[i]][j-1] = row[i]
                elif i < 5:
                    facility_spec[key[i]][j-1] = float(row[i])
                else:
                    facility_spec[key[i]][j-1] = int(row[i])
        j=j+1
    f.close()
    return facility_spec



def generate_run_files(dataset, dataset_params, facility_spec, sys_params, deck_gen_params):

    for iconfig in range(dataset["num_evaluated"], dataset_params["num_examples"]):
        if (iconfig==dataset["num_evaluated"]):
            config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
            file_exists = os.path.exists(config_location)
            if not file_exists:
                os.makedirs(config_location)

            for tind in range(dataset_params["num_profiles_per_config"]):
                run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
                isExist = os.path.exists(run_location)

                if not isExist:
                    os.makedirs(run_location)

                loc_ifriit_runfiles = sys_params["root_dir"] + "/" + sys_params["ifriit_run_files_dir"]
                if dataset_params["bandwidth_bool"]:
                    shutil.copyfile(loc_ifriit_runfiles + "/" + sys_params["ifriit_binary_filename"] + "_bandwidth",
                                    run_location + "/" + sys_params["ifriit_binary_filename"])
                else:
                    shutil.copyfile(loc_ifriit_runfiles + "/" + sys_params["ifriit_binary_filename"],
                                    run_location + "/" + sys_params["ifriit_binary_filename"])

            if (dataset_params["plasma_profile_source"] == "default") and dataset_params["run_plasma_profile"]:
                for tind in range(dataset_params["num_profiles_per_config"]):
                    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
                    shutil.copyfile(sys_params["root_dir"] + "/" +
                                    sys_params["plasma_profile_dir"] + "/" +
                                    sys_params["plasma_profile_nc"],
                                    run_location + "/" + sys_params["plasma_profile_nc"])
            elif dataset_params["plasma_profile_source"] == "multi":
                ind_interface_dt_ch = [0]
                print("!!! Hardcoded hydro evaluation time DT-CH interface at cell:" + str(ind_interface_dt_ch)+" !!!")
                path = sys_params["data_dir"] + "/" + sys_params["multi_dir"]
                multi_data = um.multi_read_ascii(path+"/"+sys_params["multi_output_ascii_filename"])
                multi_data = um.read_inputs(path+"/"+sys_params["multi_input_filename"], multi_data)
                for tind in range(dataset_params["num_profiles_per_config"]):
                    itime_multi = np.argmin(np.abs(multi_data["time"]*1.e9-dataset_params["plasma_profile_times"][tind]))
                    multi_nc, ncells, nmat = um.multi2ifriit_inputs(multi_data, itime_multi, ind_interface_dt_ch)

                    n_crit = um.critical_density(wavelength_l=multi_data["wavelength"])
                    dataset_params['laser_wavelength_nm'] = multi_data["wavelength"] * 1.0e9 # m to nm
                    zero_crossings = np.where(np.diff(np.sign(multi_nc["ne"][0,:] - n_crit)))[0]
                    if len(zero_crossings)==0:
                        ind_critical = len(multi_nc["ne"][0,:])-1
                    else:
                        ind_critical = zero_crossings[-1]
                    dataset_params['illumination_evaluation_radii'][tind] = multi_nc["xs"][ind_critical]

                    if dataset_params["run_plasma_profile"]:
                        config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
                        run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
                        nrw.save_general_netcdf(multi_nc, run_location + "/" + sys_params["plasma_profile_nc"],
                                                extra_dimension={'x': ncells, 'z':1, 'nel':nmat})

                    multi_laser_pulse_per_beam(iconfig, tind, sys_params, facility_spec, dataset_params, multi_data)
        else:
            config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
            file_exists = os.path.exists(config_location)
            if file_exists:
                shutil.rmtree(config_location)
            shutil.copytree(sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(dataset["num_evaluated"]),
                            config_location)

        for tind in range(dataset_params["num_profiles_per_config"]):
            dataset["power_emitted"][iconfig,tind] = uim.power_emitted(iconfig, tind, dataset_params, facility_spec, deck_gen_params)
            generate_input_deck(iconfig, tind, dataset_params, facility_spec, sys_params, deck_gen_params)
            generate_input_pointing_and_pulses(iconfig, tind, dataset_params, facility_spec, sys_params, deck_gen_params)



def multi_laser_pulse_per_beam(iconfig, tind, sys_params, facility_spec, dataset_params, multi_data):
    nbeams = facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam']

    path = sys_params["data_dir"] + "/" + sys_params["multi_dir"]
    f = open(path + "/" + sys_params["multi_pulse_name"], "r")
    old_pulse = f.read()
    f.close()
    old_pulse = old_pulse.splitlines()
    old_pulse_label = old_pulse[0]
    old_pulse_data = old_pulse[1:]

    old_pulse_nlines = len(old_pulse_data)
    old_pulse_time = np.zeros((old_pulse_nlines))
    old_pulse_power = np.zeros((old_pulse_nlines))

    i=0
    for line in old_pulse_data:
        old_pulse_data = line.split( )
        old_pulse_time[i] = old_pulse_data[0]
        old_pulse_power[i] = old_pulse_data[1]
        i+=1

    #print("Using laser multiplier {} for ifriit. Appropriate for scaling, not for CBET".format(multi_data["laser_time_multiplier"]))
    #new_pulse_time = old_pulse_time * multi_data["laser_time_multiplier"]
    #new_pulse_power = old_pulse_power / nbeams * multi_data["laser_power_multiplier"]
    new_pulse_time = old_pulse_time
    new_pulse_power = old_pulse_power / nbeams

    normalised_timings = new_pulse_time - dataset_params["plasma_profile_times"][tind]
    normalised_timings2 = abs(normalised_timings)-normalised_timings
    interp_ind = np.argmin(normalised_timings2[np.nonzero(normalised_timings2)])

    interp_length = abs(normalised_timings[interp_ind]) + abs(normalised_timings[interp_ind+1])
    dataset_params['default_beam_power_TW'][tind] = (abs(normalised_timings[interp_ind]) / interp_length * new_pulse_power[interp_ind+1] +
                                            abs(normalised_timings[interp_ind+1]) / interp_length * new_pulse_power[interp_ind])

    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
    f = open(run_location + "/" + sys_params["ifriit_pulse_name"], "w")
    #f.write(old_pulse_label + "\n")

    new_pulse_nlines = len(new_pulse_time)
    for iline in range(new_pulse_nlines):
        f.write("{:4.5f} {:4.5f} \n".format(new_pulse_time[iline],new_pulse_power[iline]))
    f.close()



def generate_input_deck(iconfig, tind, dataset_params, facility_spec, sys_params, deck_gen_params):
    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)

    loc_ifriit_runfiles = sys_params["root_dir"] + "/" + sys_params["ifriit_run_files_dir"]
    base_input_txt_loc = loc_ifriit_runfiles + "/" + sys_params["ifriit_input_name"]

    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
    with open(base_input_txt_loc) as old_file:
        with open(run_location+"/ifriit_inputs.txt", "w") as new_file:
            for line in old_file:
                if "NBEAMS" in line:
                    if dataset_params['custom_beam_groups_bool']:
                        nbeams = np.sum(dataset_params['beams_per_group'])
                    else:
                        nbeams = num_ifriit_beams
                    new_file.write("    NBEAMS                      = " + str(nbeams) + ",\n")
                elif "DIAGNOSE_INPUT_BEAMS_RADIUS_UM" in line:
                    new_file.write("    DIAGNOSE_INPUT_BEAMS_RADIUS_UM = " + str(dataset_params['illumination_evaluation_radii'][tind]) + "d0,\n")
                elif "CBET = .FALSE.," in line:
                    if dataset_params["run_with_cbet"]:
                        new_file.write("    CBET = .TRUE.,\n")
                        if dataset_params["bandwidth_bool"]:
                            new_file.write("    BANDWIDTH_NLINES = {},\n".format(deck_gen_params["bandwidth_num_spectral_lines"][iconfig]))
                    else:
                        new_file.write("    CBET = .FALSE.,\n")
                elif "RMAX                    = 6000.d0," in line:
                    new_file.write("    RMAX                    = {:.5f}d0,\n".format(dataset_params['target_radius'] * 5.0))
                else:
                    new_file.write(line)



def generate_input_pointing_and_pulses(iconfig, tind, dataset_params, facility_spec, sys_params, deck_gen_params):
    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)

    with open(run_location+'/ifriit_inputs.txt','a') as f:
        for j in range(int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])):
            if deck_gen_params['power_multiplier'][iconfig,j,tind] == 0.0:
                pass
            else:
                f.write('&BEAM\n')
                f.write('    LAMBDA_NM           = {:.10f}d0,\n'.format(dataset_params['laser_wavelength_nm']))
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(deck_gen_params['pointings'][iconfig,j][0],deck_gen_params['pointings'][iconfig,j][1],deck_gen_params['pointings'][iconfig,j][2]))
                ## Need to determine per beam pulses with cone fractions in order to use this method
                #if dataset_params["plasma_profile_source"] == "multi":
                #    f.write('    POWER_PROFILE_FILE_TW_NS = "'+sys_params["ifriit_pulse_name"]+'"\n')
                #    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dataset_params["plasma_profile_times"][tind]))
                #else:
                f.write('    P0_TW               = {:.10f}d0,\n'.format(deck_gen_params['power_multiplier'][iconfig,j,tind] * dataset_params['default_beam_power_TW'][tind]))

                if (dataset_params['facility'] == "custom_facility"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(facility_spec["Theta"][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(facility_spec["Phi"][j])))
                    f.write('    FOCAL_M             = {:.10f}d0,\n'.format(facility_spec["focal_length_metres"]))
                else:
                    beam = facility_spec["Beam"][j]
                    if (dataset_params['facility'] == "nif"):
                        cone_name = facility_spec["Cone"][j]
                        if ((abs(cone_name - 23.5) <= 1.0)) or (abs(cone_name - 156.5) <= 1.0):
                            cpp="NIF-inner-23"
                        elif ((abs(cone_name - 30) <= 1.0)) or (abs(cone_name - 150) <= 1.0):
                            cpp="NIF-inner-30"
                        elif ((abs(cone_name - 44.5) <= 1.0)) or (abs(cone_name - 135.5) <= 1.0):
                            cpp="NIF-outer-44"
                        elif ((abs(cone_name - 50) <= 1.0)) or ((abs(cone_name - 130) <= 1.0)):
                            cpp="NIF-outer-50"
                        else:
                            sys.exit("Unrecognised cpp/cone name ", cone_name)
                    elif (dataset_params['facility'] == "lmj"):
                        cone_name = facility_spec["Cone"][j]
                        if (abs(cone_name - 49.0) <= 1.0) or (abs(cone_name - 131.0) <= 1.0):
                            cpp="LMJ-A"
                        elif (abs(cone_name - 33.0) <= 1.0) or (abs(cone_name - 147.0) <= 1.0):
                            cpp="LMJ-B"
                        else:
                            sys.exit("Unrecognised cpp/cone name ", cone_name)
                    elif (dataset_params['facility'] == "omega"):
                        cpp=dataset_params["beamspot_predef_beam_name"][j]
                    f.write('    PREDEF_FACILITY     = "'+facility_spec['ifriit_facility_name']+'"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')

                if dataset_params["beamspot_bool"]:
                    f.write('    LAW                  = 1,\n')
                    f.write('    SG                  = {:.6f}d0,\n'.format(deck_gen_params["beamspot_order"][iconfig,j]))
                    f.write('    RAD_1_UM            = {:.6f}d0,\n'.format(deck_gen_params["beamspot_major_radius"][iconfig,j]))
                    f.write('    RAD_2_UM            = {:.6f}d0,\n'.format(deck_gen_params["beamspot_minor_radius"][iconfig,j]))
                else:
                    f.write('    PREDEF_CPP          = "'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(deck_gen_params['defocus'][iconfig,j]))
                if dataset_params["bandwidth_bool"]:
                    f.write('    BANDWIDTH_DELTA_OM_OM_PERCENT = {:.10f}d0,\n'.format(deck_gen_params["bandwidth_percentage_width"][iconfig,j]))

                if dataset_params["fuse_quad_bool"]:
                    f.write('    FUSE_QUADS          = .TRUE.,\n')
                    f.write('    FUSE_BY_POINTINGS   = .TRUE.,\n')
                else:
                    f.write('    FUSE_QUADS          = .FALSE.,\n')

                if 'xy-mispoint' in deck_gen_params.keys():
                    f.write('    XY_MISPOINT_UM      = {:.10f}d0,{:.10f}d0,\n'.format(deck_gen_params['xy-mispoint'][iconfig,j][0],deck_gen_params['xy-mispoint'][iconfig,j][1]))
                ##
                f.write('/\n')
                f.write('\n')
                j = j + 1
        f.write('\n')
        f.write('! Last line must not be empty')

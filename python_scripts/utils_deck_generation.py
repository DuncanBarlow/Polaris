import os
import shutil
import numpy as np
import csv
import healpy_pointings as hpoint
import netcdf_read_write as nrw
import utils_multi as um


def create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec):

    num_input_params = dataset_params["num_input_params"]
    num_examples = dataset_params["num_examples"]
    num_vars = dataset_params["num_variables_per_beam"]

    coord_o = np.zeros(3)
    coord_o[2] = facility_spec['target_radius'][0]

    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])

    for iex in range(dataset["num_evaluated"], num_examples):
        ex_params = dataset["input_parameters"][iex,:]
        for icone in range(facility_spec['num_cones']):
            if icone > int(facility_spec['num_cones']/2.0-1):
                bottom_hemisphere = True
            else:
                bottom_hemisphere = False
            il = (icone*num_vars) % num_input_params
            iu = ((icone+1)*num_vars-1) % num_input_params + 1
            cone_params = ex_params[il:iu]

            quad_name = facility_spec['quad_from_each_cone'][icone]
            quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
            quad_start_ind = quad_slice[0]
            cone_name = facility_spec['Cone'][quad_start_ind]
            cone_slice = np.where(facility_spec['Cone'] == cone_name)[0]
            if bottom_hemisphere:
                quad_list_in_cone = [t for t in facility_spec["Quad"][cone_slice] if "B" in t or "L" in t]
            else:
                quad_list_in_cone = [t for t in facility_spec["Quad"][cone_slice] if "T" in t or "U" in t]
            quad_list_in_cone = list(set(quad_list_in_cone))

            x = cone_params[dataset_params["theta_index"]] * 2.0 - 1.0
            y = cone_params[dataset_params["phi_index"]] * 2.0 - 1.0
            r, offset_phi = hpoint.square2disk(x, y)

            if bottom_hemisphere:
                if dataset_params["hemisphere_symmetric"]:
                    offset_phi = np.pi - offset_phi # Symmetric
                else:
                    offset_phi = (offset_phi + np.pi) % (2.0 * np.pi) # anti-symmetric
            offset_theta = r * dataset_params["surface_cover_radians"]
            deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["theta_index"]] = offset_theta
            deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["phi_index"]] = offset_phi

            if dataset_params["defocus_bool"]:
                cone_defocus = cone_params[dataset_params["defocus_index"]] * dataset_params["defocus_range"]
                deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["defocus_index"]] = cone_defocus
            else:
                cone_defocus = dataset_params["defocus_default"]
            deck_gen_params["defocus"][iex,cone_slice] = cone_defocus

            cone_power = np.zeros((dataset_params["num_powers_per_cone"]))
            for tind in range(dataset_params["num_powers_per_cone"]):
                pind = dataset_params["power_index"] + tind
                cone_power[tind] = (cone_params[pind] * (1.0 - dataset_params["min_power"]) + dataset_params["min_power"])
                deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["power_index"] + tind] = cone_power[tind]
                deck_gen_params["p0"][iex,cone_slice,tind] = facility_spec['default_power'] * cone_power[tind] * facility_spec['beams_per_ifriit_beam']

            for quad_name in quad_list_in_cone:
                quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
                beam_names = facility_spec['Beam'][quad_slice]

                deck_gen_params["port_centre_theta"][quad_slice] = np.mean(facility_spec["Theta"][quad_slice])
                deck_gen_params["port_centre_phi"][quad_slice] = np.mean(facility_spec["Phi"][quad_slice])

                if dataset_params["quad_split_bool"]:
                    quad_split_default = (np.sqrt(np.var(facility_spec["Phi"][quad_slice])) + np.sqrt(np.var(facility_spec["Theta"][quad_slice])))
                    quad_split_magnitude = cone_params[dataset_params["quad_split_index"]] * dataset_params["quad_split_range"] * quad_split_default
                    deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["quad_split_index"]] = quad_split_magnitude
                    if dataset_params["quad_split_skew_bool"]:
                        skew_factor = np.pi / 4.0 - np.pi / 2.0 * cone_params[dataset_params["quad_split_skew_index"]]
                        deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["quad_split_skew_index"]] = skew_factor
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

    if sys_params["run_gen_deck"]:
        generate_run_files(dataset, dataset_params, facility_spec, sys_params, deck_gen_params)

    nrw.save_general_netcdf(deck_gen_params, sys_params["data_dir"] + "/" + sys_params["deck_gen_params_filename"])
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
    deck_gen_params["non_expand_keys"] = ["non_expand_keys", "port_centre_theta", "port_centre_phi", "fuse_quads"]

    deck_gen_params["port_centre_theta"] = np.zeros(num_ifriit_beams)
    deck_gen_params["port_centre_phi"] = np.zeros(num_ifriit_beams)
    deck_gen_params["fuse_quads"] = [False]*num_ifriit_beams

    deck_gen_params['pointings'] = np.zeros((num_examples, num_ifriit_beams, 3))
    deck_gen_params["theta_pointings"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["phi_pointings"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["defocus"] = np.zeros((num_examples, num_ifriit_beams))
    deck_gen_params["p0"] = np.zeros((num_examples, num_ifriit_beams, dataset_params["num_powers_per_cone"]))
    deck_gen_params["sim_params"] = np.zeros((num_examples, dataset_params["num_input_params"]*2))

    return deck_gen_params



def import_nif_config(sys_params):
    facility_spec = dict()

    facility_spec['nbeams'] = 192
    facility_spec['target_radius'] = [0.0] #place holder
    facility_spec['facility'] = "NIF"
    facility_spec['num_quads'] = 48
    facility_spec['num_cones'] = 8
    facility_spec['default_power'] = 1.0 #TW per beam

    facility_spec['cone_names'] = np.array((23.5, 30, 44.5, 50, 23.5, 30, 44.5, 50))
    # The order of these is important (top-to-equator, then bottom-to-equator)
    facility_spec['quad_from_each_cone'] = np.array(('Q15T', 'Q13T', 'Q14T', 'Q11T', 'Q15B', 'Q16B', 'Q14B', 'Q13B'), dtype='<U4')
    facility_spec["beams_per_ifriit_beam"] = 1 # fuse quads?

    filename1 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/NIF_UpperBeams.txt"
    filename2 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/NIF_LowerBeams.txt"
    facility_spec = config_read_csv(facility_spec, filename1, filename2)
    facility_spec = config_formatting(facility_spec)

    return facility_spec



def import_lmj_config(sys_params, quad_split_bool):
    facility_spec = dict()

    facility_spec['nbeams'] = 80
    facility_spec['target_radius'] = [0.0] #place holder
    facility_spec['facility'] = "LMJ"
    facility_spec['num_quads'] = 20
    facility_spec['num_cones'] = 4
    facility_spec['default_power'] = 1.0 # 0.63 #TW per beam

    # The order of these is important (top-to-equator, then bottom-to-equator)
    facility_spec['quad_from_each_cone'] = np.array(('28U', '10U', '10L', '28L'), dtype='<U4')
    facility_spec["beams_per_ifriit_beam"] = 4 # fuse quads?

    filename1 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/LMJ_UpperBeams.txt"
    filename2 = sys_params["root_dir"] + "/" + sys_params["facility_config_files_dir"] + "/LMJ_LowerBeams.txt"
    facility_spec = config_read_csv(facility_spec, filename1, filename2)

    if quad_split_bool:
        facility_spec["beams_per_ifriit_beam"] = 1
        list_beam_keys = ["Beam", "Quad", "Cone", "Theta", "Phi", "PR"]
        for key in list_beam_keys:
            if key == "Theta":
                facility_spec[key] = np.array(facility_spec[key])
                facility_spec[key] = np.concatenate((facility_spec[key]+2.26, facility_spec[key]+2.26, facility_spec[key]-2.26, facility_spec[key]-2.26))
            elif key == "Phi":
                facility_spec[key] = np.array(facility_spec[key])
                facility_spec[key] = np.concatenate((facility_spec[key]-1.769, facility_spec[key]+1.769, facility_spec[key]+1.769, facility_spec[key]-1.769))
            else:
                facility_spec[key] = np.concatenate((facility_spec[key], facility_spec[key], facility_spec[key], facility_spec[key]))

    facility_spec = config_formatting(facility_spec)

    return facility_spec



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


def config_formatting(facility_spec):
    facility_spec["PR"] = np.array(facility_spec["PR"], dtype='i')
    facility_spec["Beam"] = np.array(facility_spec["Beam"], dtype='<U4')
    facility_spec["Quad"] = np.array(facility_spec["Quad"], dtype='<U4')
    facility_spec["Cone"] = np.array(facility_spec["Cone"])
    facility_spec["Theta"] = np.radians(facility_spec["Theta"])
    facility_spec["Phi"] = np.radians(facility_spec["Phi"])

    facility_spec['beams_per_cone'] = [0] * facility_spec['num_cones']
    for icone in range(facility_spec['num_cones']):
        quad_name = facility_spec['quad_from_each_cone'][icone]
        quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
        quad_start_ind = quad_slice[0]

        cone_name = facility_spec['Cone'][quad_start_ind]
        facility_spec['beams_per_cone'][icone] = int(np.count_nonzero(facility_spec["Cone"] == cone_name) / 2 * facility_spec["beams_per_ifriit_beam"])
    facility_spec['beams_per_cone'] = np.array(facility_spec['beams_per_cone'], dtype='int8')

    return facility_spec



def generate_run_files(dataset, dataset_params, facility_spec, sys_params, deck_gen_params):

    for iconfig in range(dataset["num_evaluated"], dataset_params["num_examples"]):
      if (iconfig==dataset["num_evaluated"]):
        config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
        file_exists = os.path.exists(config_location)
        if not file_exists:
            os.makedirs(config_location)

        if dataset_params["plasma_profile_source"] == "multi":
            ind_interface_dt_ch = [0]
            print("!!! Hardcoded hydro evaluation time DT-CH interface at cell:" + str(ind_interface_dt_ch)+" !!!")
            path = sys_params["data_dir"] + "/" + sys_params["multi_dir"]
            multi_data = um.multi_read_ascii(path+"/"+sys_params["multi_output_ascii_filename"])
            multi_data = um.read_inputs(path+"/"+sys_params["multi_input_filename"], multi_data)

        for tind in range(dataset_params["num_profiles_per_config"]):
            run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
            isExist = os.path.exists(run_location)

            if not isExist:
                os.makedirs(run_location)

            loc_ifriit_runfiles = sys_params["root_dir"] + "/" + sys_params["ifriit_run_files_dir"]
            shutil.copyfile(loc_ifriit_runfiles + "/" + sys_params["ifriit_binary_filename"],
                            run_location + "/" + sys_params["ifriit_binary_filename"])
            if dataset_params["run_plasma_profile"]:
                if dataset_params["plasma_profile_source"] == "default":
                    shutil.copyfile(sys_params["root_dir"] + "/" +
                                sys_params["plasma_profile_dir"] + "/" +
                                sys_params["plasma_profile_nc"],
                                run_location + "/" + sys_params["plasma_profile_nc"])
                elif dataset_params["plasma_profile_source"] == "multi":
                    itime_multi = np.argmin(np.abs(multi_data["time"]*1.e9-dataset_params["plasma_profile_times"][tind]))
                    multi_nc, ncells, nmat = um.multi2ifriit_inputs(multi_data, itime_multi, ind_interface_dt_ch)
                    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
                    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
                    nrw.save_general_netcdf(multi_nc, run_location + "/" + sys_params["plasma_profile_nc"],
                                            extra_dimension={'x': ncells, 'z':1, 'nel':nmat})

                    multi_laser_pulse_per_beam(iconfig, tind, sys_params, facility_spec)
      else:
        config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
        file_exists = os.path.exists(config_location)
        if file_exists:
            shutil.rmtree(config_location)
        shutil.copytree(sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(dataset["num_evaluated"]),
                        config_location)
 
      for tind in range(dataset_params["num_profiles_per_config"]):
          generate_input_deck(iconfig, tind, dataset_params, facility_spec, sys_params)
          if dataset_params["time_varying_pulse"]:
              pwr_ind = tind
          else:
              pwr_ind = 0
          generate_input_pointing_and_pulses(iconfig, tind, pwr_ind, dataset_params, facility_spec, sys_params, deck_gen_params)



def multi_laser_pulse_per_beam(iconfig, tind, sys_params, facility_spec):
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

    new_pulse_time = old_pulse_time
    new_pulse_power = old_pulse_power / nbeams

    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
    f = open(run_location + "/" + sys_params["ifriit_pulse_name"], "w")
    #f.write(old_pulse_label + "\n")

    new_pulse_nlines = len(new_pulse_time)
    for iline in range(new_pulse_nlines):
        f.write("{:4.5f} {:4.5f} \n".format(new_pulse_time[iline],new_pulse_power[iline]))
    f.close()



def generate_input_deck(iconfig, tind, dataset_params, facility_spec, sys_params):
    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)

    loc_ifriit_runfiles = sys_params["root_dir"] + "/" + sys_params["ifriit_run_files_dir"]
    base_input_txt_loc = loc_ifriit_runfiles + "/" + sys_params["ifriit_input_name"]

    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
    with open(base_input_txt_loc) as old_file:
        with open(run_location+"/ifriit_inputs.txt", "w") as new_file:
            for line in old_file:
                if "NBEAMS" in line:
                    new_file.write("    NBEAMS                      = " + str(num_ifriit_beams) + ",\n")
                elif "DIAGNOSE_INPUT_BEAMS_RADIUS_UM" in line:
                    new_file.write("    DIAGNOSE_INPUT_BEAMS_RADIUS_UM = " + str(facility_spec['target_radius'][tind]) + "d0,\n")
                elif "CBET = .FALSE.," in line:
                    if dataset_params["run_with_cbet"]:
                        new_file.write("    CBET = .TRUE.,\n")
                    else:
                        new_file.write("    CBET = .FALSE.,\n")
                else:
                    new_file.write(line)



def generate_input_pointing_and_pulses(iconfig, tind, pwr_ind, dataset_params, facility_spec, sys_params, deck_gen_params):
    config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"] + str(iconfig)
    run_location = config_location + "/" + sys_params["sim_dir"] + str(tind)
    if (facility_spec['facility'] == "NIF"):
        j = 0
        with open(run_location+'/ifriit_inputs.txt','a') as f:
            for beam in facility_spec['Beam']:
                cone_name = facility_spec["Cone"][np.where(facility_spec["Beam"] == beam)[0][0]]
                if (cone_name == 23.5):
                    cpp="inner-23"
                elif (cone_name == 30):
                    cpp="inner-30"
                elif (cone_name == 44.5):
                    cpp="outer-44"                                       
                else:
                    cpp="outer-50"

                f.write('&BEAM\n')
                # if (cpp=="inner-23" or cpp=="inner-30"):
                #     f.write('    LAMBDA_NM           = '+str((1052.85+0.45)/3.)+',\n')   
                # else:
                f.write('    LAMBDA_NM           = {:.10f}d0,\n'.format(1052.85/3.))
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(deck_gen_params['pointings'][iconfig,j][0], deck_gen_params['pointings'][iconfig,j][1], deck_gen_params['pointings'][iconfig,j][2]))
                if dataset_params["plasma_profile_source"] == "multi":
                    f.write('    POWER_PROFILE_FILE_TW_NS = "'+sys_params["ifriit_pulse_name"]+'"\n')
                    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dataset_params["plasma_profile_times"][tind]))
                else:
                    f.write('    P0_TW               = {:.10f}d0,\n'.format(deck_gen_params['p0'][iconfig,j,pwr_ind]))
                if (dataset_params["run_type"] == "nif"):
                    f.write('    PREDEF_FACILITY     = "NIF"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')
                    f.write('    PREDEF_CPP          = "NIF-'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    #f.write('    CPP_ROTATION_DEG    = 45.0d0,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(deck_gen_params['defocus'][iconfig,j]))
                elif (dataset_params["run_type"] == "test"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(deck_gen_params['port_centre_theta'][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(deck_gen_params['port_centre_phi'][j])))
                    f.write('    FOCAL_M             = 10.0d0,\n')
                    f.write('    SG                  = 6,\n')
                    f.write('    LAW                  = 2,\n')
                    f.write('    RAD_1_UM            = 80.0d0,\n')
                    f.write('    RAD_2_UM            = 80.0d0,\n')
                if 'fuse' in deck_gen_params.keys() and deck_gen_params['fuse'][j]:
                    f.write('    FUSE_QUADS          = .TRUE.,\n')
                    f.write('    FUSE_BY_POINTINGS   = .TRUE.,\n')
                else:
                    f.write('    FUSE_QUADS          = .FALSE.,\n')
                if 'xy-mispoint' in deck_gen_params.keys():
                    f.write('    XY_MISPOINT_UM      = {:.10f}d0,{:.10f}d0,\n'.format(deck_gen_params['xy-mispoint'][iconfig,j][0],deck_gen_params['xy-mispoint'][iconfig,j][1]))
                f.write('/\n')
                f.write('\n')
                j = j + 1
            f.write('\n')
            f.write('! Last line must not be empty')

    elif (facility_spec['facility'] == "LMJ"):
        j = 0
        with open(run_location+'/ifriit_inputs.txt','a') as f:
            for beam in facility_spec['Quad']:
                cpp="LMJ-A"

                f.write('&BEAM\n')
                f.write('    LAMBDA_NM           = {:.10f}d0,\n'.format(1052.85/3.))
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(deck_gen_params['pointings'][iconfig,j][0],deck_gen_params['pointings'][iconfig,j][1],deck_gen_params['pointings'][iconfig,j][2]))
                if dataset_params["plasma_profile_source"] == "multi":
                    f.write('    POWER_PROFILE_FILE_TW_NS = "'+sys_params["ifriit_pulse_name"]+'"\n')
                    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dataset_params["plasma_profile_times"][tind]))
                else:
                    f.write('    P0_TW               = {:.10f}d0,\n'.format(deck_gen_params['p0'][iconfig,j,pwr_ind]))
                if (dataset_params["run_type"] == "lmj"):
                    f.write('    PREDEF_FACILITY     = "'+facility_spec['facility']+'"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')
                    f.write('    PREDEF_CPP          = "'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(deck_gen_params['defocus'][iconfig,j]))
                elif (dataset_params["run_type"] == "test"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(deck_gen_params['port_centre_theta'][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(deck_gen_params['port_centre_phi'][j])))
                    f.write('    FOCAL_M             = 10.0d0,\n')
                    f.write('    SG                  = 6,\n')
                    f.write('    LAW                  = 2,\n')
                    f.write('    RAD_1_UM            = 80.0d0,\n')
                    f.write('    RAD_2_UM            = 80.0d0,\n')
                ##
                f.write('/\n')
                f.write('\n')
                j = j + 1
            f.write('\n')
            f.write('! Last line must not be empty')

    else:
        print('Unknown facility',facility_spec['facility'])

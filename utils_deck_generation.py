import os
import shutil
import numpy as np
import csv
import healpy_pointings as hpoint


def create_run_files(dataset_params, sys_params, run_data):

    num_output = dataset_params["num_output"]
    num_examples = dataset_params["num_examples"]

    coord_o = np.zeros(3)
    coord_o[2] = run_data['target_radius']

    # The order of these is important (top-to-equator, then bottom-to-equator)
    if (run_data['facility'] == "NIF"):
        run_data['quad_from_each_cone'] = ('Q15T', 'Q13T', 'Q14T', 'Q11T', 'Q15B', 'Q16B', 'Q14B', 'Q13B')
        num_ifriit_beams = run_data['nbeams']
        beams_per_ifriit_beam = 1
    elif (run_data['facility'] == "LMJ"):
        run_data['quad_from_each_cone'] = ('28U', '10U', '10L', '28L')
        num_ifriit_beams = run_data['num_quads']
        beams_per_ifriit_beam = 4

    run_data['pointings'] = np.zeros((num_ifriit_beams, 3))
    run_data["Port_centre_theta"] = np.zeros(num_ifriit_beams)
    run_data["Port_centre_phi"] = np.zeros(num_ifriit_beams)
    theta_pointings = np.zeros((num_ifriit_beams, num_examples))
    phi_pointings = np.zeros((num_ifriit_beams, num_examples))
    run_data["defocus"] = np.zeros(num_ifriit_beams)
    run_data["p0"] = np.zeros(num_ifriit_beams)
    run_data["fuse"] = [False] * num_ifriit_beams

    run_data['beams_per_cone'] = [0] * run_data['num_cones']

    sim_params = np.zeros((num_output*2, num_examples))

    for iex in range(num_examples):
        if num_examples>1:
            ex_params =  dataset_params["Y_train"][:,iex]
        else:
            ex_params =  dataset_params["Y_train"]
        for icone in range(run_data['num_cones']):
            il = (icone*dataset_params["num_sim_params"]) % num_output
            iu = ((icone+1)*dataset_params["num_sim_params"]-1) % num_output + 1
            cone_params = ex_params[il:iu]

            x = cone_params[dataset_params["theta_index"]] * 2.0 - 1.0
            y = cone_params[dataset_params["phi_index"]] * 2.0 - 1.0
            r, offset_phi = hpoint.square2disk(x, y)

            if icone > int(run_data['num_cones']/2.0-1):
                if dataset_params["hemisphere_symmetric"]:
                    offset_phi = np.pi - offset_phi # Symmetric
                else:
                    offset_phi = (offset_phi + np.pi) % (2.0 * np.pi) # anti-symmetric
            offset_theta = r * dataset_params["surface_cover_radians"]
            sim_params[icone*dataset_params["num_sim_params"]+dataset_params["theta_index"],iex] = offset_theta
            sim_params[icone*dataset_params["num_sim_params"]+dataset_params["phi_index"],iex] = offset_phi

            if dataset_params["defocus_bool"]:
                cone_defocus = cone_params[dataset_params["defocus_index"]] * dataset_params["defocus_range"]
                sim_params[icone*dataset_params["num_sim_params"]+dataset_params["defocus_index"],iex] = cone_defocus
            else:
                cone_defocus = dataset_params["defocus_default"]

            cone_power = (cone_params[dataset_params["power_index"]] * (1.0 - dataset_params["min_power"])
                          + dataset_params["min_power"])
            sim_params[icone*dataset_params["num_sim_params"]+dataset_params["power_index"],iex] = cone_power

            quad_name = run_data['quad_from_each_cone'][icone]
            quad_start_ind = run_data["Quad"].index(quad_name)
            quad_slice = slice(quad_start_ind, quad_start_ind+run_data['beams_per_quad'])

            ind_base = run_data["Quad"].index(quad_name)
            cone_name = run_data['Cone'][ind_base]
            run_data['beams_per_cone'][icone] = int(run_data["Cone"].count(cone_name)/2) * beams_per_ifriit_beam

            cone_slice = (np.where(np.array(run_data['Cone']) == cone_name)[0])
            quad_list_in_cone = np.array(run_data["Quad"])[cone_slice]

            for quad_name in quad_list_in_cone:
                ind = run_data["Quad"].index(quad_name)
                # remove beams in symmetric cone
                if np.abs(run_data["Theta"][ind] - run_data["Theta"][ind_base]) < np.radians(5.0):
                    quad_slice = slice(ind,ind+run_data['beams_per_quad'])
                    beam_names = run_data['Beam'][quad_slice]

                    run_data["Port_centre_theta"][quad_slice] = np.mean(run_data["Theta"][quad_slice])
                    run_data["Port_centre_phi"][quad_slice] = np.mean(run_data["Phi"][quad_slice])
                    port_theta = run_data["Port_centre_theta"][ind]
                    port_phi = run_data["Port_centre_phi"][ind]

                    rotation_matrix = np.matmul(np.matmul(hpoint.rot_mat(port_phi, "z"),
                                                          hpoint.rot_mat(port_theta, "y")),
                                      np.matmul(hpoint.rot_mat(offset_phi, "z"),
                                                hpoint.rot_mat(offset_theta, "y")))

                    coord_n = np.matmul(rotation_matrix, coord_o)

                    theta_pointings[quad_slice,iex] = np.arccos(coord_n[2] / run_data['target_radius'])
                    phi_pointings[quad_slice,iex] = np.arctan2(coord_n[1], coord_n[0])

                    run_data['pointings'][ind:ind+run_data['beams_per_quad'],:] = np.array(coord_n)
                    run_data["defocus"][quad_slice] = cone_defocus
                    run_data["p0"][quad_slice] = run_data['default_power'] * cone_power  * beams_per_ifriit_beam

        if sys_params["run_gen_deck"]:
            run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"] + str(iex)
            generate_input_deck(run_data, run_location)
            generate_input_pointing_and_pulses(run_data, run_location, dataset_params["run_type"])
    dataset_params["sim_params"] = sim_params
    dataset_params["theta_pointings"] = theta_pointings
    dataset_params["phi_pointings"] = phi_pointings
    return dataset_params



def import_nif_config():
    run_data = dict()

    run_data['nbeams'] = 192
    run_data['target_radius'] = 1100.0
    run_data['facility'] = "NIF"
    run_data['num_quads'] = 48
    run_data['num_cones'] = 8
    run_data['beams_per_quad'] = int(run_data['nbeams'] / run_data['num_quads'])
    run_data['default_power'] = 1.0 #TW per beam
    filename = "NIF_UpperBeams.txt"

    j = -1
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==-1:
            key = row
            for i in range(len(row)):
                run_data[row[i]] = [None] * int(run_data['nbeams'])
        else:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j] = row[i]
                elif i < 5:
                    run_data[key[i]][j] = float(row[i])
                else:
                    run_data[key[i]][j] = int(row[i])
        j=j+1
    f.close()
    filename = "NIF_LowerBeams.txt"
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==int(run_data['nbeams']/2.0):
            key = row
        else:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j-1] = row[i]
                elif i < 5:
                    run_data[key[i]][j-1] = float(row[i])
                else:
                    run_data[key[i]][j-1] = int(row[i])
        j=j+1
    f.close()

    run_data["Theta"] = np.radians(run_data["Theta"])
    run_data["Phi"] = np.radians(run_data["Phi"])

    return run_data



def import_lmj_config():
    run_data = dict()

    run_data['nbeams'] = 80 # no quad splitting so this is quads not beams
    run_data['target_radius'] = 1000.0
    run_data['facility'] = "LMJ"
    run_data['num_quads'] = 20
    run_data['num_cones'] = 4
    run_data['beams_per_quad'] = 1
    run_data['default_power'] = 0.63 #TW per beam

    filename = "LMJ_UpperBeams.txt"
    j = -1
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==-1:
            key = row
            for i in range(len(row)):
                run_data[row[i]] = [None] * int(run_data['num_quads'])
        else:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j] = row[i]
                elif i < 5:
                    run_data[key[i]][j] = float(row[i])
                else:
                    run_data[key[i]][j] = int(row[i])
        j=j+1
    f.close()
    filename = "LMJ_LowerBeams.txt"
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==int(run_data['num_quads']/2.0):
            key = row
        else:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j-1] = row[i]
                elif i < 5:
                    run_data[key[i]][j-1] = float(row[i])
                else:
                    run_data[key[i]][j-1] = int(row[i])
        j=j+1
    f.close()

    run_data["Theta"] = np.radians(run_data["Theta"])
    run_data["Phi"] = np.radians(run_data["Phi"])

    return run_data



def copy_ifriit_exc(run_location, iex):
    run_location = run_location + str(iex)
    shutil.copyfile("main", run_location+"/main")



def generate_input_deck(the_data, run_location):

    isExist = os.path.exists(run_location)

    if not isExist:
        os.makedirs(run_location)

    if (the_data['facility'] == "NIF"):
        num_ifriit_beams = the_data['nbeams']
    if (the_data['facility'] == "LMJ"):
        num_ifriit_beams = the_data['num_quads']

    longStr = ("ifriit_inputs_base.txt")
    with open(longStr) as old_file:
        with open(run_location+"/ifriit_inputs.txt", "w") as new_file:
            for line in old_file:
                if "NBEAMS" in line:
                    new_file.write("    NBEAMS                      = " + str(num_ifriit_beams) + ",\n")
                elif "DIAGNOSE_INPUT_BEAMS_RADIUS_UM" in line:
                    new_file.write("    DIAGNOSE_INPUT_BEAMS_RADIUS_UM = " + str(the_data['target_radius']) + "d0,\n")
                else:
                    new_file.write(line)



def generate_input_pointing_and_pulses(dat, run_location, run_type):
    if (dat['facility'] == "NIF"):
        j = 0
        with open(run_location+'/ifriit_inputs.txt','a') as f:
            for beam in dat['Beam']:
                cone_name = dat["Cone"][dat["Beam"].index(beam)]
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
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(dat['pointings'][j][0],dat['pointings'][j][1],dat['pointings'][j][2]))
                if 't0' in dat.keys():
                    f.write('    POWER_PROFILE_FILE_TW_NS = "pulse_'+beam+'.txt"\n')
                    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dat['t0']))
                else:
                    f.write('    P0_TW               = {:.10f}d0,\n'.format(dat['p0'][j]))
                if (run_type == "nif"):
                    f.write('    PREDEF_FACILITY     = "NIF"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')
                    f.write('    PREDEF_CPP          = "NIF-'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    #f.write('    CPP_ROTATION_DEG    = 45.0d0,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(dat['defocus'][j]))
                elif (run_type == "test"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_theta'][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_phi'][j])))
                    f.write('    FOCAL_M             = 10.0d0,\n')
                    f.write('    SG                  = 6,\n')
                    f.write('    LAW                  = 2,\n')
                    f.write('    RAD_1_UM            = 80.0d0,\n')
                    f.write('    RAD_2_UM            = 80.0d0,\n')
                if 'fuse' in dat.keys() and not dat['fuse'][j]:
                    f.write('    FUSE_QUADS          = .FALSE.,\n')
                else:
                    f.write('    FUSE_QUADS          = .TRUE.,\n')
                    f.write('    FUSE_BY_POINTINGS   = .TRUE.,\n')
                if 'xy-mispoint' in dat.keys():
                    f.write('    XY_MISPOINT_UM      = {:.10f}d0,{:.10f}d0,\n'.format(dat['xy-mispoint'][j][0],dat['xy-mispoint'][j][1]))
                f.write('/\n')
                f.write('\n')
                j = j + 1
            f.write('\n')
            f.write('! Last line must not be empty')

    elif (dat['facility'] == "LMJ"):
        j = 0
        with open(run_location+'/ifriit_inputs.txt','a') as f:
            for beam in dat['Quad']:
                cpp="LMJ-A"

                f.write('&BEAM\n')
                f.write('    LAMBDA_NM           = {:.10f}d0,\n'.format(1052.85/3.))
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(dat['pointings'][j][0],dat['pointings'][j][1],dat['pointings'][j][2]))
                if 't0' in dat.keys():
                    f.write('    POWER_PROFILE_FILE_TW_NS = "pulse_'+beam+'.txt"\n')
                    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dat['t0']))
                else:
                    f.write('    P0_TW               = {:.10f}d0,\n'.format(dat['p0'][j]))
                if (run_type == "lmj"):
                    f.write('    PREDEF_FACILITY     = "'+dat['facility']+'"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')
                    f.write('    PREDEF_CPP          = "'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(dat['defocus'][j]))
                elif (run_type == "test"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_theta'][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_phi'][j])))
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
        print('Unknown facility',dat['facility'])

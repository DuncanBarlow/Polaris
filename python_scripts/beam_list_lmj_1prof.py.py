import numpy as np
import training_data_generation as tdg
import matplotlib.pyplot as plt
import healpy as hp
import utils_intensity_map as uim
import utils_deck_generation as idg
import netcdf_read_write as nrw
import training_data_generation as tdg
import utils_healpy as uhp
import os
np_complex = np.vectorize(complex)

# initialize data
#diag_dir = "../Data/2027_Implosix/A635_T35_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128"
#diag_dir = "../Data/2027_Implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128"
#diag_dir = "../Data/2026_opportunite/A850_T15_270kJ_4p5TW_pt30deg_def0mm_qsno5_minP0p9_128"
#diag_dir = "../Data/Data_debug"
#diag_dir = "../Data/2026_Opportunite/A850_T15_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p9_128"
diag_dir = "../Data/2027_Implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128"
diag_dir = "../Data/2027_Implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128_160beams"
diag_dir = "../Data/2027_Implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qsno5_minP0p7_128_160beams"

# a conserver
#diag_dir = "../Data/2027_implosix/A635_T35_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128_grad100" # 6 %
#diag_dir = "../Data/2027_implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128_grad100" # 10 %
#diag_dir = "../Data/2026_opportunite/A850_T15_270kJ_4p5TW_pt30deg_def0mm_qsno5_minP0p9_128_grad100" # 23 %
#diag_dir = "../Data/2026_opportunite/A850_T15_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p9_128_grad100" #10 %
#diag_dir = "../Data/2026_opportunite/A850_T15_150kJ_2p5TW_pt20deg_def0mm_qson5_minP0p9_128_grad100" #10 %

#diag_dir = "../Data/2027_Implosix/A635_T35_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128_grad100" # 6% toutes methodes
#diag_dir = "../Data/2027_Implosix/A850_T20_270kJ_4p5TW_pt20deg_def0mm_qson5_minP0p7_128_gene100" # 8%, 10% autres methodes

#diag_dir = '/ccc/scratch/cont002/dam/martbert/Polaris/Data/debug_subcone_on'
#diag_dir = '/ccc/scratch/cont002/dam/martbert/Polaris/Data/debug_subcone_no'
diag_dir = '/ccc/scratch/cont002/dam/martbert/Polaris/Data/lmj_list_quads'
#diag_dir = '/ccc/scratch/cont002/dam/martbert/Polaris/Data/lmj_list_beams'

with_pointing_markers = True
import_flipped = False
old_format = False
display_steradians = False
sys_params = tdg.define_system_params(diag_dir)

# read the results
ind_profile = 0
sys_params["root_dir"] = diag_dir
dataset_params = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["dataset_params_filename"])
facility_spec = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["facility_spec_filename"])
if dataset_params["bool_group_beams_by_cone"]:
    dataset_params['quad_from_each_group'] = np.array(('Q28H', 'Q10H'), dtype='<U4')
    facility_spec, dataset_params = idg.group_beams_by_cone(facility_spec, dataset_params)
else:
    facility_spec, dataset_params = idg.group_beams_subcones_lmj(facility_spec, dataset_params)
dataset = nrw.read_general_netcdf(sys_params["root_dir"]+"/"+sys_params["trainingdata_filename"])

# Find the smallest rms
N_ex = len(dataset["rms"][:, ind_profile])
rms = np.zeros(N_ex)
for i_ex in range(N_ex):
	rms[i_ex] = 100 * dataset["rms"][i_ex, ind_profile]

index = np.argmin(rms[rms!=0.])
#index = np.argmin(rms)
mini = rms[index]
print("Best config is {:d}".format(index))

#-------------------------------------------------------------------
# plot the illumination
#-------------------------------------------------------------------

iex = index

avg_flux = dataset["avg_flux"][iex, ind_profile]
real_modes = dataset["real_modes"][iex,ind_profile,:]
imag_modes = dataset["imag_modes"][iex,ind_profile,:]
rms = dataset["rms"][iex, ind_profile]

intensity_map_normalized = uhp.modes2imap(real_modes, imag_modes, dataset_params["imap_nside"])
intensity_map_sr = (intensity_map_normalized+1)*avg_flux

if ind_profile == 0:
    if display_steradians:
        drive_map = intensity_map_sr
        drive_units = r"$\rm{W/sr}$"
    else:
        drive_map = (intensity_map_normalized+1)*avg_flux / (dataset_params['target_radius'] / 10000.0)**2
        drive_units = r"$\rm{W/cm^2}$"
else:
    drive_map = intensity_map_sr
    drive_units = r"$\rm{Mbar}$"

hp.mollview(drive_map, unit=drive_units,flip="geo")
hp.graticule()
fig_name = "/" + diag_dir.split('/')[1] + '_beams.png'
plt.savefig(diag_dir + fig_name, dpi=150, bbox_inches='tight')

complex_modes = np_complex(real_modes, imag_modes)
power_spectrum = uhp.alms2power_spectrum(complex_modes, dataset_params["LMAX"])

LMAX = dataset_params["LMAX"]
fig = plt.figure()
ax = plt.axes()
plt.plot(np.arange(LMAX), np.sqrt(power_spectrum) * 100.0)
ax.set_xticks(range(0, LMAX+1, int(LMAX/5)))
plt.xlim([0, LMAX])
plt.title("Intensity Modes")
plt.xlabel("l mode")
plt.ylabel(r"amplitude ($\%$)")


#-------------------------------------------------------------------
# plot illumination with markers for PDD
#-------------------------------------------------------------------

if with_pointing_markers:
    hp.mollview(drive_map, unit=drive_units,flip="geo")
    hp.graticule()
    
    deck_gen_params = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["deck_gen_params_filename"])

    print_list1, power_deposited = uim.readout_intensity(facility_spec, intensity_map_sr, dataset_params, ind_profile)
    print_list2 = uim.extract_run_parameters(iex, ind_profile, power_deposited, dataset_params, facility_spec, sys_params, deck_gen_params)
    print_list = print_list1 + print_list2
    stats_filename = "../" + sys_params["figure_location"]+"/stats"+str(iex) + "_" + str(ind_profile)+".txt"
    uim.print_save_readout(print_list, stats_filename)
    
    port_theta = deck_gen_params["port_centre_theta"]
    port_phi = deck_gen_params["port_centre_phi"]
    
    pointing_theta = np.squeeze(deck_gen_params["theta_pointings"][iex,:])
    pointing_phi = np.squeeze(deck_gen_params["phi_pointings"][iex,:])

    #---------------------------------------------------
    # We write the beam pointings in a txt file
    #---------------------------------------------------

    # get cone_powers
    cone_powers = np.zeros(dataset_params['num_beam_groups'])
    for icone in range(dataset_params['num_beam_groups']):

        quad_name = dataset_params['quad_from_each_group'][icone]
        quad_slice = np.where(facility_spec["Quad"] == quad_name)[0]
        quad_start_ind = quad_slice[0]
        cone_powers[icone] = deck_gen_params["p0"][iex,quad_start_ind,0] / (
                      dataset_params['default_power'] * facility_spec["beams_per_ifriit_beam"])

    if not(dataset_params["bool_group_beams_by_cone"]) :

        # get cone_powers_sorted
        cone_power_sorted = np.zeros(dataset_params['num_beam_groups'])
        for i in range(dataset_params['num_beam_groups']):
            quad_name = dataset_params['quad_from_each_group'][i]
    #        print(quad_name, dataset_params['sub_cone_list'])
            for j in range(dataset_params['num_beam_groups']):
                condition = str(quad_name[1:3]) in  dataset_params['sub_cone_list'][j]
    #            print(str(quad_name[1:3]), dataset_params['sub_cone_list'][j], condition)
                if condition :
                    cone_power_sorted[j] = cone_powers[i]
    #                print('quad' + quad_name + ' belongs to subcone = ', j, ' with power ' , cone_powers[i])

        cone_powers = cone_power_sorted.copy()

    text = ''
    text += "{:^12} {:^16} {:^16} {:^16} {:^16}\n".format("Beam", "x (mm)", "y (mm)", "z (mm)", "Pbal (0-1)")
    L = len(pointing_theta)
    rad = dataset_params['target_radius'] / 1000.
    fmt = "{:^12} {:^16.4e} {:^16.4e} {:^16.4e} {:^16.4e}\n"
    for i in range(L):

        # get power_balance

        if not(dataset_params["bool_group_beams_by_cone"]) :
            power_balance = -1.
            quad_name = facility_spec["Beam"][i][1:3]
            for j in range(facility_spec['num_cones']):
                condition = str(quad_name) in  dataset_params['sub_cone_list'][j]
                if condition :
                    #power_balance = cone_power_sorted[j]
                    power_balance = cone_powers[j]
        else :
            power_balance = -1.
            cone_name = facility_spec['Cone'][i]

            if (cone_name == 33) or (cone_name == 147) :
                power_balance = cone_powers[0]
            else :
                power_balance = cone_powers[1]

        text += fmt.format(
                              facility_spec["Beam"][i]
                            , rad * np.sin(pointing_theta[i]) * np.cos(pointing_phi[i])
                            , rad * np.sin(pointing_theta[i]) * np.sin(pointing_phi[i])
                            , rad * np.cos(pointing_theta[i])
                            , power_balance
                            )

    filename = diag_dir + "/" + diag_dir.split('/')[1] + "_beams.txt"
    with open(filename, "w") as f:
        f.write(text)



    num_ifriit_beams = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
    port_cartesian = np.zeros((num_ifriit_beams, 2))
    portc_cartesian = np.zeros((num_ifriit_beams, 2))
    pointing_cartesian = np.zeros((num_ifriit_beams, 2))

    for ibeam in range(0, num_ifriit_beams):
        port_cartesian[ibeam,0], port_cartesian[ibeam,1] = uim.angle2moll(facility_spec["Theta"][ibeam], facility_spec["Phi"][ibeam])
        portc_cartesian[ibeam,0], portc_cartesian[ibeam,1] = uim.angle2moll(deck_gen_params["port_centre_theta"][ibeam], deck_gen_params["port_centre_phi"][ibeam])
        pointing_cartesian[ibeam,0], pointing_cartesian[ibeam,1] = uim.angle2moll(pointing_theta[ibeam], pointing_phi[ibeam])
        dx = pointing_cartesian[ibeam,0] - portc_cartesian[ibeam,0]
        dy = pointing_cartesian[ibeam,1] - portc_cartesian[ibeam,1]
        # reduce line length to give space for arrow head
        head_length = 0.05
        shaft_angle = np.arctan2(dy, dx)
        shaft_length = np.sqrt(dx**2 + dy**2) - head_length
        if shaft_length > 0.0:
            dx = shaft_length * np.cos(shaft_angle)
            dy = shaft_length * np.sin(shaft_angle)
            
            if np.abs(dx) < 1.0: # don't plot arrow for pointings which cross the edge
                plt.arrow(portc_cartesian[ibeam,0], portc_cartesian[ibeam,1], dx, dy, linewidth=3, head_width=head_length, head_length=head_length, fc='k', ec='k')
        else:
            plt.arrow(portc_cartesian[ibeam,0], portc_cartesian[ibeam,1], dx, dy)

    plt.scatter(port_cartesian[:,0], port_cartesian[:,1],c="white", marker="s")
    plt.scatter(pointing_cartesian[:,0], pointing_cartesian[:,1],c="white", marker="x")

    fig_name = "/" + diag_dir.split('/')[1] + '_beams.png'
    plt.savefig(diag_dir + fig_name, dpi=150, bbox_inches='tight')

plt.show()



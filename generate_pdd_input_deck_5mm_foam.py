import numpy as np
import utils_deck_generation as idg
import training_data_generation as tdg
import utils_beam_pointing as ubp
import matplotlib.pyplot as plt

dirname_pulse = "laser_pulse_data_pdd/revolver_wetted_foam/"
filename = "laser_pulse_nif_pdd_wetted_foam.csv"

num_examples = 1
dataset_params, facility_spec = tdg.define_dataset_params(num_examples)
deck_gen_params = idg.define_deck_generation_params(dataset_params, facility_spec)

iex = 0
tind = 0
facility_spec['target_radius'] = 2343.0
facility_spec["fuse"]=[False]*facility_spec["nbeams"]
facility_spec["defocus"] = np.zeros(facility_spec["nbeams"]) + 35.0
deck_gen_params['defocus'][iex,:] = 35.0 
deck_gen_params['t0'] = 12.0
#facility_spec["default_power"] = 1.0

nbeams = facility_spec['nbeams']
beam_list = facility_spec["Beam"]
print(nbeams)

names,x,y,z = ubp.pointings_pdd(facility_spec)

deck_gen_params['p0'][iex,tind,:] = facility_spec["default_power"]

## pointing data
deck_gen_params['pointings'][iex,:,0] = x
deck_gen_params['pointings'][iex,:,1] = y
deck_gen_params['pointings'][iex,:,2] = z
deck_gen_params['port_centre_phi'] = facility_spec["Phi"]
deck_gen_params['port_centre_theta'] = facility_spec["Theta"] 

run_with_cbet = True 
base_input_txt_loc = ("ifriit_inputs_base.txt")
idg.generate_input_deck(dirname_pulse, base_input_txt_loc, run_with_cbet, facility_spec)

run_type = "nif"
idg.generate_input_pointing_and_pulses(iex, tind, facility_spec, deck_gen_params, dirname_pulse, run_type)

data = []
with open(dirname_pulse+filename) as input_file:
    for line in input_file:
        x = np.array(line.split())
        y = x.astype(float)
        data.append(y)

pulse_data = np.transpose(np.stack(data[:], axis=0))
times = pulse_data[0] * 1e9
total_power = pulse_data[1] / 1e12
print("times (ns)", times)
print("total power (TW)", total_power)
ntimes = len(times)
beam_powers = np.zeros((nbeams, len(total_power))) + total_power / nbeams

total_energy = np.trapz(total_power, x=times)
print("Total energy requested: {:.2f}".format(total_energy), "kJ")
num_cones = int(facility_spec["num_cones"]/2)
beam_powers = np.zeros((num_cones, ntimes))
cone0_multiplier = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cone1_multiplier = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cone2_multipliers = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cone3_multipliers = [1., 1., 1., 1.0, 1.0, 1.0]

beam_powers[0,:] = total_power / nbeams * cone0_multiplier
beam_powers[1,:] = total_power / nbeams * cone1_multiplier
beam_powers[2,:] = total_power / nbeams * cone2_multipliers
beam_powers[3,:] = total_power / nbeams * cone3_multipliers

unweighted_total_energy = 0.0
beams_per_cone = facility_spec["beams_per_cone"][:num_cones] * 2.0
for icone in range(num_cones):
    cone_power = beam_powers[icone,:] * beams_per_cone[icone]
    unweighted_total_energy += np.trapz(cone_power, x=times)
print("Total energy unweighted: {:.2f}".format(unweighted_total_energy), "kJ")
weighting = total_energy / unweighted_total_energy
beam_powers = beam_powers * weighting

if 't0' in deck_gen_params.keys():
    j=0
    for ibeam in range(nbeams):
      beam = facility_spec['Beam'][ibeam]
      if facility_spec['Cone'][ibeam] == 23.5:
          icone = 0
      if facility_spec['Cone'][ibeam] == 30.0:
          icone = 1
      if facility_spec['Cone'][ibeam] == 44.5:
          icone = 2
      if facility_spec['Cone'][ibeam] == 50.0:
          icone = 3
      with open(dirname_pulse+'pulse_'+beam+'.txt','w') as f:
          for i in range(len(total_power)):
              f.write(str(times[i]) + ' ' + str(beam_powers[icone,i]) + '\n')
      j = j + 1

cone_labels = ["23.5", "30.0", "44.5", "50.0"]
total_power_check = np.dot(beams_per_cone, beam_powers)
plt.figure()
plt.plot(times, total_power_check)
plt.plot(times, total_power)
plt.xlabel('Time (ns)')
plt.ylabel('Total Power (TW)')

plt.figure()
for icone in range(num_cones):
    plt.plot(times, beam_powers[icone,:], label="Cone "+cone_labels[icone])
plt.xlabel('Time (ns)')
plt.ylabel('Total Power (TW)')
plt.legend()
plt.show()

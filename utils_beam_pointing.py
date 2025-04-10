import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

def pointings_pdd(facility_spec):

  nbeams = facility_spec["nbeams"]
  target_radius = facility_spec['target_radius']
  nbeams_per_subcone = (    8,    8,    8,    8,   16,   16,   16,   16,
                           16,   16,   16,   16,    8,    8,    8,    8)
  theta_angles_subcones = ( 12.5, 30.0, 37.5, 41.5, 54.0, 65.5, 79.0, 85.5, 
             94.5, 101.0, 114.5, 126.0, 138.5, 142.5, 150.0, 167.5)

  sort_key = np.argsort(facility_spec["Theta"]*1000 + facility_spec["Phi"])
  theta = np.array(facility_spec["Theta"])[sort_key]
  phi = np.array(facility_spec["Phi"])[sort_key]
  names = np.array(facility_spec["Beam"])[sort_key]
  reverse_sort_key = np.argsort(sort_key)
  
  i=0
  for j in range(len(theta_angles_subcones)):
    theta[i:i+nbeams_per_subcone[j]] = theta_angles_subcones[j]/180.0*np.pi
    phi_angles_per_subcone = np.linspace(0.0, 2.0 * np.pi, num=nbeams_per_subcone[j]+1).transpose() 
    phi_gap = (2.0 * np.pi) / (nbeams_per_subcone[j])
    offset = 0.0#-phi_gap / 4.0
    #if (j%2==0):
    #  offset = phi_gap / 4.0
    #if (j<(len(theta_angles_subcones)/4)):
    #  offset = -offset
    #if (j>(len(theta_angles_subcones)/2-1) and j<(len(theta_angles_subcones)/4*3)):
    #  offset = -offset
    
    for ibeam in range(0, nbeams_per_subcone[j], 2):
      quad_centre = (phi[i+ibeam] + phi[i+ibeam+1]) / 2.0
      phi[i+ibeam] = quad_centre - phi_gap / 2.0 + offset
      phi[i+ibeam+1] = quad_centre + phi_gap / 2.0 + offset
      #phi[i+ibeam] = phi_angles_per_subcone[ibeam]
    i=i+nbeams_per_subcone[j]

  for i in range(facility_spec["nbeams"]):
    if phi[i] > (2.0 * np.pi):
      phi[i] = phi[i] - 2.0 * np.pi
    if phi[i] < 0.0:
      phi[i] = 2.0 * np.pi + phi[i]

  x = target_radius * np.cos(phi) * np.sin(theta)
  y = target_radius * np.sin(phi) * np.sin(theta)
  z = target_radius * np.cos(theta)
  
  tol = 1.0e-8
  x[np.abs(x) < tol] = 0
  y[np.abs(y) < tol] = 0
  z[np.abs(z) < tol] = 0
  
  fig2 = plt.figure()
  plt.plot(np.degrees(facility_spec["Theta"]), np.degrees(facility_spec["Phi"]),"s")
  plt.plot(np.degrees(theta), np.degrees(phi),"r*")
  plt.xlabel('$\\theta$')
  plt.ylabel('$\phi$')

  for i in range(facility_spec["nbeams"]):
    plt.text(np.degrees(facility_spec["Theta"][i]), np.degrees(facility_spec["Phi"][i]), facility_spec["Beam"][i],color="blue")
    plt.text(np.degrees(theta[i]),np.degrees(phi[i]),names[i],color="red",horizontalalignment='right')

  #plt.show()

  x = x[reverse_sort_key]
  y = y[reverse_sort_key]
  z = z[reverse_sort_key]
  names = names[reverse_sort_key]

  return names, x, y, z


"""
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x,y,z)
ax.scatter(x2,y2,z2, marker="*", c="r" )
ax.set_xlabel('X ($\mu m$)')
ax.set_ylabel('Y ($\mu m$)')
ax.set_zlabel('Z ($\mu m$)')

plt.show()
"""


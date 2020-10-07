from halotools_test.mock_observables import mean_radial_velocity_vs_r
from halotools_test.mock_observables import transverse_pvd_vs_r
import numpy as np

from halotools.sim_manager import FakeSim
halocat = FakeSim()
x = halocat.halo_table['halo_x']
y = halocat.halo_table['halo_y']
z = halocat.halo_table['halo_z']
sample1 = np.vstack((x,y,z)).T
vx = halocat.halo_table['halo_vx']
vy = halocat.halo_table['halo_vy']
vz = halocat.halo_table['halo_vz']
velocities = np.vstack((vx,vy,vz)).T
rbins = np.logspace(-1, 1, 10)
sv_12 = transverse_pvd_vs_r(sample1, velocities, rbins_absolute=rbins, period=halocat.Lbox)

print(sv_12)
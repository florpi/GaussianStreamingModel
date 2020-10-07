from halotools_test.mock_observables import tpcf, s_mu_tpcf
import numpy as np


from halotools.sim_manager import FakeSim
halocat = FakeSim()
x = halocat.halo_table['halo_x']
y = halocat.halo_table['halo_y']
z = halocat.halo_table['halo_z']
sample1 = np.vstack((x,y,z)).T
rbins = np.logspace(-1, 1, 10)
xi = tpcf(sample1, rbins, period=halocat.Lbox)

print(xi)
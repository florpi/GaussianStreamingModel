from halotools.mock_observables import mean_radial_velocity_vs_r
import numpy as np

def compute_mean_radial_velocity(r, pos, vel, boxsize, num_threads = 1):
    '''
	Computes the mean radial pairwise velocity as a function of r,
    using halotools.
	Args:
		r: np.array
			 binning in pair distances.
		pos: np.ndarray
			 3-D array with the position of the tracers.
		vel: np.ndarray
			3-D array with the velocities of the tracers.
		boxsize: float
			size of the simulation's box.
		num_threads: int
			number of threads to use.
	Returns:
		mean_radial_velocity: np.array
			1-D array with the mean radial pairwise velocity.
	'''

    mean_radial_velocity = mean_radial_velocity_vs_r(pos, vel, r, period = boxsize,
            num_threads = num_threads)

    return mean_radial_velocity

def compute_mean_transverse_velocity(r, pos, vel, boxsize, num_threads = 1):
    '''
	Computes the mean radial pairwise velocity as a function of r,
    using halotools.
	Args:
		r: np.array
			 binning in pair distances.
		pos: np.ndarray
			 3-D array with the position of the tracers.
		vel: np.ndarray
			3-D array with the velocities of the tracers.
		boxsize: float
			size of the simulation's box.
		num_threads: int
			number of threads to use.
	Returns:
		mean_radial_velocity: np.array
			1-D array with the mean radial pairwise velocity.
	'''

    mean_transverse_velocity = mean_transverse_velocity_vs_r(pos, vel, r, period = boxsize,
            num_threads = num_threads)

    return mean_transverse_velocity


def compute_std_radial_velocity(r, pos, vel, boxsize, num_threads = 1):
    '''
	Computes the standard deviation of the radial pairwise velocity
    as a function of r, using halotools.
	Args:
		r: np.array
			 binning in pair distances.
		pos: np.ndarray
			 3-D array with the position of the tracers.
		vel: np.ndarray
			3-D array with the velocities of the tracers.
		boxsize: float
			size of the simulation's box.
		num_threads: int
			number of threads to use.
	Returns:
		std_radial_velocity: np.array
			1-D array with the standard deviation of radial velocity.
	'''

    std_radial_velocity = radial_pvd_vs_r(pos, vel, r, period = boxsize,
            num_threads = num_threads)

    return std_radial_velocity

def compute_std_transverse_velocity(r, pos, vel, boxsize, num_threads = 1):
    '''
	Computes the standard deviation of the transverse pairwise velocity
    as a function of r, using halotools.
	Args:
		r: np.array
			 binning in pair distances.
		pos: np.ndarray
			 3-D array with the position of the tracers.
		vel: np.ndarray
			3-D array with the velocities of the tracers.
		boxsize: float
			size of the simulation's box.
		num_threads: int
			number of threads to use.
	Returns:
		std_transverse_velocity: np.array
			1-D array with the standard deviation of transverse velocity.
	'''

    std_transverse_velocity = transverse_pvd_vs_r(pos, vel, r, period = boxsize,
            num_threads = num_threads)

    return std_transverse_velocity
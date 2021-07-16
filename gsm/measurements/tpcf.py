from halotools.mock_observables import tpcf, s_mu_tpcf
from halotools.mock_observables import apply_zspace_distortion
import numpy as np


def compute_real_tpcf(r, pos, boxsize, num_threads=1, pos_cross=None):
    """
        Computes the real space two point correlation function using halotools
        Args:
                r: np.array
                         binning in pair distances.
                pos: np.ndarray
                         3-D array with the position of the tracers.
                boxsize: float
                        size of the simulation's box.
                num_threads: int
                        number of threads to use.
                pos_cross: np.ndarray
                         3-D array with the position of the tracers
                         (for cross_correlations).

        Returns:
                real_tpcf: np.array
                        1-D array with the real space tpcf.
        """
    if pos_cross is not None:
        do_auto = False
    else:
        do_auto = True
    real_tpcf = tpcf(
        pos,
        r,
        period=boxsize,
        num_threads=num_threads,
        sample2=pos_cross,
        do_auto=do_auto,
    )
    return real_tpcf


def move_to_redshift_space(pos, vel, cosmology, redshift, los_direction, boxsize):
    s_pos = pos.copy()
    z_pos = apply_zspace_distortion(
        true_pos=pos[:, los_direction],
        peculiar_velocity=vel[:, los_direction],
        redshift=redshift,
        cosmology=cosmology,
        Lbox=boxsize,
    )
    # Move tracers to redshift space
    s_pos[:, los_direction] = z_pos
    # Halotools tpcf_s_mu assumes the line of sight is always the z direction
    if los_direction != 2:
        s_pos_old = s_pos.copy()
        s_pos[:, 2] = s_pos_old[:, los_direction]
        s_pos[:, los_direction] = s_pos_old[:, 2]
    return s_pos


def compute_tpcf_s_mu(
    s,
    mu,
    pos,
    vel,
    los_direction,
    cosmology,
    boxsize,
    redshift,
    num_threads=1,
    pos_cross=None,
    vel_cross=None,
):
    """
        Computes the redshift space two point correlation function
        Args:
                s: np.array
                        binning in redshift space pair distances.
                mu: np.array
                         binning in the cosine of the angle respect to the line of sight.
                pos: np.ndarray
                        3-D array with the position of the tracers, in Mpc/h.
                vel: np.ndarray
                         3-D array with the velocities of the tracers, in km/s.
                los_direction: int
                        line of sight direction either 0(=x), 1(=y), 2(=z)
                cosmology: dict
                        dictionary containing the simulatoin's cosmological parameters.
                boxsize:  float
                        size of the simulation's box.
                num_threads: int 
                        number of threads to use.
        Returns:
                tpcf_s_mu: np.ndarray
                        2-D array with the redshift space tpcf.
        """
    if pos_cross is not None:
        do_auto = False
    else:
        do_auto = True

    s_pos = move_to_redshift_space(
        pos, vel, cosmology, redshift, los_direction, boxsize
    )
    if pos_cross is not None and vel_cross is not None:
        s_pos_cross = move_to_redshift_space(
            pos_cross, vel_cross, cosmology, redshift, los_direction, boxsize
        )
    else:
        s_pos_cross = None
    tpcf_s_mu = s_mu_tpcf(
        s_pos,
        s,
        mu,
        period=boxsize,
        estimator=u"Landy-Szalay",
        num_threads=num_threads,
        sample2=s_pos_cross,
        do_auto=do_auto,
    )
    return tpcf_s_mu

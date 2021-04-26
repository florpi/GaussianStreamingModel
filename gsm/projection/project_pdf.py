import numpy as np
from scipy.integrate import simps


def integrand(pdf_rt, v_r, v_los, cos_theta, sin_theta, r):
    v_t = (v_los - v_r * cos_theta) / sin_theta
    return 1 / sin_theta * pdf_rt(v_r=v_r, v_t=v_t, r=r)

def get_cos_theta(r, r_parallel):
    return r_parallel/r

def get_sin_theta(r, r_parallel):
    return np.sqrt(1 - get_cos_theta(r, r_parallel)**2)

def get_projected_pdf(
    pdf_rt,
    r_perpendicular,
    r_parallel,
    v_los,
    v_r_min=-100.0,
    v_r_max=100.0,
    n_v_r_bins=300,
):
    v_r = np.linspace(v_r_min, v_r_max, n_v_r_bins)
    los_pdf = np.zeros((len(r_perpendicular), len(r_parallel), len(v_los)))

    for i, r_perp in enumerate(r_perpendicular):
        for j, r_par in enumerate(r_parallel):
            r = np.sqrt(r_perp**2 + r_par**2)
            los_pdf[i, j, :] = simps(
                integrand(
                    pdf_rt,
                    v_r.reshape(1, -1),
                    v_los=v_los.reshape(-1, 1),
                    cos_theta=get_cos_theta(r, r_par),
                    sin_theta=get_sin_theta(r, r_par),
                    r=r,
                ),
                v_r,
                axis=-1,
            )
    return los_pdf

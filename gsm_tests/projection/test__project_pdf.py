from scipy.stats import multivariate_normal
import numpy as np
from scipy.integrate import simps
from gsm.projection.project_pdf import get_projected_pdf

def mean(r):
    return r**2

def gaussian_pdf(vels, r):
    rv = multivariate_normal(
        mean= [mean(r),0.],
        cov = [5,5]
    )
    return rv.pdf(vels)


def analytical_mean(r_parallel, r_perp):
    r = np.sqrt(r_perp.reshape(-1,1)**2 + r_parallel.reshape(1,-1)**2)
    return  r_parallel.reshape(1,-1)/r * mean(r)

def analytical_std(r_parallel, r_perp):
    return 5


def pdf_rt(r, v_r, v_t):
    if v_r.shape[0] == 1:
        v_r = np.broadcast_to(v_r, v_t.shape)
    elif v_t.shape[-1] == 1:
        v_t = np.broadcast_to(v_t, v_r.shape)
    vels = np.stack((v_r,v_t),axis=-1)
    return gaussian_pdf(vels, r=r)


def test__get_analytical_mean():
    r_parallel = np.linspace(1,5,10)
    r_perp = np.linspace(1.,5,10)
    v_los = np.linspace(-100,100,300)
    los_pdf = get_projected_pdf(
        pdf_rt,r_perp, r_parallel, v_los,
    ) 
    actual = simps(
        v_los*los_pdf, 
        v_los,
        axis=-1
    )
    desired = analytical_mean(r_perp,r_parallel)
    np.testing.assert_almost_equal(actual, desired, decimal=4)

def test__get_analytical_std():
    r_parallel = np.linspace(1,5,10)
    r_perp = np.linspace(1.,5,10)
    v_los = np.linspace(-100,100,300)
    los_pdf = get_projected_pdf(
        pdf_rt,r_perp, r_parallel, v_los,
    ) 
    actual_mean = simps(
        v_los*los_pdf, 
        v_los,
        axis=-1
    )
    actual = simps(
        v_los**2*los_pdf, 
        v_los,
        axis=-1
    ) - actual_mean**2
    desired = analytical_std(r_perp,r_parallel)

    np.testing.assert_almost_equal(actual, desired, decimal=2)

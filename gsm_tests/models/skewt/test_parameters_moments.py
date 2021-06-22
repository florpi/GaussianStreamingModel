import numpy as np
from scipy.integrate import simps
import pytest
from gsm.models.skewt import skewt
from gsm.models.skewt.moments2parameters import (
    moments2parameters,
    interpolate_moments2parameters,
)
from gsm.models.skewt.parameters2moments import parameters2moments


def test_moments2parameters():
    v = np.linspace(-100.0, 100.0, 100)
    w_true = 5.0
    v_c_true = 4.5
    alpha_true = -1.0
    nu_true = 10.2

    skewt_true = skewt.pdf(v, w_true, v_c_true, alpha_true, nu_true)

    true_mean = simps(skewt_true * v, v)
    true_std = np.sqrt(simps(skewt_true * (v - true_mean) ** 2, v))
    true_gamma1 = simps(skewt_true * (v - true_mean) ** 3, v) / true_std ** 3
    true_gamma2 = simps(skewt_true * (v - true_mean) ** 4, v) / true_std ** 4 - 3.0

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(-0.5, 5)
    )

    np.testing.assert_almost_equal(w_true, w, decimal=2)
    np.testing.assert_almost_equal(v_c_true, v_c, decimal=2)
    np.testing.assert_almost_equal(alpha_true, alpha, decimal=2)
    np.testing.assert_almost_equal(nu_true, nu, decimal=2)

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w_true, v_c_true, alpha_true, nu_true)

    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)


'''
def test_interpolate_moments():

    r_parallel = np.arange(0.5, 100, 1.)
    r_perp = np.arange(0.5, 50, 1.)

    mean = lambda r_perp,r_parallel: r_parallel.reshape(1,-1) * (r_perp + 10.).reshape(-1,1)
    std = lambda r_perp, r_parallel: r_parallel.reshape(1,-1) * (r_perp+10.).reshape(-1,1)
    gamma1 = lambda r_perp, r_parallel: r_parallel.reshape(1,-1) * (r_perp+20.).reshape(-1,1)
    gamma2 = lambda r_perp, r_parallel: r_parallel.reshape(1,-1) * (r_perp-20.).reshape(-1,1)

    w, v_c, alpha, nu = interpolate_moments2parameters(
        r_perp,
        r_parallel,
        mean(r_perp, r_parallel),
        std(r_perp, r_parallel),
        gamma1(r_perp, r_parallel),
        gamma2(r_perp, r_parallel),
    )
    r_perp_test = 10.6
    r_par_test = 10.6
    w_true, v_c_true, alpha_true, nu_true = moments2parameters(
        r_par_test*(r_perp_test + 10.),
        r_par_test*(r_perp_test+10.),
        r_par_test*(r_perp_test+20.),
        r_par_test*(r_perp_test-20.),
    )
    assert w(r_perp_test,r_par_test) == pytest.approx(w_true, rel=0.01)
    assert v_c(r_perp_test,r_par_test) == pytest.approx(v_c_true, rel=0.01)
    assert alpha(r_perp_test,r_par_test) == pytest.approx(alpha_true, rel=0.01)
    assert nu(r_perp_test,r_par_test) == pytest.approx(nu_true, rel=0.01)

'''

def test_small_scale_moments():
    true_mean = -6.
    true_std = 2.
    true_gamma1 = -1.
    true_gamma2 = 3.2

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(-0.7, 5)
    )

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w, v_c, alpha, nu)

    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)
    assert nu > 1


def test_large_scale_moments():
    true_mean = 0.
    true_std = 6
    true_gamma1 = 0.
    true_gamma2 = 0.2

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(-0.7, 5)
    )

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w, v_c, alpha, nu)

    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)
    assert nu > 1


def test_other_moments():
    true_mean = -3.
    true_std = 2.
    true_gamma1 = 0.52
    true_gamma2 = 2.27

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(-0.7, 5)
    )

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w, v_c, alpha, nu)

    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)

def test_not_working_moments():
    true_mean = 0.
    true_std = 5.
    true_gamma1 = -0.025
    true_gamma2 = 0.23

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(-0.7, 5.)
    )

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w, v_c, alpha, nu)

    print(gamma1_estimated, gamma2_estimated)
    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)
    assert nu > 1

def test_not_working_moments2():
    true_mean = -4.
    true_std = 2.7
    true_gamma1 = 1.78
    true_gamma2 = 2.7

    w, v_c, alpha, nu = moments2parameters(
        true_mean, true_std, true_gamma1, true_gamma2, p0=(1., 5.)
    )

    (
        mean_estimated,
        std_estimated,
        gamma1_estimated,
        gamma2_estimated,
    ) = parameters2moments(w, v_c, alpha, nu)

    print(gamma1_estimated, gamma2_estimated)
    np.testing.assert_almost_equal(mean_estimated, true_mean, decimal=2)
    np.testing.assert_almost_equal(std_estimated, true_std, decimal=2)
    np.testing.assert_almost_equal(gamma1_estimated, true_gamma1, decimal=2)
    np.testing.assert_almost_equal(gamma2_estimated, true_gamma2, decimal=2)
    assert nu > 1



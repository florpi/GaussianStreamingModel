from gsm.moments.perturbation_theory.linear import integrand_v_r, integrand_psi_r, integrand_psi_t, sigma_v_sq
from scipy import integrate
import numpy as np
import pytest


def analytical_v_r(r, k_max):

    return (r- (np.sin(r*k_max)/k_max))/r**2

@pytest.mark.parametrize("r,k_max", [(3, 10), (10, 100), (2, 30)])
def test_integrand_v_r(r, k_max):

    power = lambda k: 1/k 
    integral = integrate.quad(
            lambda x: integrand_v_r(x, r, power),
            0,
            k_max,
            )[0]

    expected = analytical_v_r(r, k_max)
    np.testing.assert_almost_equal( integral, expected, decimal=2)

def analytical_psi_t(r, k_max):
    return analytical_v_r(r, k_max)/r

@pytest.mark.parametrize("r,k_max", [(3, 10), (10, 100), (2, 30)])
def test_integrand_psi_t(r, k_max):

    power = lambda k: k
    integral = integrate.quad(
            lambda x: integrand_psi_t(x, r, power),
            0,
            k_max,
            )[0]

    expected = analytical_psi_t(r, k_max)
    np.testing.assert_almost_equal( integral, expected, decimal=2)

def analytical_psi_r(r, k_max):
    return (2*np.sin(r*k_max) - k_max*r*(np.cos(k_max*r)+1))/k_max/r**3

@pytest.mark.parametrize("r,k_max", [(3, 10), (10, 100), (2, 30)])
def test_integrand_psi_r(r, k_max):

    power = lambda k: k
    integral = integrate.quad(
            lambda x: integrand_psi_r(x, r, power),
            0,
            k_max,
            )[0]

    expected = analytical_psi_r(r, k_max)
    np.testing.assert_almost_equal( integral, expected, decimal=2)



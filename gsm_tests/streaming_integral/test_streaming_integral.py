import numpy as np
import pytest
from scipy.integrate import simps, quadrature, quad

from gsm.streaming_integral  import real2redshift
from gsm.models.gaussian import from_los as gaussian_from_los


@pytest.mark.parametrize("a,b", [(2, 3), (3, 2), (10, 1)])
def test__analytical(a, b):
    mean = lambda rparallel, rperp: 0
    scale = lambda rparallel, rperp: 1
    tpcf = lambda r: r**2
    gaussian_pdf = gaussian_from_los.losmoments2gaussian(mean, scale)
    s = np.sqrt(a**2+b**2)
    mu = a/np.sqrt(a**2 + b**2)
    integrand = real2redshift.integrand_s_mu(s, mu, tpcf, gaussian_pdf)
    r_integrand = np.linspace(-100,100,100)
    result = simps(integrand(r_integrand), r_integrand, axis=-1)
    analytical_result = 2 + a**2 + b**2
    assert result == pytest.approx(analytical_result, rel=0.05)


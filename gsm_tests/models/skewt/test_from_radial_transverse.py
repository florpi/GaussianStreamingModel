import numpy as np
import pytest
from gsm.models.skewt import from_radial_transverse

def test__from_radial_transverse():
    m_10 = lambda x: x
    c_20 = lambda x: x**2
    c_02 = lambda x: (x-5)**2
    c_30 = lambda x: x
    c_12 = lambda x: 1/x
    c_22 = lambda x: x
    c_40 = lambda x: x
    c_04 = lambda x: 1/x
 
    mean, std, gamma1, gamma2 = from_radial_transverse.project_moments(
            m_10, c_20, c_02, c_12, c_30, c_22, c_40, c_04
            )

    r_perp = 10.
    r_parallel = 5.
    r = np.sqrt(r_perp**2 + r_parallel**2)
    mu = r_parallel/r
    
    assert mean(r_perp,r_parallel) == pytest.approx(m_10(r)*mu,rel=0.01)

    true_c2 = mu**2*c_20(r) + (1-mu**2)*c_02(r)
    assert std(r_perp,r_parallel) == pytest.approx(np.sqrt(true_c2), rel=0.01)

    true_c3 = 3.*mu*(1-mu**2)*c_12(r) + mu**3*c_30(r)
    true_gamma1 = true_c3/true_c2**(3./2.)
    assert gamma1(r_perp,r_parallel) == pytest.approx(true_gamma1, rel=0.01)

    true_c4 = (1-mu**2)**2*c_04(r) + 6*mu**2*(1-mu**2)*c_22(r) + mu**4*c_40(r)
    true_gamma2 = true_c3/true_c2**2. - 3.
    assert gamma2(r_perp,r_parallel) == pytest.approx(true_gamma2, rel=0.01)


import numpy as np
import pytest
from gsm.models.gaussian import from_radial_transverse

def test__from_radial_transverse():
    m_10 = lambda x: x
    c_20 = lambda x: x**2
    c_02 = lambda x: (x-5)**2
 
    mean, std = from_radial_transverse.project_moments(m_10, c_20, c_02)

    r_perp = 10.
    r_parallel = 5.
    r = np.sqrt(r_perp**2 + r_parallel**2)
    mu = r_parallel/r
    
    assert mean(r_perp,r_parallel) == pytest.approx(m_10(r)*mu,rel=0.01)
    assert std(r_perp,r_parallel) == pytest.approx(np.sqrt(mu**2*c_20(r) + (1-mu**2)*c_02(r)), rel=0.01)



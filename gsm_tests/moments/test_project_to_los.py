from collections import namedtuple
import pytest
import numpy as np
from gsm.moments.project_to_los import project_to_los

def test__project_to_los():
    m_10 = lambda x: x
    c_20 = lambda x: x**2
    c_02 = lambda x: (x-5)**2

    Moments = namedtuple('Moments', ['m_10', 'c_20', 'c_02'])
    moments = Moments(m_10, c_20, c_02)

    mean = project_to_los(moments, 1, mode='m')
    c_2 = project_to_los(moments, 2, mode='c')

    r_perp = 10.
    r_parallel = 5.
    r = np.sqrt(r_perp**2 + r_parallel**2)
    mu = r_parallel/r
    
    assert mean(r_perp,r_parallel) == pytest.approx(m_10(r)*mu,rel=0.01)
    assert c_2(r_perp,r_parallel) == pytest.approx(mu**2*c_20(r) + (1-mu**2)*c_02(r), rel=0.01)

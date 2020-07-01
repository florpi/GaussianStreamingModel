import numpy as np
import pytest
from scipy.integrate import simps
from gsm.models.skewt import skewt 

def test__norm():
    x = np.linspace(-100,100,100)
    w, v_c, alpha, nu = 2,2,-1,4
    norm = simps(skewt.pdf(x, w, v_c, alpha, nu),
            x)

    assert norm == pytest.approx(1., rel=0.01) 

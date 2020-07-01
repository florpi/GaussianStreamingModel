import numpy as np
from collections import namedtuple
from typing import Callable
from gsm.models.gaussian import from_los
from gsm.moments.project_to_los import project_to_los

def project_moments(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    """
    Args:
        m_10: function that takes pair separation (r) as input and returns the mean 
        radial pairwise velocity
        c_20:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity
        c_02:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity

    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    Moments = namedtuple('Moments', ['m_10', 'c_20', 'c_02'])
    moments = Moments(m_10, c_20, c_02)
    mean = project_to_los(moments, 1, mode='m')
    c_2 = project_to_los(moments, 2, mode='c')
    std = lambda r_perp, r_parallel: np.sqrt(c_2(r_perp, r_parallel))
    return mean, std

def moments2gaussian(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    mean, std = project_moments(m_10, c_20, c_02)
    return from_los.losmoments2gaussian(mean, std)


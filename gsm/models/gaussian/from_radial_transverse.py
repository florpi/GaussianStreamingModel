import numpy as np
from typing import Callable

from gsm.models.gaussian import from_los

def moments2gaussian(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
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
    mean = project_moments(m_10)
    std = project_moments(c_02, c_20)
    return from_los.losmoments2gaussian(mean, std)


import numpy as np
from typing import Callable
from scipy.stats import norm

def losmoments2gaussian(mean: Callable, scale: Callable)->Callable:
    """
    Args:
        mean: function that takes r_parallel and r_perp as inputs and returns the mean 
        line of sight pairwise velocity
        std:  function that takes r_parallel and r_perp as inputs and returns the standard 
        deviation of the line of sight pairwise velocity


    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    def pdf_los(vlos: np.array, r_perp: np.array, r_parallel: np.array):
        return norm.pdf(
            vlos, loc=mean(r_perp, r_parallel), scale=scale(r_perp, r_parallel)
        )

    return pdf_los

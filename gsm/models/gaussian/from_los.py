import numpy as np
from scipy.stats import norm

def losmoments2gaussian(mean, scale):
    """
    Args:
        mean: np.array
        std: np.array

    Returns:
        pdf_los: function


    """

    def pdf_los(vlos, r_perp, r_parallel):

        return norm.pdf(
            vlos, loc=mean(r_parallel, r_perp), scale=scale(r_parallel, r_perp)
        )

    return pdf_los

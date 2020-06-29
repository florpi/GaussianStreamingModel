import numpy as np
from streaming.moments.project_moments import project_moment_to_los
import streaming.moments.compute_moments as cm 
from scipy.stats import norm


def losmoments2gaussian(mean, scale):
    '''
    Args:
        mean: np.array
        std: np.array

    Returns:
        pdf_los: function


    '''

    def pdf_los(vlos, rperp, rparallel):

        return norm.pdf(vlos, 
                loc = mean(rparallel, rperp), 
                scale = scale(rpararallel, rperp)
                )
    return pdf_los



from scipy.stats import t
import numpy as np


def pdf(v, w, v_c, alpha, nu):
    """ Probability Density Function of a Skewed-Student-t distribution in one dimension.
    Args: 
	    v: random variable.
	    w: scale parameter.
	    v_c: location parameter.
	    alpha: skewness parameter.
	    nu: degrees of freedom.
    Returns:
	    Skewt PDF evaluated at v
    """
    rescaled_v = (v - v_c) / w
    cdf_arg = alpha * rescaled_v * ((nu + 1) / (rescaled_v ** 2 + nu)) ** 0.5
    values = (nu + 1) / (rescaled_v ** 2 + nu)
    return (
        2.0 / w * t.pdf(rescaled_v, scale=1, df=nu) * t.cdf(cdf_arg, df=nu + 1, scale=1)
    )

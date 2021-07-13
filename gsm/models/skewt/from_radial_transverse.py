import numpy as np
from pathlib import Path
from collections import namedtuple
from typing import Callable
from gsm.models.skewt import from_los
from gsm.moments.project_to_los import project_to_los
from gsm.models.skewt.moments2parameters import (
    interpolate_moments2parameters,
    direct_spline_moments2parameters,
)


def project_moments(
    m_10: Callable,
    c_20: Callable,
    c_02: Callable,
    c_12: Callable,
    c_30: Callable,
    c_22: Callable,
    c_40: Callable,
    c_04: Callable,
) -> Callable:
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

    Moments = namedtuple(
        "Moments", ["m_10", "c_20", "c_02", "c_12", "c_30", "c_22", "c_40", "c_04"]
    )
    moments = Moments(m_10, c_20, c_02, c_12, c_30, c_22, c_40, c_04)
    mean = project_to_los(moments, 1, mode="m")
    c_2 = project_to_los(moments, 2, mode="c")
    c_3 = project_to_los(moments, 3, mode="c")
    c_4 = project_to_los(moments, 4, mode="c")
    std = lambda r_perp, r_parallel: np.sqrt(c_2(r_perp, r_parallel))

    gamma1 = lambda r_perp, r_parallel: c_3(r_perp, r_parallel) / c_2(
        r_perp, r_parallel
    ) ** (3.0 / 2.0)
    gamma2 = (
        lambda r_perp, r_parallel: c_4(r_perp, r_parallel)
        / c_2(r_perp, r_parallel) ** 2
        - 3.0
    )
    return mean, std, gamma1, gamma2


def moments2skewt(
    m_10: Callable,
    c_20: Callable,
    c_02: Callable,
    c_12: Callable,
    c_30: Callable,
    c_40: Callable,
    c_04: Callable,
    c_22: Callable,
    r_max: float=70.,
    n_eval: int = 70,
    use_spl: bool = False,
) -> Callable:

    mean, std, gamma1, gamma2 = project_moments(
        m_10=m_10,
        c_20=c_20,
        c_02=c_02,
        c_12=c_12,
        c_30=c_30,
        c_40=c_40,
        c_04=c_04,
        c_22=c_22,
    )
    r_perp = np.geomspace(0.7, r_max, n_eval)
    r_parallel = np.geomspace(0.7, r_max, n_eval)
    if not use_spl:
        w, v_c, alpha, nu = interpolate_moments2parameters(
            r_perp, r_parallel, mean=mean, std=std, gamma1=gamma1, gamma2=gamma2
        )
    else:
        w, v_c, alpha, nu = direct_spline_moments2parameters(
            r_perp, r_parallel, mean=mean, std=std, gamma1=gamma1, gamma2=gamma2
        )
    return from_los.losmoments2skewt(w, v_c, alpha, nu)

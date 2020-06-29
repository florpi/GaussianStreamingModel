import numpy as np
from typing import NamedTuple, Callable
from scipy.special import binom


def get_moment(
    moments: NamedTuple, r: np.array, r_order: int, t_order: int, mode: str
) -> np.array:
    """
    Given a named tuple containing the radial and transverse moments, returns the ```r_order```
    radial moment and the ```t_order``` transverse moment.

    Args:
        moments: Named tuple containing the radial and transverse moments. 
        r:  pair separation.
        r_order: order of the radial moment
        t_order: order of the transverse moments
        mode: either ```c``` for central moments or ```m``` for moments.

    Returns:
        moment

    Example naming moments Tuple: ('m_10': Radial mean, 'c_20': Second order radial central moment)
    """
    if t_order % 2 != 0:
        # Due to isotropy all momens with t_order odd vanish
        return np.zeros_like(r)
    elif (r_order == 0) and (t_order == 0):
        # The PDF is normalised
        return np.ones_like(r)
    elif (mode == "c") and (r_order + t_order == 1):
        # The first order central moments are zero
        return np.zeros_like(r)
    else:
        return getattr(moments, f"{mode}_{r_order}{t_order}")(r)


def project_to_los(moments: NamedTuple, n: int, mode: str = "c") -> Callable:
    """ 
    Project the moments of the radial and tangential velocity field onto the line of sight moments.

    Args:
        moments: Named tuple containing the radial and transverse moments.
        n: order of the moment.
        mode: Type of moment. If central moments use c, if moments about the origin use m.
    Returns:
        2D function of r_parallel and r_perpendicular that returns the 
        n-th moment of the line of sight velocity PDF 
    """

    def los_moment(r_perpendicular, r_parallel):
        r_perpedicular = np.atleast_2d(r_perpendicular)
        r_parallel = np.atleast_2d(r_parallel)

        r = np.sqrt(r_parallel ** 2 + r_perpendicular ** 2)
        mu = r_parallel / r

        return np.sum(
            [
                binom(n, k)
                * mu ** k
                * np.sqrt(1 - mu ** 2) ** (n - k)
                * get_moment(moments, r, r_order=k, t_order=n - k, mode=mode)
                for k in range(n + 1)
            ],
            axis=0,
        )

    return los_moment

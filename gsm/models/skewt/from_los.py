import numpy as np
from gsm.models.skewt import skewt
from typing import Callable
from scipy.special import gamma
from scipy.stats import t
from scipy.optimize import fsolve, curve_fit
from scipy.special import gamma
from scipy.interpolate import interp2d


def losmoments2skewt(w: Callable, v_c: Callable, alpha: Callable, nu: Callable):
    def pdf_los(vlos, r_perp, r_parallel):
        return skewt.pdf(
            vlos,
            w(r_perp, r_parallel),
            v_c(r_perp, r_parallel),
            alpha(r_perp, r_parallel),
            nu(r_perp, r_parallel),
        )

    return pdf_los

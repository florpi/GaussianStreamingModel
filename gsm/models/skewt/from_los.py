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
        # tricky hack, RectBivariateSpline sadly only takes sorted values, but r_parallel
        # won't be necessarily sorted
        sorted_r_perp = np.sort(r_perp[:, 0])
        idx_to_unsort_perp = r_perp[:, 0].argsort().argsort()

        sorted_r_parallel = np.sort(r_parallel[0, :])
        idx_to_unsort_parallel = r_parallel[0, :].argsort().argsort()
        return skewt.pdf(
            v=vlos,
            w=w(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            v_c=v_c(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            alpha=alpha(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            nu=nu(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
        )

    return pdf_los

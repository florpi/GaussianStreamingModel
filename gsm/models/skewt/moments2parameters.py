import numpy as np
from typing import List, Callable
from scipy.special import gamma
from scipy.optimize import fsolve, minimize, root
from scipy.interpolate import RectBivariateSpline


def gamma1_constrain(alpha, dof, gamma1):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    return gamma1 - delta * b_dof * (
        (dof * (3 - delta ** 2)) / (dof - 3)
        - 3 * dof / (dof - 2.0)
        + 2 * delta ** 2 * b_dof ** 2
    ) * (dof / (dof - 2) - delta ** 2 * b_dof ** 2) ** (-1.5)


def gamma2_constrain(alpha, dof, gamma2):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    return (
        gamma2
        - (
            3 * dof ** 2 / ((dof - 2) * (dof - 4))
            - 4 * delta ** 2 * b_dof ** 2 * dof * (3 - delta ** 2) / (dof - 3)
            + 6 * delta ** 2 * b_dof ** 2 * dof / (dof - 2)
            - 3 * delta ** 4 * b_dof ** 4
        )
        * (dof / (dof - 2.0) - delta ** 2 * b_dof ** 2) ** (-2.0)
        + 3.0
    )


def constrains(x, gamma1, gamma2):
    alpha, nu = x
    return (gamma1_constrain(alpha, nu, gamma1), gamma2_constrain(alpha, nu, gamma2))


def moments2parameters(mean, std, gamma1, gamma2, p0=(-0.7, 5)):
    alpha, nu = fsolve(constrains, p0, args=(gamma1, gamma2))
    delta = alpha / np.sqrt(1 + alpha ** 2)
    b = (nu / np.pi) ** 0.5 * gamma((nu - 1) / 2.0) / gamma(nu / 2.0)
    w = std / np.sqrt(nu / (nu - 2) - delta ** 2 * b ** 2)
    v_c = mean - w * delta * b
    return w, v_c, alpha, nu


def interpolate_moments2parameters(
    r_perp: np.array,
    r_parallel: np.array,
    mean: Callable,
    std: Callable,
    gamma1: Callable,
    gamma2: Callable,
)->List[Callable]:
    st_parameters = np.zeros((r_perp.shape[0], r_parallel.shape[0], 4))
    for i in range(len(r_perp)):
        for j in range(len(r_parallel)):
            st_parameters[i, j, :] = moments2parameters(
                mean(r_perp[i], r_parallel[j])[0][0], 
                std(r_perp[i], r_parallel[j])[0][0], 
                gamma1(r_perp[i], r_parallel[j])[0][0], 
                gamma2(r_perp[i], r_parallel[j])[0][0], 
            )
    callable_st_parameters = []
    for p in range(st_parameters.shape[-1]):
        callable_st_parameters.append(
                RectBivariateSpline(r_perp, r_parallel, st_parameters[:,:, p])
        )

    return callable_st_parameters

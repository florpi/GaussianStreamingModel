import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Callable
from scipy.special import gamma
from scipy.optimize import fsolve, minimize, root
from scipy.interpolate import RectBivariateSpline

spl_path = Path(__file__).resolve().parents[0] / "gamma2params.csv"


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


def moments2parameters_low_order(mean, std, alpha, nu):
    delta = alpha / np.sqrt(1 + alpha ** 2)
    b = (nu / np.pi) ** 0.5 * gamma((nu - 1) / 2.0) / gamma(nu / 2.0)
    w = std / np.sqrt(nu / (nu - 2) - delta ** 2 * b ** 2)
    v_c = mean - w * delta * b
    return w, v_c


def moments2parameters(mean, std, gamma1, gamma2, p0=(-0.7, 5)):
    alpha, nu = fsolve(constrains, p0, args=(gamma1, gamma2))
    w, v_c = moments2parameters_low_order(mean, std, alpha, nu)
    return w, v_c, alpha, nu


def interpolate_moments2parameters(
    r_perp: np.array,
    r_parallel: np.array,
    mean: Callable,
    std: Callable,
    gamma1: Callable,
    gamma2: Callable,
) -> List[Callable]:
    st_parameters = np.zeros((r_perp.shape[0], r_parallel.shape[0], 4))
    gamma1_values = []
    gamma2_values = []
    for i in range(len(r_perp)):
        for j in range(len(r_parallel)):
            gamma1_values.append(gamma1(r_perp[i], r_parallel[j])[0][0])
            gamma2_values.append(gamma2(r_perp[i], r_parallel[j])[0][0])
            st_parameters[i, j, :] = moments2parameters(
                mean(r_perp[i], r_parallel[j])[0][0],
                std(r_perp[i], r_parallel[j])[0][0],
                gamma1(r_perp[i], r_parallel[j])[0][0],
                gamma2(r_perp[i], r_parallel[j])[0][0],
            )
    callable_st_parameters = []
    for p in range(st_parameters.shape[-1]):
        callable_st_parameters.append(
            RectBivariateSpline(r_perp, r_parallel, st_parameters[:, :, p])
        )

    return callable_st_parameters


def direct_spline_moments2parameters(
    r_perp: np.array,
    r_parallel: np.array,
    mean: Callable,
    std: Callable,
    gamma1: Callable,
    gamma2: Callable,
) -> List[Callable]:

    df = pd.read_csv(spl_path)
    alpha_interp, nu_interp = get_interpolators(df)
    gamma1_values = gamma1(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    gamma2_values = gamma2(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    mean_values = mean(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    std_values = std(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    alpha = alpha_interp(gamma1_values, gamma2_values, grid=False)
    nu = nu_interp(gamma1_values, gamma2_values, grid=False)
    w, v_c = moments2parameters_low_order(mean_values, std_values, alpha, nu)
    st_parameters = [w, v_c, alpha, nu]
    callable_st_parameters = []
    for param in st_parameters:
        callable_st_parameters.append(RectBivariateSpline(r_perp, r_parallel, param))
    return callable_st_parameters


def generate_gamma_grid(min_gamma1, min_gamma2, max_gamma1, max_gamma2, n):
    p0 = (-0.7, 5)
    gamma1_values = np.linspace(min_gamma1, max_gamma1, n)
    gamma2_values = np.linspace(min_gamma2, max_gamma2, n)
    rows = []
    for i, gamma1 in enumerate(gamma1_values):
        for j, gamma2 in enumerate(gamma2_values):
            alpha, nu = fsolve(constrains, p0, args=(gamma1, gamma2))
            if alpha == -0.7 and nu == 5.0:
                continue
            rows.append([gamma1, gamma2, alpha, nu])
    return pd.DataFrame(rows, columns=["gamma1", "gamma2", "alpha", "nu"])


def get_interpolators(df):
    gamma1 = np.unique(df["gamma1"].values)
    gamma2 = np.unique(df["gamma2"].values)

    alpha_spline = RectBivariateSpline(
        gamma1, gamma2, df["alpha"].to_numpy().reshape((len(gamma1), len(gamma2))),
    )
    nu_spline = RectBivariateSpline(
        gamma1, gamma2, df["nu"].to_numpy().reshape((len(gamma1), len(gamma2))),
    )
    return alpha_spline, nu_spline


if __name__ == "__main__":
    df = generate_gamma_grid(-1.0, 0.0, 1.5, 4.0, n=1000)
    df.to_csv(
        "/cosma/home/dp004/dc-cues1/GaussianStreamingModel/gsm_tests/models/skewt/gamma2params.csv",
        index=False,
    )

    alpha, nu = get_interpolators(df)
    print(alpha(-0.6, 1.4))
    print(nu(-0.6, 1.4))

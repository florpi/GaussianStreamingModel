import numpy as np
from scipy.special import gamma


def mean(v_c, w, alpha, dof):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)
    return v_c + w * delta * b_dof


def std(w, alpha, dof):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)
    return np.sqrt(w ** 2 * (dof / (dof - 2) - delta ** 2 * b_dof ** 2))


def gamma1(alpha, dof):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    return (
        delta
        * b_dof
        * (
            (dof * (3 - delta ** 2)) / (dof - 3)
            - 3 * dof / (dof - 2.0)
            + 2 * delta ** 2 * b_dof ** 2
        )
        * (dof / (dof - 2) - delta ** 2 * b_dof ** 2) ** (-1.5)
    )


def gamma2(alpha, dof):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    value = (
        3 * dof ** 2 / ((dof - 2) * (dof - 4))
        - 4 * delta ** 2 * b_dof ** 2 * dof * (3 - delta ** 2) / (dof - 3)
        + 6 * delta ** 2 * b_dof ** 2 * dof / (dof - 2)
        - 3 * delta ** 4 * b_dof ** 4
    ) * (dof / (dof - 2.0) - delta ** 2 * b_dof ** 2) ** (-2.0) - 3.0

    return value


def parameters2moments(w, v_c, alpha, dof):
    return (
        mean(v_c, w, alpha, dof),
        std(w, alpha, dof),
        gamma1(alpha, dof),
        gamma2(alpha, dof),
    )

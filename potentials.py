#!/usr/bin/env python

"""Defines various test 1D and 2D potential energy functions.

   Author: Dominic Phillips (dominicp6)
"""

import autograd.numpy as np


def cart2pol(x: float, y: float) -> (float, float):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return r, theta


# Shallow well (1D)
def shallow_well_potential(x: float) -> float:
    return 0.01 * x**2


# Double well (1D)
def double_well_potential(x: float) -> float:
    h = 2
    c = 2
    return -(1 / 4) * (x**2) * (h**4) + (1 / 2) * (c**2) * (x**4)


# Quadruple Well (2D)
def quadruple_well_potential(x: np.array) -> float:
    h = 2
    c = 2
    return (-(1 / 4) * (x[0] ** 2) * (h**4) + (1 / 2) * (c**2) * (x[0] ** 4)) + (
        -(1 / 4) * (x[1] ** 2) * (h**4) + (1 / 2) * (c**2) * (x[1] ** 4)
    )


# Ring Double Well (2D)
def ring_double_well_potential(x: np.array) -> float:
    theta0 = np.pi
    r0 = 1
    w = 0.2
    d = 5
    r, theta = cart2pol(x[0], x[1])

    return (
        (1 / r) * np.exp(r / r0)
        - d * np.exp(-((x[0] - r0) ** 2 + (x[1]) ** 2) / (2 * w**2))
        - d
        * np.exp(
            -((x[0] - r0 * np.cos(theta0)) ** 2 + (x[1] - r0 * np.sin(theta0)) ** 2)
            / (2 * w**2)
        )
    )


# Moro-Cardin potential (2D)
def moro_cardin_potential(x: np.array) -> float:
    return 5 * (x[0] ** 2 - 1) ** 2 + 10 * np.arctan(7 * np.pi / 9) * x[1] ** 2


# Muller-Brown Potential (2D)
def muller_brown_potential(x: np.array) -> float:
    A = (-200, -100, -170, 15)
    a = (-1, -1, -6.5, 0.7)
    b = (0, 0, 11, 0.6)
    c = (-10, -10, -6.5, 0.7)
    x0 = (1, 0, -0.5, -1)
    y0 = (0, 0.5, 1.5, 1)

    V = 0
    for k in range(4):
        V += A[k] * np.exp(
            a[k] * (x[0] - x0[k]) ** 2
            + b[k] * (x[0] - x0[k]) * (x[1] - y0[k])
            + c[k] * (x[1] - y0[k]) ** 2
        )

    return V

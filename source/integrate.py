import numpy as np


def generate_grid(T, freq, method='trapezoidal'):
    # maybe implements simpson's method

    T = T
    freq = freq
    dt = 1 / freq
    t = np.arange(0, T + dt, dt)

    weight = np.ones_like(t) 
    weight[0] = weight[0] / 2
    weight[1] = weight[-1] / 2
    weight = weight / weight.sum() * T
    return t, weight


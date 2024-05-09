import numpy as np

def lipschitz_constant_1d(func, domain, num_points, eps):
    """
    Estimate the Lipschitz constant of a 1d function within a given domain.
    """
    x0, xn = domain
    
    points = np.linspace(x0, xn + eps, num=num_points)
    grid = np.array(np.meshgrid(points, points))

    grid = grid.reshape(2, -1).T

    x_diff = np.abs(grid[:, 0] - grid[:,1])

    mask = x_diff > eps #  Avoid division by zero

    x1 = grid[:, 0].reshape(-1,1)
    x2 = grid[:, 1].reshape(-1,1)
    f_diff = np.abs(func(x1) - func(x2))

    lipschitz_constant = np.max(f_diff[mask] / x_diff[mask])

    return lipschitz_constant     
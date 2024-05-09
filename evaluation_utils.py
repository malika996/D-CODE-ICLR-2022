import numpy as np
import equations
import data
from config import get_interpolation_config
from integrate import generate_grid
import basis
import data
from interpolate import get_ode_data

class Metric:

  def __init__(self,
               equation_number,
               ode_true,
               f_hat,
               t,
               freq,
               n_sample,

               ):

    _, weights = generate_grid(t, freq)
    self.weights = weights

    self.ode_true = ode_true
    self.ode_hat = equations.InferredODE(self.ode_true.dim_x, f_hat_list=f_hat, T=t)
    self.equation_number = equation_number

    self.dg_true = data.DataGenerator(self.ode_true, t, freq=freq, n_sample=n_sample, noise_sigma=0.,
                                                 init_high=self.ode_true.init_high,)
    self.xt_true = self.dg_true.xt

    self.dg_hat = data.DataGenerator(self.ode_hat, t, freq=freq, n_sample=n_sample, noise_sigma=0.,
                            init_high=self.ode_true.init_high, )
    self.xt_hat = self.dg_hat.xt

    x_in = [self.xt_true[:, i] for i in range(self.xt_true.shape[1])]

    dxdt_true_list = []
    dxdt_hat_list = []
    for sample in x_in:
      dxdt_true_list.append(self.ode_true._dx_dt(sample.T)[self.equation_number])
      dxdt_hat_list.append(f_hat[self.equation_number](sample).flatten())

    self.dxdt_true = np.squeeze(np.array(dxdt_true_list)).T
    self.dxdt_hat = np.squeeze(np.array(dxdt_hat_list)).T
    self.xt_true = np.squeeze(self.xt_true)
    self.xt_hat = np.squeeze(self.xt_hat)

    
  def d_x(self):
    """ Distance measure expressed as \|(f_hat - f_true) o x\|_2"""
    return np.sqrt(self.weights @ ((self.dxdt_hat - self.dxdt_true) ** 2))

  def x_norm(self):
    """ Distance measure expressed as \|x_hat - x_true \|_2"""
    if self.ode_true.dim_x == 1:
      return np.sqrt(self.weights  @ ((self.xt_hat - self.xt_true) ** 2))
    else:
      x_norm_list = []
      for dim_x in range(self.ode_true.dim_x):
        x_norm_list.append(np.sqrt(self.weights  @ ((self.xt_hat[:,:,dim_x] - self.xt_true[:,:,dim_x]) ** 2)))
      return x_norm_list

  def C_fxg(self, n_basis):
    """D-CODE objective"""
    freq = self.dg_true.freq
    t = np.arange(0, self.dg_true.T + 1/freq, 1/freq)
    config = get_interpolation_config(self.ode_true, 0)
    basis = config['basis']
    basis_func = basis(self.dg_true.T, n_basis)
    g_dot = basis_func.design_matrix(self.dg_true.solver.t, derivative=True)
    self.g_dot = g_dot
    g = basis_func.design_matrix(self.dg_true.solver.t, derivative=False)
    self.g = g

    if self.ode_true.dim_x == 1:
      c2 = ( self.xt_true * self.weights[:, None]).T @ g_dot
      theta_pred = np.matmul((self.dxdt_hat * self.weights[:, None]).T,g)
      return np.sum((theta_pred + c2) ** 2, axis = -1)
    else:
      loss_list = []
      for dim_x in range(self.ode_true.dim_x):
        c2 = ( self.xt_true[:,:, dim_x] * self.weights[:, None]).T @ g_dot
        theta_pred = np.matmul((self.dxdt_hat * self.weights[:, None]).T,g)
        loss_list.append(np.sum((theta_pred + c2) ** 2, axis = -1))
      return loss_list
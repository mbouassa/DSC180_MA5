from utils import *
import numpy as np, math
import numpy as np
import matplotlib.pyplot as plt
import sys

def kernel(data):

  n, noise_std, gamma, p, lam = 50, 0.1, 10, 8, 1e-8

  def target_poly(x):
      return x**5 - 3* x**4

  target_xsinx = lambda x: x * np.sin(x)

  K_poly = lambda x, z, gamma=gamma, p=p: (1 + np.sum(x*z)/gamma)**p

  def K_poly_mat(x, z, gamma=gamma, p=p): 
      """
      if x is (m, d) and z is (n, d)
      then output is (m, n)
      """
      return (1 + np.sum(x[:,None,:]*z[None,:,:], axis=-1)/gamma)**p

  Phi_poly = lambda X, p: np.hstack([X**i for i in range(p+1)])

  Phi_poly_scaled = lambda X, p, gamma=gamma: (
      Phi_poly(X, p)*
      np.array([np.sqrt(math.comb(p, i)/gamma**i) for i in range(p+1)])
  )

  

  target_fn = target_xsinx
  x = np.linspace(-5, 5, 1000).reshape(-1, 1)
  y = target_fn(x) + np.random.randn(*x.shape) * noise_std
  X = np.random.rand(n, 1) * 10 - 5
  Y = target_fn(X) + np.random.randn(*X.shape) * noise_std
  
  K_XX = np.array([[K_poly(x1, x2) for x2 in X] for x1 in X])
  K_XX_broadcasting = K_poly_mat(X, X)
  alpha_hat = np.linalg.solve(K_XX+lam*np.eye(len(X)), Y)

  K_xX = K_poly_mat(x, X)

  yhat = K_xX @ alpha_hat
  return yhat


if __name__ == (__main__):

  args = sys.argv[1:]
  if args[0] == "test":
    
    print(kernel(data))
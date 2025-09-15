import numpy as np
A = 0.1
omega = np.pi
epsilon = 0.25
def double_gyre(x, y, t=0):
    a = epsilon * np.sin(omega * t)
    b = 1 - 2 * a
    f = a * x**2 + b * x
    df_dx = 2 * a * x + b
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df_dx
    return u, v
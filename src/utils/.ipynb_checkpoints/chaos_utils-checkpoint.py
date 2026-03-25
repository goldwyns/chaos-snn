# src/utils/chaos_utils.py
import numpy as np

def sigma_tanh(u):
    return 0.5 * (1.0 + np.tanh(u))

def chaos_modulate_numpy(X, alpha=3.5, beta=0.2, gamma=0.5, lam=0.7, z0=None, rng_seed=42):
    """
    Vectorized chaos modulation for windows X shaped (N, C, T)
    Returns u (mixed drive) same shape (N, C, T), dtype=float32.

    Parameters:
      X: np.ndarray (N, C, T) - preprocessed x' (z-scored / bandpassed)
      alpha, beta, gamma, lam: floats as in your equations
      z0: optional initial z (N,C) or None -> random in (0,1)
    """
    N, C, T = X.shape
    X = X.astype(np.float32)

    rng = np.random.default_rng(rng_seed)
    if z0 is None:
        # initialize z at random in (0,1)
        z = rng.random((N, C)).astype(np.float32)
    else:
        z = np.array(z0, dtype=np.float32).reshape(N, C)

    u_out = np.zeros_like(X, dtype=np.float32)

    # iterate time dimension vectorized
    for n in range(T):
        x_t = X[:, :, n]               # (N, C)
        # logistic term alpha * z * (1 - z)
        term = alpha * z * (1.0 - z)   # (N, C)
        drive = term + beta * x_t      # (N, C)
        z = (1.0 - gamma) * z + gamma * sigma_tanh(drive)  # update, (N,C)
        z_tilde = 2.0 * z - 1.0        # (N, C)
        u = lam * x_t + (1.0 - lam) * z_tilde  # mixed drive (N, C)
        u_out[:, :, n] = u

    return u_out  # float32

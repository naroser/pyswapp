import numpy as np

# %% helper functions for solving system of equations
def linear_LSQR(x,y,w=None):
    """
    Estimate coefficients of a line y = k*x+phi0

    Parameters
    ----------
    x : np.ndarray, offsets
    y : np.ndarray, phases
    w : np.ndarray, weights

    Returns
    -------
    k: float, wavenumber
    phi0: float, shift
    """


    if w is None:
        W = np.diag(np.ones(len(x)))
    else:
        W = np.diag(w) # weighted LSQR

    A = np.vstack([-x, np.ones(len(x))]).T # design matrix

    N = np.transpose(A) @ W @ A
    beta = np.linalg.inv(N) @ np.transpose(A)@ W @ y

    k = beta[0]
    phi0 = beta[1]

    return k, phi0

def phase_response(x, k, phi0):
    """return the phase"""
    return -k * x + phi0

def compute_chi2(data, resp, error):
    """compute rms and chi^2"""
    misfit = data - resp
    error_weighted_misfit = np.abs(misfit) / error

    rms = np.sqrt(np.mean(np.abs(misfit) ** 2))
    rrms = np.sqrt(np.mean((np.abs(misfit)/np.abs(data)) ** 2))
    chi2 = np.sum(error_weighted_misfit**2)
    return rms, rrms, chi2

def tomo2D_phasediff(lam,f,A,dphi,w):
    """
    Tomographic like approach (Barone et al., 2019)

    Parameters
    ----------
    lam : int, regularization strength
    f : float, frequency value of the analysis
    A : np.ndarray, design matrix
    dphi : np.ndarray, phase differences
    w : np.ndarray, weights

    Returns
    -------
    phi_vel: np.ndarray, phase velocities
    phi_model: np.ndarray, predicted phase differences
    """

    nm = A.shape[1]

    # first-order finite difference operator
    G = np.zeros((nm - 1, nm))
    for i in range(nm - 1):
        G[i, i] = 1
        G[i, i + 1] = -1

    # weighted least squares matrices
    ATW = A.T @ w
    Ni = ATW @ A  # data term
    Mi = G.T @ G  # smoothness term

    # solve (Ni + λ² Mi) m = Aᵀ W dφ
    rhs = ATW @ dphi

    Mreg = Ni + (lam ** 2) * Mi
    m = np.linalg.solve(Mreg, rhs)
    omega = 2 * np.pi * f

    eps = 1e-12
    m = np.where(np.abs(m) < eps, eps, m)

    phi_vel = -omega / m  # phase velocity
    phi_model = A @ m  # predicted phase differences

    return phi_vel, phi_model


def lambda_search(f,A,dphi,w, scale = 0.6, lam_max = 200):
    """ Search for optimum weight

    Parameters
    ----------
    f : float, frequency value of the analysis
    A : np.ndarray, design matrix
    dphi : np.ndarray, phase differences
    w : np.ndarray, weights
    scale: float, percentage value to decrease lambda
    lam_max: int, maximum value of lambda

    Returns
    -------
    lam: int, optimal lambda

    """

    k = np.arange(int(np.log(1 / lam_max) / np.log(scale)) + 2)
    lam = (lam_max * scale ** k)
    lam = np.unique(lam[lam >= 1].astype(int))

    Cost = np.array([np.linalg.norm(dphi - tomo2D_phasediff(l, f, A, dphi, w)[1]) for l in lam])

    return lam[np.argmin(Cost)]


from collections.abc import Iterable
import numpy as np

# %% physics
def wavenumber(f,vel):
    """compute wavenumber"""
    return 2*np.pi*f/vel


def phase_velocity(f,k):
    """compute phase velocity"""
    return 2*np.pi*f/k


def slowness(vel):
    """compute seismic slowness"""
    return 1./vel


def wavelength(f,vel):
    """compute wavelength"""
    return vel/f


def frequency(lam,vel):
    """compute frequency"""
    return vel/lam


def lorentzian_err(offsets, vel, f, nchannels = 24, dx = 1, **kwargs):
    """
    Estimate dispersion-curve uncertainty after O'Neill et al. (2003).

    Parameters
    ----------
    offsets : array-like or None
        Receiver offsets. If provided, dx and nchannels are inferred from it.
    vel : float or array-like
        Phase velocity.
    f : float or array-like
        Frequency.
    nchannels : int, optional
        Number of channels (ignored if offsets provided).
    dx : float, optional
        Receiver spacing (ignored if offsets provided).
    maxerr : float
        Maximum relative velocity error (e.g., 0.4 means ≤40% of velocity).
    minvelerr : float
        Minimum absolute velocity error (m/s).
    a : float
        Parameter controlling error-bar tightening.

    Returns
    -------
    deltac : float or np.ndarray
        Estimated dispersion-curve error.


    References
    ----------
    O’Neill, A., Dentith, M., & List, R., 2003. Full-waveform P-SV
    reflectivity inversion of surface waves for shallow engineering
    applications, Exploration Geophysics, 34(3), 158–173.
    """

    maxerr = kwargs.get('maxerr', 0.4)
    minvelerr = kwargs.get('minvelerr', 20)
    a = kwargs.get('a', 0.3)

    # Ensure arrays
    vel = np.asarray(vel, dtype=float)
    f = np.asarray(f, dtype=float)

    # Determine dx and nchannels from offsets if provided
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=float)
        if len(offsets) < 2:
            raise ValueError("Offsets must contain at least two values.")
        nchannels = len(offsets)
        dx = abs(offsets[1] - offsets[0])

    # wavelength function must exist
    lam = wavelength(f, vel)

    # avoid division problems
    lam = np.maximum(lam, 1e-12)

    # geometric factor
    fac = 10 ** (1 / np.sqrt(nchannels * dx))

    # define a common denominator for clarity
    denom = 2 * vel / lam * (nchannels * fac) * dx
    denom = np.maximum(denom, 1e-12)

    avec = 1 / vel - 1 / denom
    bvec = 1 / vel + 1 / denom

    deltac = (10 ** -a) * np.abs(1 / avec - 1 / bvec)

    # Enforce max and min error thresholds
    deltac = np.maximum(deltac, minvelerr)
    deltac = np.minimum(deltac, maxerr * vel)

    # Return scalar if inputs were scalar
    if deltac.size == 1:
        return float(deltac)
    return deltac

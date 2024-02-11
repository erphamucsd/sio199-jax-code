'''
Date: 2/11/2024
Parametrization of large-scale condensation.
'''

import jax.numpy as jnp
from jax import jit

@jit
def get_large_scale_condensation_tendencies(psa, qa, qsat, itop):
    from physical_constants import p0, cp, alhc, alhs, grav
    from geometry import fsg, dhs


    ix, il, kx = psa.shape
    rhlsc = 0.9    # Maximum relative humidity threshold (at sigma=1)
    drhlsc = 0.1   # Vertical range of relative humidity threshold
    rhblsc = 0.95  # Relative humidity threshold for boundary layer
    trlsc = 4.0    # Relaxation time (in hours) for specific humidity
    
    # Initialize arrays
    precls = jnp.zeros((ix, il), dtype=psa.dtype)
    dtlsc = jnp.zeros((ix, il, kx), dtype=psa.dtype)
    dqlsc = jnp.zeros((ix, il, kx), dtype=psa.dtype)

    rtlsc = 1.0 / (trlsc * 3600.0)
    tfact = alhc / cp
    prg = p0 / 9.81

    psa2 = psa ** 2.0

    # Tendencies of temperature and moisture NB. A maximum heating rate is 
    # imposed to avoid grid-point-storm instability
    k = jnp.arange(1, kx)
    sig2 = fsg[k] ** 2.0
    rhref = rhlsc + drhlsc * (sig2 - 1.0)
    rhref = jnp.where(k == kx - 1, jnp.maximum(rhref, 0.0), rhref)

    dqmax = 10.0 * sig2 * rtlsc

    dqa = rhref * qsat[..., 1:] - qa[..., 1:]
    dqa_neg = jnp.where(dqa < 0.0, dqa, 0.0)

    itop = jnp.minimum(jnp.where(dqa_neg < 0.0, jnp.minimum(k + 1, kx - 1), kx - 1), kx - 1)
    dqlsc = jnp.where(dqa < 0.0, dqa * rtlsc, 0.0)
    dtlsc = jnp.where(dqa < 0.0, tfact * jnp.minimum(-dqlsc, dqmax * psa2), 0.0)

    # Large-scale precipitation
    k = jnp.arange(2, kx)
    pfact = dhs[k] * prg
    precls = jnp.sum(-pfact[..., jnp.newaxis] * dqlsc[..., 2:], axis=-1)

    precls *= psa

    return itop, precls, dtlsc, dqlsc
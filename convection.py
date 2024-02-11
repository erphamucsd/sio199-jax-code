'''
Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedke (1993) mass-flux convection scheme.
'''

import jax.numpy as jnp
from jax import lax

from speedyf90_types import p

# Constants for convection 
psmin = jnp.array(0.8, dtype=p) # Minimum (normalised) surface pressure for the occurrence of convection
trcnv = jnp.array(6.0, dtype=p) # Time of relaxation (in hours) towards reference state
rhbl = jnp.array(0.9, dtype=p) # Relative humidity threshold in the boundary layer
rhil = jnp.array(0.7, dtype=p) # Relative humidity threshold in intermeduate layers for secondary mass flux
entmax = jnp.array(0.5, dtype=p) # Maximum entrainment as a fraction of cloud-base mass flux
smf = jnp.array(0.8, dtype=p) # Ratio between secondary and primary mass flux at cloud-base

# Compute convective fluxes of dry static energy and moisture using a simplified mass-flux scheme
def get_convection_tendencies(psa, se, qa, qsat, itop, cbmf, precnv, dfse, dfqa):
    from physical_constants import p0, alhc, alhs, wvi, grav
    from geometry import fsg, dhs

    ix, il, kx = psa.shape  # Assuming psa has shape (ix, il, kx)

    # Initialization
    nl1 = kx - 1
    nlp = kx + 1
    fqmax = 5.0

    fm0 = p0 * dhs[-1] / (grav * trcnv * 3600.0)
    rdps = 2.0 / (1.0 - psmin)

    dfse = jnp.zeros_like(dfse)
    dfqa = jnp.zeros_like(dfqa)

    cbmf = jnp.zeros_like(cbmf)
    precnv = jnp.zeros_like(precnv)

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, fsg[:, None] - 0.5)**2.0
    sentr = entmax / jnp.sum(entr[:, :, 1:nl1], axis=-1)
    entr = jnp.where(jnp.arange(1, kx-1)[:, None] < nl1, entr * sentr[:, None], entr)

    # Check conditions for convection
    qdif = diagnose_convection(psa, se, qa, qsat)

    # Create index arrays
    i_indices = jnp.arange(ix)[:, None]
    j_indices = jnp.arange(il)[None, :]

    # Define functions for operations within the loop
    def convection_loop(i, j, args):
        itop_ij = itop[i, j]
        cbmf_ij = cbmf[i, j]

        # Boundary layer (cloud base)
        k = kx - 1
        k1 = k - 1

        # Maximum specific humidity in the PBL
        qmax = jnp.maximum(1.01 * qa[i, j, k], qsat[i, j, k])

        # Dry static energy and moisture at upper boundary
        sb = se[i, j, k1] + wvi[k1, 1] * (se[i, j, k] - se[i, j, k1])
        qb = qa[i, j, k1] + wvi[k1, 1] * (qa[i, j, k] - qa[i, j, k1])
        qb = jnp.minimum(qb, qa[i, j, k])

        # Cloud-base mass flux
        fpsa = psa[i, j] * jnp.minimum(1.0, (psa[i, j] - psmin) * rdps)
        fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif[i, j] / (qmax - qb))

        # Upward fluxes at upper boundary
        fus = fmass * se[i, j, k]
        fuq = fmass * qmax

        # Downward fluxes at upper boundary
        fds = fmass * sb
        fdq = fmass * qb

        # Net flux of dry static energy and moisture
        dfse = fds - fus
        dfqa = fdq - fuq

        # Intermediate layers (entrainment)
        def intermediate_layers(k, fse_fqa):
            k1 = k - 1

            # Fluxes at lower boundary
            dfse = fse_fqa[0]
            dfqa = fse_fqa[1]

            # Mass entrainment
            enmass = entr[k] * psa[i, j] * cbmf_ij
            fmass += enmass

            # Upward fluxes at upper boundary
            fus += enmass * se[i, j, k]
            fuq += enmass * qa[i, j, k]

            # Downward fluxes at upper boundary
            sb = se[i, j, k1] + wvi[k1, 1] * (se[i, j, k] - se[i, j, k1])
            qb = qa[i, j, k1] + wvi[k1, 1] * (qa[i, j, k] - qa[i, j, k1])
            fds = fmass * sb
            fdq = fmass * qb

            # Net flux of dry static energy and moisture
            dfse += fds - fus
            dfqa += fdq - fuq

            # Secondary moisture flux
            delq = rhil * qsat[i, j, k] - qa[i, j, k]
            if delq > 0.0:
                fsq = smf * cbmf_ij * delq
                dfqa += fsq
                dfqa += -fsq

            return dfse, dfqa

        # Perform loop for intermediate layers
        dfse, dfqa = lax.fori_loop(kx - 2, itop_ij, intermediate_layers, (dfse, dfqa))

        # Top layer (condensation and detrainment)
        k = itop_ij

        # Flux of convective precipitation
        qsatb = qsat[i, j, k] + wvi[k, 1] * (qsat[i, j, k+1] - qsat[i, j, k])
        precnv_ij = jnp.maximum(fuq - fmass * qsatb, 0.0)

        # Net flux of dry static energy and moisture
        dfse += fus - fds + alhc * precnv_ij
        dfqa += fuq - fdq - precnv_ij

        return dfse, dfqa, precnv.at[i, j].set(precnv_ij)

    # Perform loop over indices
    dfse, dfqa, precnv = lax.fori_loop(ix, lambda i, args: lax.fori_loop(il, lambda j, args: convection_loop(i, j, args), args), (dfse, dfqa, precnv))

    return dfse, dfqa
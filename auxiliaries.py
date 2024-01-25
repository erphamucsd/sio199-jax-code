'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

import jax
import jax.numpy as np

from params import ix, il

# Replacement for types file
p = np.float64

# Physical variables shared among all physics schemes
precnv = np.zeros((ix, il), dtype=p) # Convective precipitation [g/(m^2 s)], total
precls = np.zeros((ix, il), dtype=p) # Large-scale precipitation [g/(m^2 s)], total
snowcv = np.zeros((ix, il), dtype=p) # Convective precipitation [g/(m^2 s)], snow only
snowls = np.zeros((ix, il), dtype=p) # Large-scale precipitation [g/(m^2 s)], snow only
cbmf = np.zeros((ix, il), dtype=p) # Cloud-base mass flux
tsr = np.zeros((ix, il), dtype=p) # Top-of-atmosphere shortwave radiation (downward)
ssrd = np.zeros((ix, il), dtype=p) # Surface shortwave radiation (downward-only)
ssr = np.zeros((ix, il), dtype=p) # Surface shortwave radiation (net downward)
slrd = np.zeros((ix, il), dtype=p) # Surface longwave radiation (downward-only)
slr = np.zeros((ix, il), dtype=p) # Surface longwave radiation (net upward)
olr = np.zeros((ix, il), dtype=p) # Outgoing longwave radiation (upward)

# Third dimension -> 1:land, 2:sea, 3: weighted average
slru = np.zeros((ix, il, 3), dtype=p) # Surface longwave emission (upward)
ustr = np.zeros((ix, il, 3), dtype=p) # U-stress
vstr = np.zeros((ix, il, 3), dtype=p) # V-stress
shf = np.zeros((ix, il, 3), dtype=p) # Sensible heat flux
evap = np.zeros((ix, il, 3), dtype=p) # Evaporation [g/(m^2 s)]
hfluxn = np.zeros((ix, il, 3), dtype=p) # Net heat flux into surface

'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

import jax
import jax.numpy as np

from params import ix, il

# Replacement for types.py
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

'''NOT SURE ABOUT THIS SECTION.'''
# Make module variables mutable for updates
precnv = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, precnv)
precls = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, precls)
snowcv = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, snowcv)
snowls = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, snowls)
cbmf = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, cbmf)
tsr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, tsr)
ssrd = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, ssrd)
ssr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, ssr)
slrd = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, slrd)
slr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, slr)
olr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, olr)
slru = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, slru)
ustr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, ustr)
vstr = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, vstr)
shf = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, shf)
evap = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, evap)
hfluxn = jax.tree_util.tree_map(jax.tree_util.PartialMutableLeaf, hfluxn)

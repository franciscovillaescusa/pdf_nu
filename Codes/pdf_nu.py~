import numpy as np
import sys,os
import readgadget
import MAS_library as MASL

##################################### INPUT ###################################
root   = '/projects/QUIJOTE/Snapshots'
cosmo  = 'Mnu_p'
folder = 0

snapnum = 4

grid   = 512
ptypes = [2]
MAS    = 'CIC'
do_RSD = False
axis   = 0
###############################################################################

# get the name of the snapshot
snapshot = '%s/%s/%d/snapdir_%03d/snap_%03d'%(root,cosmo,folder,snapnum,snapnum)

# compute density field
delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)

# compute the modulus of the velocity
#V = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)

print(np.sum(delta))
print(np.min(delta), np.max(delta))

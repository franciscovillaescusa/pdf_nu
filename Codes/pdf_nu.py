import numpy as np
import sys,os
import readgadget
import MAS_library as MASL

##################################### INPUT ###################################
# cosmology parameters
root         = '/projects/QUIJOTE/Snapshots'
cosmo        = 'Mnu_p'
folder_out   = '../Results/0.01eV'
snapnum      = 4
realizations = 500

# density field parameters
grid   = 512
ptypes = [2]
MAS    = 'CIC'
do_RSD = False
axis   = 0

# pdf parameters
delta_min, delta_max = 0.0, 6.0
bins = 100
###############################################################################

# do a loop over all snapshots
for i in range(realizations):

    print(i)

    # get the name of the snapshot and the output file
    snapshot = '%s/%s/%d/snapdir_%03d/snap_%03d'%(root,cosmo,i,snapnum,snapnum)
    fout     = '%s/pdf_nu0.01_%d_%d_z=0.txt'%(folder_out,i,grid)

    # compute density field
    delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
    delta = delta/np.mean(delta, dtype=np.float64)

    print(np.sum(delta))
    print(np.min(delta), np.max(delta))

    # define pdf bins and compute pdf itself
    pdf_bins  = np.linspace(delta_min, delta_max, bins+1)
    pdf_mean  = 0.5*(pdf_bins[1:] + pdf_bins[:-1])
    pdf_width = pdf_bins[1:] - pdf_bins[:-1]
    pdf = np.histogram(delta,bins=pdf_bins)[0]
    pdf = pdf/pdf_width/np.sum(pdf, dtype=np.float64)

    # save results to file
    np.savetxt(fout, np.transpose([pdf_mean, pdf]))


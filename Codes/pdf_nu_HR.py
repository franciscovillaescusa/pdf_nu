import numpy as np
import sys,os,h5py
import readgadget
import MAS_library as MASL

##################################### INPUT ###################################
# cosmology parameters
root         = '/mnt/sdceph/users/fvillaescusa/Caputo'
realizations = 1

# density field parameters
ptypes = [2]
MAS    = 'CIC'
do_RSD = False
axis   = 0

# pdf parameters
bins = 200
###############################################################################

# redshift dictionary
z = {10:0, 9:0.5, 8:1, 7:2, 6:3, 5:4, 4:5, 3:6, 2:7, 1:8, 0:9}

# do a loop over the different grid sizes
for grid in [300, 400, 500]:

    # do a loop over the different neutrino masses
    for cosmo,prefix in zip(['0_HR'], ['0.1eV_HR']):

        # do a loop over the different redshifts
        for snapnum in [10,9,8,7,6,5,4,3,2,1,0]:

            # get name of output file
            fout = '../Results/Results_%s_%d_z=%s.hdf5'%(prefix,grid,z[snapnum])
            #if os.path.exists(fout):  continue

            # define the arrays containing the variance and the pdf of the fields
            var_tot = np.zeros(realizations,        dtype=np.float64)
            pdf_tot = np.zeros((realizations,bins), dtype=np.float64)

            # compute delta_max from snapshot
            snapshot = '%s/%s/snapdir_%03d/snap_%03d'%(root,cosmo,snapnum,snapnum)
            delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
            delta = delta/np.mean(delta, dtype=np.float64)
            delta_min, delta_max = np.min(delta), np.max(delta)

            # define the pdf bins
            pdf_bins  = np.linspace(delta_min, delta_max, bins+1)
            pdf_mean  = 0.5*(pdf_bins[1:] + pdf_bins[:-1])
            pdf_width = pdf_bins[1:] - pdf_bins[:-1]

            # compute pdf and variance
            i = 0
            var_tot[i] = np.var(delta)
            pdf_tot[i] = np.histogram(delta,bins=pdf_bins)[0]
            pdf_tot[i] = pdf_tot[i]/pdf_width/np.sum(pdf_tot[i], dtype=np.float64)

            # only master save results to file
            f = h5py.File(fout, 'w')
            f.create_dataset('variance', data=var_tot)
            f.create_dataset('pdf',      data=pdf_tot)
            f.create_dataset('pdf_bins', data=pdf_mean)
            f.close()



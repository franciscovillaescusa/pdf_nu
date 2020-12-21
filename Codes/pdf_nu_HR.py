import numpy as np
import sys,os,h5py
import readgadget
import MAS_library as MASL
import smoothing_library as SL

##################################### INPUT ###################################
# cosmology parameters
root         = '/mnt/sdceph/users/fvillaescusa/Caputo'
realizations = 1

# density field parameters
ptypes = [2]
MAS    = 'CIC'
do_RSD = False
axis   = 0
grid   = 1500

# smoothing parameters
BoxSize = 1000.0 #Mpc/h
Filter  = 'Gaussian'
threads = 1

# pdf parameters
bins = 200
###############################################################################

# redshift dictionary
z = {10:0, 9:0.5, 8:1, 7:2, 6:3, 5:4, 4:5, 3:6, 2:7, 1:8, 0:9}

# do a loop over the different grid sizes
for R in [2.0, 3.0, 4.0, 5.0, 7.5, 10.0]:

    # compute FFT of the filter
    W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

    # do a loop over the different neutrino masses
    for cosmo in ['0.1eV', '0.2eV', '0.4eV']:

        # do a loop over the different redshifts
        for snapnum in [10,9,8,7,6,5,4,3,2,1,0]:

            # get name of output file
            fout = '../Results/Results_Gaussian_%s_HR_%s_z=%s.hdf5'%(cosmo,R,z[snapnum])
            if os.path.exists(fout):  continue

            # define the arrays containing the variance and the pdf of the fields
            var_tot = np.zeros(realizations,        dtype=np.float64)
            pdf_tot = np.zeros((realizations,bins), dtype=np.float64)

            # compute delta_max from snapshot
            snapshot = '%s/%s/0_HR/snapdir_%03d/snap_%03d'%(root,cosmo,snapnum,snapnum)
            delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
            delta = delta/np.mean(delta, dtype=np.float64)
            delta_smoothed = SL.field_smoothing(delta, W_k, threads) #smooth the field
            delta_min, delta_max = np.min(delta_smoothed), np.max(delta_smoothed)
            del delta

            # define the pdf bins
            pdf_bins  = np.linspace(delta_min, delta_max, bins+1)
            pdf_mean  = 0.5*(pdf_bins[1:] + pdf_bins[:-1])
            pdf_width = pdf_bins[1:] - pdf_bins[:-1]

            # compute pdf and variance
            i = 0
            var_tot[i] = np.var(delta_smoothed)
            pdf_tot[i] = np.histogram(delta_smoothed,bins=pdf_bins)[0]
            pdf_tot[i] = pdf_tot[i]/pdf_width/np.sum(pdf_tot[i], dtype=np.float64)
            del delta_smoothed
            
            # only master save results to file
            f = h5py.File(fout, 'w')
            f.create_dataset('variance', data=var_tot)
            f.create_dataset('pdf',      data=pdf_tot)
            f.create_dataset('pdf_bins', data=pdf_mean)
            f.close()



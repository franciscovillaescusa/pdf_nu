from mpi4py import MPI
import numpy as np
import sys,os,h5py
import readgadget
import MAS_library as MASL

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

##################################### INPUT ###################################
# cosmology parameters
root         = '/projects/QUIJOTE/Snapshots'
realizations = 100

# density field parameters
ptypes = [2]
MAS    = 'CIC'
do_RSD = False
axis   = 0

# pdf parameters
bins = 200
###############################################################################

# redshift dictionary
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}

# find the numbers that each cpu will work with
numbers = np.where(np.arange(realizations)%nprocs==myrank)[0]

# do a loop over the different grid sizes
for grid in [300, 400, 500]:

    # do a loop over the different neutrino masses
    for cosmo,prefix in zip(['Mnu_p', 'Mnu_pp', 'Mnu_ppp'], ['0.1eV', '0.2eV', '0.4eV']):

        # do a loop over the different redshifts
        for snapnum in [4,3,2,1,0]:

            # get name of output file
            fout = '../Results/Results_%s_%d_z=%s.hdf5'%(prefix,grid,z[snapnum])
            #if os.path.exists(fout):  continue

            # define the arrays containing the variance and the pdf of the fields
            var     = np.zeros(realizations,        dtype=np.float64)
            var_tot = np.zeros(realizations,        dtype=np.float64)
            pdf     = np.zeros((realizations,bins), dtype=np.float64)
            pdf_tot = np.zeros((realizations,bins), dtype=np.float64)

            # compute delta_max from first snapshot
            snapshot = '%s/%s/0/snapdir_%03d/snap_%03d'%(root,cosmo,snapnum,snapnum)
            delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
            delta = delta/np.mean(delta, dtype=np.float64)
            delta_min, delta_max = np.min(delta), np.max(delta)

            # define the pdf bins
            pdf_bins  = np.linspace(delta_min, delta_max, bins+1)
            pdf_mean  = 0.5*(pdf_bins[1:] + pdf_bins[:-1])
            pdf_width = pdf_bins[1:] - pdf_bins[:-1]

            # do a loop over all snapshots
            for i in numbers:

                print(i)
                # get the name of the snapshot and the output file
                snapshot = '%s/%s/%d/snapdir_%03d/snap_%03d'%(root,cosmo,i,snapnum,snapnum)

                # compute density field
                delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
                delta = delta/np.mean(delta, dtype=np.float64)

                # compute pdf and variance
                var[i] = np.var(delta)
                pdf[i] = np.histogram(delta,bins=pdf_bins)[0]
                pdf[i] = pdf[i]/pdf_width/np.sum(pdf[i], dtype=np.float64)

            # combine all measurements 
            comm.Reduce(var, var_tot, root=0)
            comm.Reduce(pdf, pdf_tot, root=0)

            # only master save results to file
            if myrank==0:
                f = h5py.File(fout, 'w')
                f.create_dataset('variance', data=var_tot)
                f.create_dataset('pdf',      data=pdf_tot)
                f.create_dataset('pdf_bins', data=pdf_mean)
                f.close()

            # wait until master is done writting the results
            comm.Barrier()

This repository contains the codes and results of the neutrino PDF project

The main code (mpi parallelized) used to compute the neutrino pdf for the different Quijote simulations, at different redshifts, and for different grid sizes is in Codes/pdf_nu.py.

The results are stored in the Results folder under a generic name like:

Results_Mnu_grid_z=redshift.hdf5

Where Mnu is the neutrino mass, grid is the size of the grid used to compute the neutrino density field and redshift is the simulation redshift. Those are hdf5 files that can be read in python as this:

```python
import numpy as np
import hdf5

f = h5py.File('Results_0.1eV_500_z=0.hdf5', 'r')
var = f['variance'][:] #variance of the neutrino density field for all simulations
pdf = f['pdf'][:]      #pdf of all simulations
f.close()
```

For instance, the pdf and the variance of the realization 13 can be obtained as ```pdf[13]``` and ```var[13]```, respectively.
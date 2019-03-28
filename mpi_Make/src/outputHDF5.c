#include "outputHDF5.h"

void output_hdf5(const int nDims, const int nrow, const int ncol)
{
    // Implementing HDF5 output format
    hid_t   group_id;         /* group identifier */
    hid_t   file_id, dset_id; /* file and dataset identifier  */
    hid_t   dspace_id;        /* data space identifier */
    hid_t   plist_id;         /* property list identifier */
    hsize_t dimsf[ nDims ];   /* dataset diemsions */
    herr_t __attribute__((unused)) status;
    dimsf[ 0 ] = NX;
    dimsf[ 1 ] = NY;

    /*    // Set up file for parallel I/O access
         plist_id = H5Pcreate(H5P_FILE_ACCESS);
             H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
             */
    // Create new file and release property list identifier
    // file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // H5Pclose(plist_id);

    dspace_id = H5Screate_simple(nDims, dimsf, NULL);
    dset_id   = H5Dcreate2(file_id, "/TSet", H5T_IEEE_F64LE, dspace_id, H5P_DEFAULT, H5P_DEFAULT,
                         H5P_DEFAULT);

    // Closing HDF5 Files
    status = H5Dclose(dset_id);
    status = H5Sclose(dspace_id);
    status = H5Fclose(file_id);
}

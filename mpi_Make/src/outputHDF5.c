#include "outputHDF5.h"

static const char *Scalnames[ HDF5_NUM_SCALARS ] = {"Temperature"};

void output_hdf5(const int nDims, const int nrow, const int ncol, const int nGhostLayers, const REAL *phi)
{
    // Implementing HDF5 output format
    hid_t   group_id;            /* group identifier */
    hid_t   file_id, dset_id;    /* file and dataset identifier  */
    hid_t   filespace, memspace; /* file and memory dataspace identifiers */
    hid_t   plist_id;            /* property list identifier */
    hsize_t dimsf[ nDims ];      /* dataset diemsions */
    hsize_t count[ nDims ];
    herr_t __attribute__((unused)) status;

    // Creating set for X, Y, Z coordinate arrays
    dimsf[ 0 ] = ncol;
    dimsf[ 1 ] = nrow;

    // Set up file for parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Create new file and release property list identifier
    file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id); // For Parallel I/O
    H5Pclose(plist_id);
    group_id = H5Gcreate(file_id, "MPI2DTemp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

#if 1 // Writing Coordinates
    REAL *  x_coords = ( REAL * ) malloc(ncol * sizeof(REAL));
    REAL *  y_coords = ( REAL * ) malloc(nrow * sizeof(REAL));

    hsize_t k;
    for (k = 0; k < dimsf[ 0 ]; k++) {
        x_coords[ k ] = k * DX;
    }
    for (k = 0; k < dimsf[ 1 ]; k++) {
        y_coords[ k ] = k * DY;
    }

    hsize_t x_size = dimsf[ 0 ];
    hsize_t y_size = dimsf[ 1 ];

    printf("Writing Coordinates\n");
    char* names[] = {"X","Y"};

    // Dataset Write Section
    for (int coords = 0; coords < nDims; coords++) {
        switch (coords) {
            case 0:
                filespace = H5Screate_simple(1, &x_size, NULL);
                break;
            case 1:
                filespace = H5Screate_simple(1, &y_size, NULL);
                break;
            default:
                filespace = 0;
        }

        plist_id = H5Pcreate(H5P_DATASET_CREATE);
        dset_id  = H5Dcreate(group_id, names[ coords ], H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT,
                            plist_id, H5P_DEFAULT);
        H5Pclose(plist_id);

        switch (coords) {
            case 0:
                memspace = H5Screate_simple(1, &x_size, NULL);
                break;
            case 1:
                memspace = H5Screate_simple(1, &y_size, NULL);
                break;
            default:
                memspace = 0;
        }

        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        if (filespace != 0 && memspace != 0) {
            if (coords == 0)
                status
                = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, x_coords);
            if (coords == 1)
                status
                = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, y_coords);
        }
        H5Dclose(dset_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
    }
#endif

#if 1 // Writing Scalar Values
    printf("Writing Scalars: %p\n", phi);

    for (int i = 0; i < HDF5_NUM_SCALARS; i++) {
        filespace = H5Screate_simple(2, dimsf, NULL);

        plist_id = H5Pcreate(H5P_DATASET_CREATE);
        dset_id  = H5Dcreate(group_id, Scalnames[ i ], H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT,
                            plist_id, H5P_DEFAULT);
        H5Sclose(filespace);
        H5Pclose(plist_id);

        memspace = H5Screate_simple(2, dimsf, NULL);

        filespace = H5Dget_space(dset_id);
        plist_id  = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, phi);

        H5Dclose(dset_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
    }
#endif

#if 1
    free(x_coords);
    free(y_coords);
#endif

    printf("Finished HDF5\n");
}

void dump_paraview_xdmf(const REAL* phi)
{

}

#include "outputHDF5.h"

static const char *Scalnames[ HDF5_NUM_SCALARS ] = {"Temperature"};

static void remove_ghost_scalar(const REAL *phi, hsize_t *dimsf, const int nrow, const int ncol,
                                const int nGhostLayers, REAL *target)
{
    for (int j = 1; j < nrow + 1; j++) {
        for (int i = 1; i < ncol + 1; i++) {
            int targetIC       = (i - 1) + ncol * (j - 1);
            int phiIC          = i + (ncol + nGhostLayers) * j;
            target[ targetIC ] = phi[ phiIC ];
            printf("%6.2f ", target[ targetIC ]);
        }
        printf("\n");
    }
}

void output_hdf5(const int nDims, const int nrow, const int ncol, const int nGhostLayers,
                 const REAL *phi)
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
    dimsf[ X ] = NX;
    dimsf[ Y ] = NY;

    // Set up file for parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Create new file and release property list identifier
    file_id = H5Fcreate(FILEhdf5, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id); // For Parallel I/O
    H5Pclose(plist_id);
    group_id = H5Gcreate(file_id, "MPI2DTemp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Writing Coordinates
    REAL *x_coords = ( REAL * ) malloc(dimsf[ 0 ] * sizeof(REAL));
    REAL *y_coords = ( REAL * ) malloc(dimsf[ 1 ] * sizeof(REAL));

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
    char *names[] = {"X", "Y"};

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
            if (coords == X)
                status
                = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, x_coords);
            if (coords == Y)
                status
                = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, y_coords);
        }
        H5Dclose(dset_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
    }

    // Removing Ghost Layers
    REAL *target = ( REAL * ) malloc(dimsf[ X ] * dimsf[ Y ] * sizeof(REAL));
    remove_ghost_scalar(phi, dimsf, nrow, ncol, nGhostLayers, target);

#if 1 // Writing Scalar Values
    printf("Writing Scalars: %p\n", target);

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

        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, target);

        H5Dclose(dset_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
    }
#endif

    printf("Finished HDF5\n");

    // Freeing Variables
    free(x_coords);
    free(y_coords);
    free(target);
}

void dump_paraview_xdmf(const REAL *phi, double time)
{
    FILE *fp     = FILExdmf;
    char *output = "MPI2DTemp";

    fopen(fp, "w");

    printf("Starting XDMF (Visit) FILE* = %p output = |%s| time = %f (ms)", fp, output, time*1e3);

    int precision = (sizeof(REAL) == sizeof(double)) ? 8 : 4;

    fprintf(fp, "      <Grid Name=\"mesh1\" GridType=\"Uniform\">\n");
    fprintf(fp, "        <Time Type=\"Single\" Value=\"%.*f\" />\n", precision, time);
    fprintf(fp, "          <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"%d %d\"/>\n\n", NX, NY);
    fprintf(fp, "          <Geometry GeometryType=\"XY\">\n");
    fprintf(fp, "            <DataItem Dimensions=\"%d\" NumberType=\"Float\" Precision=\"%d\" Format=\"HDF\">\n", NX, precision);
    fprintf(fp, "              %s:/%s/X\n", fp, output);
    fprintf(fp, "            </DataItem>\n");
    fprintf(fp, "            <DataItem Dimensions=\"%d\" NumberType=\"Float\" Precision=\"%d\" Format=\"HDF\">\n", NY, precision);
    fprintf(fp, "              %s:/%s/Y\n", fp, output);
    fprintf(fp, "            </DataItem>\n");

#if 0
   for(int i = 0; i < HDF5_NUM_SCALARS; i++)
   {
      if (Scalnames[i]) {
         fprintf(fp,"          <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", Scalnames[i]);
         fprintf(fp,"            <DataItem Dimensions=\"%d %d\" NumberType=\"Float\" Precision=\"%d\" Format=\"HDF\">\n", NX, NY, precision);
         fprintf(fp,"              %s:/%s/%s\n", fp, output, Scalnames[i]);
         fprintf(fp,"            </DataItem>\n");
         fprintf(fp,"          </Attribute>\n\n");
      }
   }
#endif

    fprintf(fp, "      </Grid>\n\n");

    fclose(fp);
    printf("Finished XDMF (Paraview)\n");
}

#if (OUTPUT)
void outputMatrix(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, char *name)
{
    FILE *file = fopen(name, "w");
    for (INT j = 0; j < (nrow + nGhostLayers); j++) {
        for (INT i = 0; i < (ncol + nGhostLayers); i++) {
            fprintf(file, "%6.2f ", phi[ IC ]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

void outputMatrix1(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, char *name)
{
    FILE *file = fopen(name, "w");
    for (INT j = 1; j < nrow + 1; j++) {
        for (INT i = 1; i < ncol + 1; i++) {
            fprintf(file, "%6.2f ", phi[ IC ]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}
#endif

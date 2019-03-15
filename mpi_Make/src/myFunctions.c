#include "myFunctions.h"

void boundaryConditions(REAL *phi, const INT ncol, const INT nrow, const int nGhostLayers,
                        const int *direction)
{
    if (direction[ N ] < 0) {
        for (INT j = 1; j < 2; j++) {
            for (INT i = 0; i < (ncol + nGhostLayers); i++) {
                phi[ IC ] = TMAX;
            }
        }
    }
    if (direction[ S ] < 0) {
        for (INT j = nrow; j < (nrow + 1); j++) {
            for (INT i = 0; i < ncol + nGhostLayers; i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
    if (direction[ W ] < 0) {
        for (INT j = 0; j < (nrow + nGhostLayers); j++) {
            for (INT i = 0; i < 2; i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
    if (direction[ E ] < 0) {
        for (INT j = 0; j < (nrow + nGhostLayers); j++) {
            for (INT i = ncol; i < (ncol + 2); i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
}

void decomposeMesh_2D(INT *coord_2D, INT *direction, INT *location, INT *p_location, INT *nrow,
                      INT *ncol)
{
    // Determining Cartesian Start and End of XY direction in location[]
    location[ W ] = coord_2D[ X ] * NX / IDIM;
    location[ E ] = location[ W ] + NX / IDIM;
    *ncol         = location[ E ] - location[ W ];
    location[ S ] = coord_2D[ Y ] * NY / JDIM;
    location[ N ] = location[ S ] + NY / JDIM;
    *nrow         = location[ N ] - location[ S ];

    // Determing Process Start and End location of XY direction in p_location[]
    if (direction[ N ] < 0) {
        p_location[ N ] = 2;
    } else {
        p_location[ N ] = 1;
    }
    if (direction[ S ] < 0) {
        p_location[ S ] = *nrow;
    } else {
        p_location[ S ] = *nrow + 1;
    }
    if (direction[ W ] < 0) {
        p_location[ W ] = 2;
    } else {
        p_location[ W ] = 1;
    }
    if (direction[ E ] < 0) {
        p_location[ E ] = *ncol;
    } else {
        p_location[ E ] = *ncol + 1;
    }
}

void exchange_ISend_and_IRecv(REAL *phi, const int *direction, const INT ncol, const INT nrow,
                              const int nGhostLayers)
{
    MPI_Request request[ 8 ];

    int tag0      = 0; // North tag
    int tag1      = 1; // South tag
    int tag2      = 2; // East tag
    int tag3      = 3; // West tag
    int nElements = ncol + nGhostLayers;

    // Creating Columns from phi Matrix
    MPI_Datatype MPI_column_W, MPI_column_E;
    MPI_Type_vector(nrow + 2, 1, nElements, MPI_DOUBLE, &MPI_column_W);
    MPI_Type_vector(nrow + 2, 1, nElements, MPI_DOUBLE, &MPI_column_E);
    MPI_Type_commit(&MPI_column_W);
    MPI_Type_commit(&MPI_column_E);

    // North
    MPI_Isend(phi + nElements, nElements, MPI_DOUBLE, direction[ N ], tag0, MPI_COMM_WORLD,
              &request[ 0 ]);
    MPI_Irecv(phi + ((nrow + 1) * nElements), nElements, MPI_DOUBLE, direction[ S ], tag0,
              MPI_COMM_WORLD, &request[ 1 ]);

    // South
    MPI_Isend(phi + (nrow * nElements), nElements, MPI_DOUBLE, direction[ S ], tag1, MPI_COMM_WORLD,
              &request[ 2 ]);
    MPI_Irecv(phi, nElements, MPI_DOUBLE, direction[ N ], tag1, MPI_COMM_WORLD, &request[ 3 ]);

    // East
    MPI_Isend(phi + (ncol + 1), 1, MPI_column_E, direction[ E ], tag2, MPI_COMM_WORLD,
              &request[ 4 ]);
    MPI_Irecv(phi, 1, MPI_column_E, direction[ W ], tag2, MPI_COMM_WORLD, &request[ 5 ]);

    // West
    MPI_Isend(phi + 1, 1, MPI_column_W, direction[ W ], tag3, MPI_COMM_WORLD, &request[ 6 ]);
    MPI_Irecv(phi + (ncol + 1), 1, MPI_column_W, direction[ E ], tag3, MPI_COMM_WORLD,
              &request[ 7 ]);

    MPI_Waitall(8, request, MPI_STATUS_IGNORE);
    MPI_Type_free(&MPI_column_W);
    MPI_Type_free(&MPI_column_E);
}

void SolveHeatEQ(const REAL *phi, REAL *phi_new, const int *p_location, const INT ncol,
                 const INT nrow, const INT nGhostLayers)
{
    for (INT j = p_location[ N ]; j < p_location[ S ]; j++) {
        for (INT i = p_location[ W ]; i < p_location[ E ]; i++) {
            phi_new[ IC ] = (((phi[ IP1 ] - 2.0f * phi[ IC ] + phi[ IM1 ]) / (DX * DX))
                             + ((phi[ JP1 ] - 2.0f * phi[ IC ] + phi[ JM1 ])) / (DY * DY))
                            * DT
                            + phi[ IC ];
        }
    }
}

void setAlltoValue(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, REAL value)
{
    for (INT j = 0; j < nrow + nGhostLayers; j++) {
        for (INT i = 0; i < ncol + nGhostLayers; i++) {
            phi[ IC ] = value;
        }
    }
}

#if(OUTPUT)
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

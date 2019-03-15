/*
 * Parallelizing 2D Heat Equations solver using 5 points equations
 *
 * Author: Dustin (Ting-Hsuan) Ma
 *
 * To Compile: mpicc -o MPI.exe -std=c99 -O3 -Wall -lm mpiHeat.c
 * To Run: mpirun -np 4 ./MPI.exe
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef double REAL;
typedef int    INT;

// Parameters
#define IDIM ( INT ) 6
#define JDIM ( INT ) 6 
#define NX ( INT ) IDIM * 10000
#define NY ( INT ) JDIM * 10000

// 2D-Decomposition Terms
#define X ( INT ) 0
#define Y ( INT ) 1
#define FALSE ( INT ) 0
#define TURE ( INT ) 1
#define N ( INT ) 0
#define S ( INT ) 1
#define E ( INT ) 2
#define W ( INT ) 3

// Spacial
#define LX ( REAL ) 20.0f
#define LY ( REAL ) 20.0f
#define DX LX / (( REAL ) NX - 1.0f)
#define DY LY / (( REAL ) NY - 1.0f)

// Temperature
#define TMAX ( REAL ) 100.0f
#define TMIN ( REAL ) 0.0f

// Time
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define DT ( REAL ) 0.25f * MIN(DX, DY) * MIN(DX, DY)
#define MAXITER LX *LX

// Calculation index
#define IC i + j *(ncol + nGhostLayers)
#define IP1 (i + 1) + j *(ncol + nGhostLayers)
#define IM1 (i - 1) + j *(ncol + nGhostLayers)
#define JP1 i + (j + 1) * (ncol + nGhostLayers)
#define JM1 i + (j - 1) * (ncol + nGhostLayers)

// Process
#define MASTER 0

// Debug
#define DEBUG 0

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
    MPI_Isend(phi + (nrow * nElements), nElements, MPI_DOUBLE, direction[ S ], tag1,
              MPI_COMM_WORLD, &request[ 2 ]);
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

void SolveHeatEQ(const REAL *phi, REAL *phi_new, const int *p_location, const INT ncol, const INT nrow, const INT nGhostLayers)
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


INT main(INT argc, char **argv)
{
    INT nProcs;     // number of processes
    INT myRank;     // process rank
    INT nrow, ncol; // number of row needed to be allocated by local array
    INT direction[ 4 ], location[ 4 ], p_location[ 4 ];
    INT nDims   = 2; // dimension of Cartesian decomposition X=0, Y=1
    INT reorder = 1; // allow system to optimize(reorder) the mapping of processes to physical cores
    INT nGhostLayers = 2;

    MPI_Init(&argc, &argv);                 // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // get the number of processes

    // Defining size and determining values for MPI topology
    INT dimension[ nDims ], periodic[ nDims ], coord_2D[ nDims ];
    dimension[ X ] = IDIM;  // X dimension size
    dimension[ Y ] = JDIM;  // Y dimension size
    periodic[ X ]  = FALSE; // X periodicity
    periodic[ Y ]  = FALSE; // Y Periodicity

    MPI_Comm comm2D;
    MPI_Cart_create(MPI_COMM_WORLD, nDims, dimension, periodic, reorder, &comm2D);
    MPI_Comm_rank(comm2D, &myRank);
    MPI_Cart_coords(comm2D, myRank, nDims, coord_2D);
    MPI_Cart_shift(comm2D, Y, +1, &direction[ S ], &direction[ N ]); // North and South
    MPI_Cart_shift(comm2D, X, +1, &direction[ W ], &direction[ E ]); // West and East

    // Mesh Decompotistion
    decomposeMesh_2D(coord_2D, direction, location, p_location, &nrow, &ncol);

    printf("myRank=%d, N=%2.1d, S=%2.1d, E=%2.1d, W=%2.1d, ncol=%d, nrow=%d, Coord=<%d, %d>, "
           "X~[%d,%d], Y~[%d,%d], "
           "pX=[%d,%d], pY=[%d,%d] \n",
           myRank, direction[ N ], direction[ S ], direction[ E ], direction[ W ], ncol, nrow,
           coord_2D[ X ], coord_2D[ Y ], location[ W ], location[ E ], location[ S ], location[ N ],
           p_location[ W ], p_location[ E ], p_location[ N ], p_location[ S ]);

    // Allocating Memory for every process after mesh decomposition
    REAL *phi, *phi_new, *tmp;
    phi = ( REAL * ) calloc((nrow + nGhostLayers) * (ncol + nGhostLayers), sizeof(*phi));
    phi_new = ( REAL * ) calloc((nrow + nGhostLayers) * (ncol + nGhostLayers), sizeof(*phi_new));

//    setAlltoValue(phi, nrow, ncol, nGhostLayers, ( REAL ) myRank + 1.0);
    boundaryConditions(phi, ncol, nrow, nGhostLayers, direction);

    MPI_Barrier(MPI_COMM_WORLD);
    double startT = MPI_Wtime( );
//    for (REAL iter = 0.f; iter < MAXITER; iter += DT) {
    for (int iter = 0; iter < 10; iter++) {
        exchange_ISend_and_IRecv(phi, direction, ncol, nrow, nGhostLayers);
        SolveHeatEQ(phi, phi_new, p_location, ncol, nrow, nGhostLayers);
        boundaryConditions(phi_new, ncol, nrow, nGhostLayers, direction);
        tmp     = phi;
        phi     = phi_new;
        phi_new = tmp;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double finishT = MPI_Wtime( );

    // Barrier before recording the finish time
    double elapsedTime = finishT - startT;
    double wallTime;
    MPI_Reduce(&elapsedTime, &wallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == MASTER) {
         printf("Wall-clock time = %.3f (ms) \n", wallTime * 1e3);
    }

    // Deallocating Arrays
    free(phi);
    phi = NULL;

    MPI_Finalize( );
    return EXIT_SUCCESS;
}

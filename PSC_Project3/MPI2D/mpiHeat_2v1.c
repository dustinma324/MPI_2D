/*
 * Parallelizing 2D Heat Equations solver using 5 points equations
 *
 * Author: Dustin (Ting-Hsuan) Ma
 *
 * To Compile: mpicc -o MPI.exe -lm mpiHeat.c
 * To Run: mpirun -np 4 ./MPI.exe
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef double REAL;
typedef int    INT;

// Parameters
#define IDIM ( INT ) 2
#define JDIM ( INT ) 2
#define NX ( INT ) 10
#define NY ( INT ) 10

// 2D-Decomposition Terms
#define X ( INT ) 0
#define Y ( INT ) 1
#define FALSE ( INT ) 0
#define TURE ( INT ) 1
#define NORTH ( INT ) 0
#define SOUTH ( INT ) 1
#define EAST ( INT ) 2
#define WEST ( INT ) 3

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
#define IC i + j *NX
#define IP1 (i + 1) + j *NX
#define IM1 (i - 1) + j *NX
#define JP1 i + (j + 1) * NX
#define JM1 i + (j - 1) * NX

// Process
#define MASTER 0

// Debug
#define DEBUG 0
#define DEVELOPE 1

void boundaryConditions(REAL *phi, const int nrow, const int ncol, const int nGhostLayers,
                        const int *direction)
{
    if (direction[ NORTH ] == -1) {
        for (INT j = 1; j < 2; j++) {
            for (INT i = 0; i < (ncol + nGhostLayers); i++) {
                phi[ IC ] = TMAX;
            }
        }
    }
    if (direction[ SOUTH ] == -1) {
        for (INT j = nrow; j < (nrow + 1); j++) {
            for (INT i = 0; i < ncol + nGhostLayers; i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
    if (direction[ WEST ] == -1) {
        for (INT j = 0; j < (nrow + nGhostLayers); j++) {
            for (INT i = 1; i < 2; i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
    if (direction[ EAST ] == -1) {
        for (INT j = 0; j < (nrow + nGhostLayers); j++) {
            for (INT i = ncol; i < (ncol + 1); i++) {
                phi[ IC ] = TMIN;
            }
        }
    }
}

void decomposeMesh_2D(const int *coord_2D, const int *direction, int *location, int *p_location,
                      int *nrow, int *ncol)
{
    // Determining Cartesian Start and End of XY direction in location[]
    location[ WEST ]  = coord_2D[ X ] * NX / IDIM;
    location[ EAST ]  = location[ WEST ] + NX / IDIM;
    *nrow             = location[ EAST ] - location[ WEST ];
    location[ SOUTH ] = coord_2D[ Y ] * NY / JDIM;
    location[ NORTH ] = location[ SOUTH ] + NY / JDIM;
    *ncol             = location[ NORTH ] - location[ SOUTH ];

    // Determing Process Start and End location of XY direction in p_location[]
    if (direction[ NORTH ] == -1) {
        p_location[ NORTH ] = 2;
    } else {
        p_location[ NORTH ] = 1;
    }
    if (direction[ SOUTH ] == -1) {
        p_location[ SOUTH ] = *nrow + 1;
    } else {
        p_location[ SOUTH ] = *nrow + 2;
    }
    if (direction[ WEST ] == -1) {
        p_location[ WEST ] = 2;
    } else {
        p_location[ WEST ] = 1;
    }
    if (direction[ EAST ] == -1) {
        p_location[ EAST ] = *ncol + 1;
    } else {
        p_location[ EAST ] = *ncol + 2;
    }
}

void SolveHeatEQ(const REAL *phi, REAL *phi_new, const int *p_location)
{
    for (INT j = p_location[ NORTH ]; j <= p_location[ SOUTH ]; j++) {
        for (INT i = p_location[ WEST ]; i <= p_location[ EAST ]; i++) {
            phi_new[ IC ] = (((phi[ IP1 ] - 2.0f * phi[ IC ] + phi[ IM1 ]) / (DX * DX))
                             + ((phi[ JP1 ] - 2.0f * phi[ IC ] + phi[ JM1 ])) / (DY * DY))
                            * DT
                            + phi[ IC ];
        }
    }
}

void exchange_ISend_and_IRecv(REAL *phi, const int *direction, const int nrow, const int ncol,
                              const int nGhostLayers)
{
    MPI_Request request[ 2 ];
    MPI_Status  stats[ 2 ];

    int tag0      = 0; // North tag
    int tag1      = 1; // South tag
    int tag2      = 2; // East tag
    int tag3      = 3; // West tag
    int nElements = ncol + nGhostLayers;

/*    // Creating Columns from phi Matrix
    MPI_Datatype MPI_column_W, MPI_column_E;
    MPI_Type_vector(nrow + 2, 1, nElements, MPI_DOUBLE, &MPI_column_W);
    MPI_Type_vector(nrow + 2, 1, nElements, MPI_DOUBLE, &MPI_column_E);
    MPI_Type_commit(&MPI_column_W);
    MPI_Type_commit(&MPI_column_E);
*/
    // North
//    MPI_Isend(phi + nElements, nElements, MPI_DOUBLE, direction[ NORTH ], tag0, MPI_COMM_WORLD, &request[ 0 ]);
//    MPI_Irecv(phi + ((nrow+1) * nElements), nElements, MPI_DOUBLE, direction[ SOUTH ], tag0, MPI_COMM_WORLD,
//              &request[ 1 ]);
/*
    // South
    MPI_Isend(phi + ((nrow + 1) * nElements), nElements, MPI_DOUBLE, direction[ SOUTH ], tag1,
              MPI_COMM_WORLD, &request[ 2 ]);
    MPI_Irecv(phi, nElements, MPI_DOUBLE, direction[ NORTH ], tag1, MPI_COMM_WORLD, &request[ 3 ]);
*/
    /*    // East
        MPI_Isend(phi + (nElements - 1), 1, MPI_column_E, direction[ EAST ], tag2, MPI_COMM_WORLD,
                  &request[ 4 ]);
        MPI_Irecv(phi, 1, MPI_column_E, direction[ WEST ], tag2, MPI_COMM_WORLD, &request[ 5 ]);

        // West
        MPI_Isend(phi + 1, 1, MPI_column_W, direction[ WEST ], tag3, MPI_COMM_WORLD, &request[ 6 ]);
        MPI_Irecv(phi + (nrow + 1), 1, MPI_column_W, direction[ EAST ], tag3, MPI_COMM_WORLD,
                  &request[ 7 ]);
    */
//    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
/*    MPI_Type_free(&MPI_column_W);
    MPI_Type_free(&MPI_column_E);
*/
}

void outputMatrix(const REAL *phi, const int nrow, const int ncol, char *name)
{
    FILE *file = fopen(name, "w");
    for (INT j = 0; j < (nrow + 2); j++) {
        for (INT i = 0; i < (ncol + 2); i++) {
            fprintf(file, "%6.2f ", phi[ IC ]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

#if (DEVELOPE)
void setAlltoValue(REAL *in, const int nrow, const int ncol, const int nGhostLayers,
                   const REAL value)
{
    for (INT j = 0; j < (nrow + nGhostLayers); j++) {
        for (INT i = 0; i < (ncol + nGhostLayers); i++) {
            //in[ IC ] = value;
            in[IC] = (REAL)j*((REAL)ncol+2.f) + (REAL)i;
        }
    }
}
#endif

int main(int argc, char **argv)
{
    int nProcs;     // number of processes
    int myRank;     // process rank
    int nrow, ncol; // number of row needed to be allocated by local array
    int direction[ 4 ], location[ 4 ], p_location[ 4 ];
    int nDims   = 2; // dimension of Cartesian decomposition X=0, Y=1
    int reorder = 1; // allow system to optimize(reorder) the mapping of processes to physical cores
    int nGhostLayers = 2;

    MPI_Init(&argc, &argv);                 // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // get the number of processes

    // Defining size and determining values for MPI topology
    int dimension[ nDims ], periodic[ nDims ], coord_2D[ nDims ];
    dimension[ X ] = IDIM;  // X dimension size
    dimension[ Y ] = JDIM;  // Y dimension size
    periodic[ X ]  = FALSE; // X periodicity
    periodic[ Y ]  = FALSE; // Y Periodicity

    MPI_Comm comm2D;
    MPI_Cart_create(MPI_COMM_WORLD, nDims, dimension, periodic, reorder, &comm2D);
    MPI_Comm_rank(comm2D, &myRank);
    MPI_Cart_coords(comm2D, myRank, nDims, coord_2D);
    MPI_Cart_shift(comm2D, Y, +1, &direction[ SOUTH ], &direction[ NORTH ]); // North and South
    MPI_Cart_shift(comm2D, X, +1, &direction[ WEST ], &direction[ EAST ]);   // West and East

    // Mesh Decompotistion
    decomposeMesh_2D(coord_2D, direction, location, p_location, &nrow, &ncol);

#if (DEVELOPE)
    printf("myRank=%d, N=%2.1d, S=%2.1d, E=%2.1d, W=%2.1d, nrow=%d, ncol=%d, Coord=<%d, %d>, X~[%d,%d], Y~[%d,%d], "
           "pX=[%d,%d], pY=[%d,%d] \n",
           myRank, direction[ NORTH ], direction[ SOUTH ], direction[ EAST ], direction[ WEST ], nrow, ncol,
           coord_2D[ X ], coord_2D[ Y ], location[ WEST ], location[ EAST ], location[ SOUTH ],
           location[ NORTH ], p_location[ WEST ], p_location[ EAST ], p_location[ NORTH ],
           p_location[ SOUTH ]);
#endif

    // Allocating Memory for every process after mesh decomposition
    REAL *phi, *phi_new, *tmp, *phi_global;
    phi     = ( REAL * ) calloc((nrow+2)*(ncol+2), sizeof(*phi));
    phi_new = ( REAL * ) calloc((nrow+2)*(ncol+2), sizeof(*phi_new));

#if (DEVELOPE)
    if (myRank == 0) setAlltoValue(phi, nrow, ncol, nGhostLayers, 11);
    if (myRank == 1) setAlltoValue(phi, nrow, ncol, nGhostLayers, 22);
    if (myRank == 2) setAlltoValue(phi, nrow, ncol, nGhostLayers, 33);
    if (myRank == 3) setAlltoValue(phi, nrow, ncol, nGhostLayers, 44);
#endif

    // Initializing Problem
//    boundaryConditions(phi, nrow, ncol, nGhostLayers, direction);

    /*
        if (myRank == MASTER) {
            phi_global = ( REAL * ) malloc(NX * NY, sizeof(*phi_global));
        }
    */

#if (DEBUG)
    // Performing calculation and timing for scalibility
    MPI_Barrier(MPI_COMM_WORLD);
    double startT = MPI_Wtime( );
    for (REAL iter = 0.f; iter < MAXITER; iter += DT) {
        exchange_ISend_and_IRecv(phi, direction, nrow, ncol, nGhostLayers);
        SolveHeatEQ(phi, phi_new, p_location);
        boundaryConditions(phi_new, nrow, ncol, nGhostLayers, direction);
        tmp     = phi;
        phi     = phi_new;
        phi_new = tmp;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double finishT = MPI_Wtime( );
#endif

#if (DEVELOPE)
    exchange_ISend_and_IRecv(phi, direction, nrow, ncol, nGhostLayers);
    if (myRank == 0) outputMatrix(phi, nrow, ncol, "Process0.txt");
    if (myRank == 1) outputMatrix(phi, nrow, ncol, "Process1.txt");
    if (myRank == 2) outputMatrix(phi, nrow, ncol, "Process2.txt");
    if (myRank == 3) outputMatrix(phi, nrow, ncol, "Process3.txt");
#endif

        /*
            if (myRank == MASTER) {
                printf("***********FINAL OUTPUT AFTER GATHER***********\n");
                outputMatrix(phi_global, NX, NY, "Temperature.csv");

                free(phi_global);
                phi_global = NULL;
            }
        */

#if (DEBUG)
    // Barrier before recording the finish time
    double elapsedTime = finishT - startT;
    double wallTime;
    MPI_Reduce(&elapsedTime, &wallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == MASTER) {
        printf("Wall-clock time = %.3f (ms) \n", wallTime * 1e3);
    }
#endif

    // Deallocating Arrays
    free(phi);
    free(phi_new);
    phi     = NULL;
    phi_new = NULL;

    MPI_Finalize( );
    return EXIT_SUCCESS;
}

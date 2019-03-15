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

// Spacial
#define LX ( REAL ) 20.0f
#define LY ( REAL ) 20.0f
#define NX ( INT ) 10000
#define NY ( INT ) 10000
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
#define DEBUG 1

void boundaryConditions(REAL *phi, const int nrow, const int nGhostLayers, const int *direction)
{
    if (direction[ 0 ] == -1) {
        for (INT j = 0; j < nrow + nGhostLayers; j++) {
            for (INT i = 0; i < NX; i++) {
                if (j == 1) phi[ IC ] = TMAX;
                if (i == 0) phi[ IC ] = TMIN;
                if (i == NX - 1) phi[ IC ] = TMIN;
            }
        }
    }
    if (direction[ 1 ] == -1) {
        for (INT j = 0; j < nrow + nGhostLayers; j++) {
            for (INT i = 0; i < NX; i++) {
                if (j == nrow + 1) phi[ IC ] = TMIN;
                if (i == 0) phi[ IC ] = TMIN;
                if (i == NX - 1) phi[ IC ] = TMIN;
            }
        }
    } else {
        for (INT j = 0; j < nrow + nGhostLayers; j++) {
            for (INT i = 0; i < NX; i++) {
                if (i == 0) phi[ IC ] = TMIN;
                if (i == NX - 1) phi[ IC ] = TMIN;
            }
        }
    }
}

void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *location, int *nrow,
                      int *p_location)
{
    location[ 0 ] = myRank * N / nProcs;
    location[ 1 ] = location[ 0 ] + N / nProcs;
    *nrow         = location[ 1 ] - location[ 0 ];

    /* For serial case */
    if (nProcs == 1) {
        p_location[ 0 ] = 2;
        p_location[ 1 ] = *nrow - 1;
    } else {
        /* For parallel processes */
        if (myRank == MASTER) {
            p_location[ 0 ] = 2;
        } else {
            p_location[ 0 ] = 1;
        }
        if (myRank == nProcs - 1) {
            p_location[ 1 ] = *nrow - 1;
        } else {
            p_location[ 1 ] = *nrow + 1;
        }
    }
}

void SolveHeatEQ(const REAL *phi, REAL *phi_new, const int *p_location)
{
    for (INT j = p_location[ 0 ]; j <= p_location[ 1 ]; j++) {
        for (INT i = 1; i < NX - 1; i++) {
            phi_new[ IC ] = (((phi[ IP1 ] - 2.0f * phi[ IC ] + phi[ IM1 ]) / (DX * DX))
                             + ((phi[ JP1 ] - 2.0f * phi[ IC ] + phi[ JM1 ])) / (DY * DY))
                            * DT
                            + phi[ IC ];
        }
    }
}

void exchange_SendRecv(REAL *in, const int src, const int dest, const int nrow, const int myRank)
{
    int tag0 = 0; // send tag
    int tag1 = 1; // send tag
    MPI_Sendrecv(in + (nrow * NX), NX, MPI_DOUBLE, dest, tag0, in, NX, MPI_DOUBLE, src, tag0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Sending down
    MPI_Sendrecv(in + NX, NX, MPI_DOUBLE, src, tag1, in + (nrow + 1) * NX, NX, MPI_DOUBLE, dest,
                 tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Sending up
}

void exchange_ISend_and_IRecv(REAL *in, const int *direction, const int nrow, const int myRank)
{
    MPI_Request request[ 4 ];
    MPI_Status  stats[ 4 ];

    int tag0 = 0; // up send tag
    int tag1 = 1; // down send tag
    MPI_Isend(in + (nrow * NX), NX, MPI_DOUBLE, direction[ 1 ], tag1, MPI_COMM_WORLD,
              &request[ 0 ]); // send to down
    MPI_Isend(in + NX, NX, MPI_DOUBLE, direction[ 0 ], tag0, MPI_COMM_WORLD,
              &request[ 1 ]); // send to up

    MPI_Irecv(in, NX, MPI_DOUBLE, direction[ 0 ], tag1, MPI_COMM_WORLD,
              &request[ 2 ]); // receive from up
    MPI_Irecv(in + ((nrow + 1) * NX), NX, MPI_DOUBLE, direction[ 1 ], tag0, MPI_COMM_WORLD,
              &request[ 3 ]); // receive from down

    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
}

void outputMatrix(const REAL *phi, char *name)
{
    FILE *file = fopen(name, "w");
    for (INT j = 0; j < NY; j++) {
        for (INT i = 0; i < NX; i++) {
            fprintf(file, "%6.2f ", phi[ IC ]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

int main(int argc, char **argv)
{
    int nProcs;          // number of processes
    int myRank;          // process rank
    int nrow;            // number of row needed to be allocated by local array
    int direction[ 2 ];  // 0 = up ,1 = down, 2 = left, 3 = right
    int location[ 2 ];   // 0 = start, 1 = end
    int p_location[ 2 ]; // 0 = startLocal, 1 = endLocal

    MPI_Init(&argc, &argv);                 // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // get the number of processes

    int nDims = 1; // dimension of Cartesian decomposition 1 => slices
    int dimension[ nDims ];
    int isPeriodic[ nDims ];
    int reorder = 1; // allow system to optimize(reorder) the mapping of processes to physical cores

    dimension[ 0 ]  = nProcs;
    isPeriodic[ 0 ] = 0; // periodicty of each dimension

    MPI_Comm comm1D; // define a communicator that would be assigned a new topology
    MPI_Cart_create(MPI_COMM_WORLD, nDims, dimension, isPeriodic, reorder, &comm1D);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); // get the rank of a process after REORDERING!
    MPI_Cart_shift(comm1D, 0, 1, &direction[ 0 ], &direction[ 1 ]);

    // Mesh Decompotistion
    decomposeMesh_1D(NY, nProcs, myRank, location, &nrow, p_location);
    int nGhostLayers = 2;
    int AllocSize    = (nrow + nGhostLayers) * NX;

    // Allocating Memory for every process after mesh decomposition
    REAL *phi, *phi_new, *tmp, *phi_global;
    phi     = ( REAL * ) calloc(AllocSize, sizeof(*phi));
    phi_new = ( REAL * ) calloc(AllocSize, sizeof(*phi_new));

    // Initializing Problem
    boundaryConditions(phi, nrow, nGhostLayers, direction);

#if (DEBUG)
    printf("myRank=%d, mySource=%2.1d, myDestination=%2.1d, nrow=%2.1d, [start=%2.1d, end=%2.1d, "
           "startLocal=%2.1d, endLocal=%2.1d] \n",
           myRank, direction[ 0 ], direction[ 1 ], nrow, location[ 0 ], location[ 1 ],
           p_location[ 0 ], p_location[ 1 ]);

    // Allocating/Initializing only within the Root Process (WILL BE REMOVED FOR HDF5
    // IMPLEMENTATION)
    if (myRank == MASTER) {
        phi_global = ( REAL * ) calloc(NX * NY, sizeof(*phi_global));
    }
#endif

    // Performing calculation and timing for scalibility
    MPI_Barrier(MPI_COMM_WORLD);
    double startT = MPI_Wtime( );
//    for (REAL iter = 0.f; iter < MAXITER; iter += DT) {
    for (int iter = 0; iter < 10; iter ++) {
//	exchange_SendRecv(phi, direction[0], direction[1], nrow, myRank);
        exchange_ISend_and_IRecv(phi, direction, nrow, myRank);
        SolveHeatEQ(phi, phi_new, p_location);
        boundaryConditions(phi_new, nrow, nGhostLayers, direction);
        tmp     = phi;
        phi     = phi_new;
        phi_new = tmp;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double finishT = MPI_Wtime( );

#if (DEBUG)
    // Gather data from rest of the processes into Root
    MPI_Gather(phi + NX, nrow * NX, MPI_DOUBLE, phi_global, nrow * NX, MPI_DOUBLE, MASTER,
               MPI_COMM_WORLD); // Gather can be replaced when using HDF5 output.
    // Output data from Root (WILL BE REMOVED FOR HDF5 IMPLEMENTATION)
    if (myRank == MASTER) {
        printf("***********FINAL OUTPUT AFTER GATHER***********\n");
        outputMatrix(phi_global,"Temperature.csv");

        free(phi_global);
        phi_global = NULL;
    }
#endif

    // Barrier before recording the finish time
    double elapsedTime = finishT - startT;
    double wallTime;
    MPI_Reduce(&elapsedTime, &wallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == MASTER) {
        printf("Wall-clock time = %.3f (ms) \n", wallTime * 1e3);
    }

    // Deallocating Arrays
    free(phi);
    free(phi_new);
    phi     = NULL;
    phi_new = NULL;

    MPI_Finalize( );
    return EXIT_SUCCESS;
}

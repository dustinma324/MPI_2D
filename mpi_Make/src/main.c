/*
 * Parallelizing 2D Heat Equations solver using 5 points equations
 *
 * Author: Dustin (Ting-Hsuan) Ma
 *
 * To Compile: mpicc -o MPI.exe -std=c99 -O3 -Wall -lm mpiHeat.c
 * To Run: mpirun -np 4 ./MPI.exe
 *
 */

#include "definitions.h"
#include "myFunctions.h"
#include "outputHDF5.h"

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
    phi     = ( REAL * ) calloc((nrow + nGhostLayers) * (ncol + nGhostLayers), sizeof(*phi));
    phi_new = ( REAL * ) calloc((nrow + nGhostLayers) * (ncol + nGhostLayers), sizeof(*phi_new));

    //    setAlltoValue(phi, nrow, ncol, nGhostLayers, ( REAL ) myRank + 1.0);
    boundaryConditions(phi, ncol, nrow, nGhostLayers, direction);

    MPI_Barrier(MPI_COMM_WORLD);
    double startT = MPI_Wtime( );
    for (REAL iter = 0.f; iter < MAXITER; iter += DT) {
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
    if (myRank == MASTER) { printf("Wall-clock time = %.3f (ms) \n", wallTime * 1e3); }

    output_hdf5(nDims, nrow, ncol, phi);

    // Deallocating Arrays
    free(phi);
    phi = NULL;

    MPI_Finalize( );
    return EXIT_SUCCESS;
}

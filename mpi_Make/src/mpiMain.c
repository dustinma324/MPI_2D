/*
 * Parallelizing 2D Heat Equations solver using 5 points equations
 *
 * Author: Dustin (Ting-Hsuan) Ma
 *
 * To Compile: mpicc -o MPI.exe -lm mpiHeat.c
 * To Run: mpirun -np 4 ./MPI.exe
 *
 */

#include "definitions.h"
#include "myFunctions.h"

int main(int argc, char **argv)
{
    int nProcs; // number of processes
    int myRank; // process rank
    int src;    // handles for communication, source process id
    int dest;   // handles for communication, destination process id
    int start;  // start index for each partial domain
    int end;    // end index for each partial domain
    int nrow;   // number of row needed to be allocated by local array

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
    MPI_Cart_shift(comm1D, 0, 1, &src,
                   &dest); // Let MPI find out the rank of processes for source and destination

    // Mesh Decompotistion
    decomposeMesh_1D(NY, nProcs, myRank, &start, &end);
    nrow = (end - start);
    printf("myRank=%d, mySource=%2.1d, myDestination=%2.1d, nrow=%2.1d, start=%2.1d, end=%2.1d\n",
           myRank, src, dest, nrow, start, end);
    int nGhostLayers = 2;
    int AllocSize    = (nrow + nGhostLayers) * NX;

    // Allocating Memory for every process after mesh decomposition
    REAL *local, *local_new, *tmp, *theta_new;
    local     = ( REAL * ) calloc(AllocSize, sizeof(*local));
    local_new = ( REAL * ) calloc(AllocSize, sizeof(*local_new));

    // Allocating/Initializing only within the Root Process
    if (myRank == MASTER) {
        theta_new
        = ( REAL * ) calloc(NX * NY, sizeof(*theta_new)); // Final output memory allocation
        initializeM(local, nrow, nGhostLayers);
        initializeM(local_new, nrow, nGhostLayers);
    }

    // Performing calculation and timing for scalibility
    MPI_Barrier(MPI_COMM_WORLD);
    double startT = MPI_Wtime( );
    for (REAL iter = 0.f; iter < MAXITER; iter += DT) {
        // exchange_Send_and_Receive(local, src, dest, nrow, myRank);	//Blocking
        exchange_SendRecv(local, src, dest, nrow, myRank); // Nonblocking
        SolveHeatEQ(local, local_new, nrow, myRank, nProcs);

        tmp       = local;
        local     = local_new;
        local_new = tmp;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double finishT = MPI_Wtime( );

    // Gather data from rest of the processes into Root
    MPI_Gather(local + NX, nrow * NX, MPI_DOUBLE, theta_new, nrow * NX, MPI_DOUBLE, MASTER,
               MPI_COMM_WORLD);

    /*    // Testing
        if (myRank == MASTER) {
            printf("***********Debug Matrix************\n");
            print2Display(local, start, end, nrow, myRank, nProcs);
        }
    */

    // Output data from Root
    if (myRank == MASTER) {
        printf("***********FINAL OUTPUT AFTER GATHER***********\n");
        //outputMatrix(theta_new);
    }

    // Barrier before recording the finish time
    double elapsedTime = finishT - startT;
    double wallTime;
    MPI_Reduce(&elapsedTime, &wallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myRank == 0) {
        printf("Wall-clock time = %.3f (ms) \n", wallTime * 1e3);
    }

    // Deallocating Arrays
    free(local);
    free(local_new);
    free(theta_new);

    local     = NULL;
    local_new = NULL;
    theta_new = NULL;

    MPI_Finalize( );
    return EXIT_SUCCESS;
}

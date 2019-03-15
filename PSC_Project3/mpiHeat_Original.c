/*
 * In this project, I tried to implement parallel programing with the heat equation
 * of a 2D square. I first wrote a serial version of what the Heat equation would be,
 * then I start implementing the MPI parallell programming. Throughout the duration
 * of working with MPI, I have not found the reason to why my solver isn't propegating
 * to the rest of the proccesses. I tested the Send and Receive code block, but when
 * solving the problem, I couldn't get it.
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

// Spacial
#define LX ( REAL ) 20.0f
#define LY (REAL) LX
#define NX ( INT ) 12
#define NY (INT) NX
#define DX LX / (( REAL ) NX - 1.0f)
#define DY (REAL) DX

// Temperature
#define TMAX ( REAL ) 100.0f
#define TMIN ( REAL ) 0.0f

// Time
#define DT ( REAL ) 0.25f * DX *DX
#define MAXITER 500

// Calculation index
#define IC i + j *NX
#define IP1 (i + 1) + j *NX
#define IM1 (i - 1) + j *NX
#define JP1 i + (j + 1) * NX
#define JM1 i + (j - 1) * NX

#define MASTER 0

typedef double       REAL;
typedef const double C_REAL;
typedef int          INT;

void initializeM(REAL *in, const int nrow, const int nGhostLayers)
{
    for (INT j = 0; j < nrow + nGhostLayers; j++) {
        for (INT i = 0; i < NX; i++) {
            if (j == 1) {
                in[ IC ] = TMAX;
            }
            if (j == nrow) {
                in[ IC ] = TMIN;
            }
            if (i == 0) {
                in[ IC ] = TMIN;
            }
            if (i == NX - 1) {
                in[ IC ] = TMIN;
            }
        }
    }
}

void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *start, int *end)
{
    *start = myRank * N / nProcs;
    *end   = *start + N / nProcs;
}

void SolveHeatEQ(C_REAL *now, REAL *out, const int nrow, const int myRank, const int nProcs)
{
    int startLoc;
    if (myRank == 0)
        startLoc = 2;
    else
        startLoc = 1;

    int endLoc;
    if (myRank == nProcs - 1)
        endLoc = nrow;
    else
        endLoc = nrow + 1;

    for (INT j = startLoc; j < endLoc; j++) {
        for (INT i = 1; i < NX - 1; i++) {
            out[ IC ] = (((now[ IP1 ] - 2.0f * now[ IC ] + now[ IM1 ]) / (DX * DX))
                         + ((now[ JP1 ] - 2.0f * now[ IC ] + now[ JM1 ])) / (DY * DY))
                        * DT
                        + now[ IC ];
        }
    }
}

void exchange_SendRecv(REAL *in, const int src, const int dest, const int nrow, const int myRank)
{
    int tag0 = 0;                                                           // up send tag
    int tag1 = 1;                                                           // down send tag
    MPI_Send(in + (nrow * NX), NX, MPI_DOUBLE, dest, tag1, MPI_COMM_WORLD); // send to down
    MPI_Send(in + NX, NX, MPI_DOUBLE, src, tag0, MPI_COMM_WORLD);           // send to up

    MPI_Recv(in, NX, MPI_DOUBLE, src, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive from up
    MPI_Recv(in + ((nrow + 1) * NX), NX, MPI_DOUBLE, dest, tag0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE); // receive from down
}

void outputMatrix(C_REAL *in)
{
    for (INT j = 0; j < NY; j++) {
        for (INT i = 0; i < NX; i++) {
            printf("%8.4f ", in[ IC ]);
        }
        printf("\n");
    }
    printf("\n");
}

void print2Display(C_REAL *in, const int start, const int end, const int nrow, const int myRank,
                   const int nProcs)
{
    int nGhostLayers = 2;
    for (int j = 0; j < nrow + nGhostLayers; j++) {
        for (int i = 0; i < NX; i++) {
            printf("%8.4f ", in[ IC ]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int nProcs; /* number of processes */
    int myRank; /* process rank */
    int src;    /* handles for communication, source process id */
    int dest;   /* handles for communication, destination process id */
    int start;  /* start index for each partial domain */
    int end;    /* end index for each partial domain */
    int nrow;   /* number of row needed to be allocated by local array */

    MPI_Init(&argc, &argv);                 /* initialize MPI */
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); /* get the number of processes */

    int nDims = 1; // dimension of Cartesian decomposition 1 => slices
    int dimension[ nDims ];
    int isPeriodic[ nDims ];
    int reorder = 1; // allow system to optimize(reorder) the mapping of processes to physical cores

    dimension[ 0 ]  = nProcs;
    isPeriodic[ 0 ] = 0; // periodicty of each dimension

    MPI_Comm comm1D; // define a communicator that would be assigned a new topology
    MPI_Cart_create(MPI_COMM_WORLD, nDims, dimension, isPeriodic, reorder, &comm1D);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); /* get the rank of a process after REORDERING! */
    MPI_Cart_shift(comm1D, 0, 1, &src,
                   &dest); /* Let MPI find out the rank of processes for source and destination */

    // Mesh Decompotistion
    decomposeMesh_1D(NY, nProcs, myRank, &start, &end);
    nrow = (end - start);
    printf("myRank=%d, mySource=%2.1d, myDestination=%2.1d, nrow = %d, start = %d, end = %d \n",
           myRank, src, dest, nrow, start, end);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initializing Matrix
    int nGhostLayers = 2;
    int AllocSize    = (nrow + nGhostLayers) * NX;

    REAL *local, *local_new;
    local     = ( REAL * ) calloc(AllocSize, sizeof(*local));
    local_new = ( REAL * ) calloc(AllocSize, sizeof(*local_new));

    REAL *theta_new = ( REAL * ) calloc(NX * NY, sizeof(*theta_new));

    if (myRank == MASTER) {
        initializeM(local, nrow, nGhostLayers);
        initializeM(local_new, nrow, nGhostLayers);
    }

    // Starting the solver
    REAL *tmp;
    for (int iter = 0; iter < MAXITER; iter++) {
        exchange_SendRecv(local, src, dest, nrow, myRank);
        MPI_Barrier(MPI_COMM_WORLD);
        SolveHeatEQ(local, local_new, nrow, myRank, nProcs);

        tmp       = local;
        local     = local_new;
        local_new = tmp;
    }

    MPI_Gather(local + NX, nrow * NX, MPI_DOUBLE, theta_new, nrow * NX, MPI_DOUBLE, MASTER,
               MPI_COMM_WORLD);

    if (myRank == MASTER) {
        printf("***********FINAL OUTPUT AFTER GATHER***********\n");
        outputMatrix(theta_new);
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

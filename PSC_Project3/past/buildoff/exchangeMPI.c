/*
Program: exchangeMPI

C code to test MPI_Send and MPI_Recv and different flavors of them
A matrix a[i][j] is decomposed with a one dimensional slicing strategy.
The portion of the array on each process is initialized to the process rank
and an exchange is performed. Results can be printed both before and after the
exchange to demonstrate the correctness.

Author: Inanc Senocak

to compile: mpicc -o2 exchangeMPI.c -o run.exe
to execute: mpirun -np #procs ./run.exe

*/

#include <math.h>
#include <mpi.h> /* need it for MPI functions */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#define NROW 10
#define NCOL 10

/* external variables */

void initData(int *phi, const int myRank, const int start, const int end)
{
    //  store like a matrix phi[i][j]
    int i, j, ic;
    int nrow = (end - start) + 1;

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < NCOL; j++) {
            ic        = j + i * NCOL;
            phi[ ic ] = myRank;
        }
    }
}

void print2Display(const int *phi, const int start, const int end)
{
    int i, j, ic;
    int nrow = (end - start) + 1;

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < NCOL; j++) {
            ic = j + i * NCOL;
            printf("%d ", phi[ ic ]);
        }
        printf("\n");
    }
}

void exchange_Blocking(int *phi, const int start, const int end, int src, int dest, const int myRank,
                       const int nProcs)
{
    /* this implementation depends on the buffering. Hence, it is not recommended */

    int tag0 = 0;
    int tag1 = 1;

    int e = (end - 1) * NCOL;
    int s = 0;

    MPI_Send(&phi[ e ], NCOL, MPI_INT, dest, tag0,
             MPI_COMM_WORLD); // send do not complete until matching receive takes place
    MPI_Recv(&phi[ s ], NCOL, MPI_INT, src, tag0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    s       = (start + 1) * NCOL;
    e       = end * NCOL;
    int tmp = dest;
    dest    = src;
    src     = tmp;

    MPI_Send(&phi[ s ], NCOL, MPI_INT, dest, tag1, MPI_COMM_WORLD);
    MPI_Recv(&phi[ e ], NCOL, MPI_INT, src, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void exchange_nonBlocking(int *phi, const int start, const int end, int src, int dest, const int myRank,
                          const int nProcs)
{
    /* this is the recommended implementation with nonblocking communications */

    int tag0 = 0;
    int tag1 = 1;
    int tmp;

    MPI_Request request[ 4 ]; // to determine whether an operation completed
    MPI_Status status[ 4 ];

    int s, e;

    s = 0;

    MPI_Irecv(&phi[ s ], NCOL, MPI_INT, src, tag0, MPI_COMM_WORLD, &request[ 0 ]);

    e    = end * NCOL;
    tmp  = dest;
    dest = src;
    src  = tmp;

    MPI_Irecv(&phi[ e ], NCOL, MPI_INT, src, tag1, MPI_COMM_WORLD, &request[ 1 ]);

    e    = (end - 1) * NCOL;
    tmp  = dest;
    dest = src;
    src  = tmp;

    MPI_Isend(&phi[ e ], NCOL, MPI_INT, dest, tag0, MPI_COMM_WORLD, &request[ 2 ]);

    s    = (start + 1) * NCOL;
    tmp  = dest;
    dest = src;
    src  = tmp;

    MPI_Isend(&phi[ s ], NCOL, MPI_INT, dest, tag1, MPI_COMM_WORLD, &request[ 3 ]);

    MPI_Waitall(4, request, status);
}

void exchange_SendRecv(int *phi, const int start, const int end, int src, int dest, const int myRank,
                       const int nProcs)
{
    int tag0 = 0;
    int tag1 = 1;
    int e    = (end - 1) * NCOL;
    int s    = 0;

    MPI_Sendrecv(&phi[ e ], NCOL, MPI_INT, dest, tag0, &phi[ s ], NCOL, MPI_INT, src, tag0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    s = (start + 1) * NCOL;
    e = end * NCOL;

    // NOTE WE ARE NOW SWAPPING THE SRC & DEST
    int tmp = dest;
    dest    = src;
    src     = tmp;
    MPI_Sendrecv(&phi[ s ], NCOL, MPI_INT, dest, tag1, &phi[ e ], NCOL, MPI_INT, src, tag1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *start, int *end,
                      const int nGhostLayers)
{
    int remainder = N % nProcs;
    if (remainder == 0) {
        *start = 0;
        *end   = (N / nProcs) + nGhostLayers - 1;
    } else {
        *start               = 0;
        int pointsPerProcess = (N - remainder) / nProcs + 1;
        if (myRank == (nProcs - 1))
            *end = (N - pointsPerProcess * (nProcs - 1)) + nGhostLayers - 1;
        else
            *end = pointsPerProcess + nGhostLayers - 1;
    }
}

int main(int argc, char *argv[])
{
    int nProcs; /* number of processes */
    int myRank; /* process rank */
    int src;    /* handles for communication, source process id */
    int dest;   /* handles for communication, destination process id */
    int start;  /* start index for each partial domain */
    int end;    /* end index for each partial domain */

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
    MPI_Cart_shift(comm1D, 0, 1, &src, &dest); /* Let MPI find out the rank of processes for source and destination */

    //    printf("\n");
    //    printf("myRank=%d mySource=%d myDestination=%d\n",myRank, src, dest);
    //    printf("\n");

    int nGhostLayers;
    if (myRank == 0 || myRank == nProcs - 1) {
        nGhostLayers = 1;
    } else {
        nGhostLayers = 2;
    }

    decomposeMesh_1D(NROW, nProcs, myRank, &start, &end, nGhostLayers);

    int nrow = (end - start) + 1;

    int *u = malloc(NCOL * nrow * sizeof(*u));

    initData(u, myRank, start, end);

    MPI_Barrier(MPI_COMM_WORLD); // make sure all processes initialized their portion of the problem

    // int testRank = nProcs-1;
    int testRank = 2;

    if (myRank == testRank) {
        printf("\n");
        printf("myRank=%d myStart=%d myEnd=%d my_nrow=%d\n", myRank, start, end, nrow);
        printf("myRank=%d nGhostLayers=%d mySource=%d myDestination=%d\n", myRank, nGhostLayers, src, dest);
        printf("\n");
        printf("Before the exchange\n");
        print2Display(u, start, end);
        printf("================\n");
        printf("\n");
    }
    exchange_SendRecv(u, start, end, src, dest, myRank, nProcs);
    // exchange_Blocking( u, start, end, src, dest, myRank, nProcs);
    // exchange_nonBlocking( u, start, end, src, dest, myRank, nProcs);

    if (myRank == testRank) {
        printf("After the exchange\n");
        printf("================\n");
        print2Display(u, start, end);
    }
    //  writeOutput( u );
    free(u);
    MPI_Finalize( );

    return EXIT_SUCCESS;
}


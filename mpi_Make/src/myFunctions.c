#include "myFunctions.h"

void initializeM(REAL *in, const int nrow, const int nGhostLayers)
{
    for (INT j = 0; j < nrow + nGhostLayers; j++) {
        for (INT i = 0; i < NX; i++) {
            if (j == 1) {
                in[ IC ] = TMAX;
            }
            if (j == nrow + 1) {
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

void SolveHeatEQ(const REAL *now, REAL *out, const int nrow, const int myRank, const int nProcs)
{
    int startLoc, endLoc;

    /* Special condition that applies for ROOT process */
    if (myRank == MASTER)
        startLoc = 2;
    else
        startLoc = 1;

    /* Special condition that applied for nProcs-1 process */
    if (myRank == nProcs - 1)
        endLoc = nrow;
    else
        endLoc = nrow + 1;

    /* Heat solving iteration only performs calculations on the center cells of matrix */
    for (INT j = startLoc; j < endLoc; j++) {
        for (INT i = 1; i < NX - 1; i++) {
            out[ IC ] = (((now[ IP1 ] - 2.0f * now[ IC ] + now[ IM1 ]) / (DX * DX))
                         + ((now[ JP1 ] - 2.0f * now[ IC ] + now[ JM1 ])) / (DY * DY))
                        * DT
                        + now[ IC ];
        }
    }
}

/* Blocking send and receive
 * 
 * Sperately sends NX values down and up, and the respective
 * process initiate a receive call.
 */
void exchange_Send_and_Receive(REAL *in, const int src, const int dest, const int nrow,
                               const int myRank)
{
    int tag0 = 0;                                                           // up send tag
    int tag1 = 1;                                                           // down send tag
    MPI_Send(in + (nrow * NX), NX, MPI_DOUBLE, dest, tag1, MPI_COMM_WORLD); // send to down
    MPI_Send(in + NX, NX, MPI_DOUBLE, src, tag0, MPI_COMM_WORLD);           // send to up

    MPI_Recv(in, NX, MPI_DOUBLE, src, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive from up
    MPI_Recv(in + ((nrow + 1) * NX), NX, MPI_DOUBLE, dest, tag0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE); // receive from down
}

/* Nonblocking send&receive
 * 
 * The function sends NX values to the bottom process.
 * In return, the bottom process also sends NX values
 * back up to the top process
 */
void exchange_SendRecv(REAL *in, const int src, const int dest, const int nrow, const int myRank)
{
    int tag0 = 0; // send tag
    int tag1 = 1; // send tag
    MPI_Sendrecv(in + (nrow * NX), NX, MPI_DOUBLE, dest, tag0, in, NX, MPI_DOUBLE, src, tag0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Sending down
    MPI_Sendrecv(in + NX, NX, MPI_DOUBLE, src, tag1, in + (nrow + 1) * NX, NX, MPI_DOUBLE, dest,
                 tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Sending up
}

void outputMatrix(const REAL *in)
{
    for (INT j = 0; j < NY; j++) {
        for (INT i = 0; i < NX; i++) {
            printf("%8.4f ", in[ IC ]);
        }
        printf("\n");
    }
    printf("\n");
}

void print2Display(const REAL *in, const int start, const int end, const int nrow, const int myRank,
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


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Spacial
#define LX (REAL) 20.0f
#define LY (REAL) LX
#define NX (INT) 10
#define NY (INT) NX
#define DX LX / ((REAL) NX - 1.0f)
#define DY (REAL) DX

// Temperature
#define TMAX (REAL) 100.0f
#define TMIN (REAL) 0.0f

// Time
#define DT (REAL) 0.25f * DX *DX
#define MAXITER 100

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

void SolveHeatEQ(C_REAL *now, REAL *out)
{
	for (INT j = 1; j < NY - 1; j++) {
		for (INT i = 1; i < NX - 1; i++) {
			out[IC] = (((now[IP1] - 2.0f * now[IC] + now[IM1]) / (DX * DX))
			           + ((now[JP1] - 2.0f * now[IC] + now[JM1])) / (DY * DY))
			          * DT
			          + now[IC];
		}
	}
}

void meshGrid(REAL *xGrid, REAL *yGrid)
{
	for (INT j = 0; j < NY; j++) {
		for (INT i = 0; i < NX; i++) {
			xGrid[IC] = i * DX;
			yGrid[IC] = j * DY;
		}
	}
}

void initializeM(REAL *a)
{
	for (INT j = 0; j < NY; j++) {
		for (INT i = 0; i < NX; i++) {
			INT idx = i + j * NX;
			a[idx]  = (REAL) idx;
		}
	}
}
/*
void initializeM(REAL *in)
{
        for (INT j = 0; j < NY; j++) {
                for (INT i = 0; i < NX; i++) {
                        if (j == 0) {
                                in[IC] = TMIN;
                        }
                        if (j == NY - 1) {
                                in[IC] = TMAX;
                        }
                        if (i == 0) {
                                in[IC] = TMIN;
                        }
                        if (i == NX - 1) {
                                in[IC] = TMIN;
                        }
                }
        }
}
*/

void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *start, int *end)
{
        *start = myRank * N / nProcs;
        *end   = *start + (myRank * N / nProcs);
}

void outputMatrix(C_REAL *in)
{
        for (INT j = 0; j < NY; j++) {
                for (INT i = 0; i < NX; i++) {
                        printf("%8.4f ", in[IC]);
                }
                printf("\n");
        }
        printf("\n");
}

void print2Display(C_REAL *in, const int start, const int end, const int nrow)
{
    for (int j = 0; j < nrow; j++) {
        for (int i = 0; i < NX; i++) {
            printf("%8.4f ", in[IC]);
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

	MPI_Init(&argc, &argv);                 /* initialize MPI */
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs); /* get the number of processes */

	int nDims = 1; // dimension of Cartesian decomposition 1 => slices
	int dimension[nDims];
	int isPeriodic[nDims];
	int reorder = 1; // allow system to optimize(reorder) the mapping of processes to physical cores

	dimension[0]  = nProcs;
	isPeriodic[0] = 0; // periodicty of each dimension

	MPI_Comm comm1D; // define a communicator that would be assigned a new topology
	MPI_Cart_create(MPI_COMM_WORLD, nDims, dimension, isPeriodic, reorder, &comm1D);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank); /* get the rank of a process after REORDERING! */
	MPI_Cart_shift(comm1D, 0, 1, &src, &dest); /* Let MPI find out the rank of processes for source and destination */

	//Defining ghost layer for each process
	int nGhostLayers;
	if (myRank == MASTER || myRank == nProcs - 1) {
		nGhostLayers = 1;
	} else {
		nGhostLayers = 2;
	}
	
	// Mesh decomposition
	decomposeMesh_1D(NX, nProcs, myRank, &start, &end);
	int nrow = (end - start) + 1;
        REAL *theta     = (REAL *) calloc(NX * NY, sizeof(*theta));
        REAL *theta_new = (REAL *) calloc(NX * NY, sizeof(*theta_new));
	REAL *localdata = (REAL *) calloc(nrow * NX, sizeof(*localdata));
        initializeM(theta);

	// Testing Ranks
        int testRank = 0;

	//Master Process
	if (myRank == testRank){
		printf("***** BEFORE SCATTER *****\n");
                printf("myRank=%d myStart=%d myEnd=%d my_nrow=%d\n", myRank, start, end, nrow);
                printf("myRank=%d nGhostLayers=%d mySource=%d myDestination=%d\n", myRank, nGhostLayers, src, dest);
		print2Display(localdata,start,end,nrow);
		printf("\n");
	//	outputMatrix(theta);
	}

	// Scatter information from to the rest of the processes
	MPI_Scatter(theta, nrow*NX, MPI_DOUBLE, localdata, nrow*NX, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	// Gathering information from the rest of the processes
	MPI_Gather(localdata, nrow*NX, MPI_DOUBLE, theta_new, nrow*NX, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	// Deallocating memory in all processes
	free(localdata);
        free(theta);
        free(theta_new);
        theta_new = NULL;
        theta     = NULL;
	localdata = NULL;

	MPI_Finalize();
	return EXIT_SUCCESS;
}

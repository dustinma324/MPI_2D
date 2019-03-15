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

void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *start, int *end,
                      const int nGhostLayers)
{
	int remainder = N % nProcs;
	if (remainder == 0) {
		*start = 0;
		*end   = (N / nProcs) + nGhostLayers - 1;
	} else {
		*start = 0;
		int pointsPerProcess = (N - remainder) / nProcs + 1;
		if (myRank == (nProcs - 1))
			*end = (N - pointsPerProcess * (nProcs - 1)) + nGhostLayers - 1;
		else
			*end = pointsPerProcess + nGhostLayers - 1;
	}
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

void print2Display(C_REAL *in, const int start, const int end)
{
    int nrow = (end - start) + 1;

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
	int numdata;

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
	
	// Finding out number of data sent to each process
        int remainder = (NX * NX) % nProcs;
        if (remainder == 0) {
                numdata = (NX * NX) / nProcs;
        } else {
                numdata = ((NX * NX) - remainder) / nProcs + 1;
        }

	// Mesh decomposition
	decomposeMesh_1D(NY, nProcs, myRank, &start, &end, nGhostLayers);
        int   nrow      = (end - start) + 1;
        REAL *theta, *theta_new;
        REAL *localdata = (REAL *) calloc(nrow * NX, sizeof(*localdata));

	// Testing Ranks
        int testRank = 0;
	
	// Initializing Data special to Root processor
	if (myRank == MASTER) {
		// Allocating memory only in root process
		theta     = (REAL *) calloc(NX * NY, sizeof(*theta));
		theta_new = (REAL *) calloc(NX * NY, sizeof(*theta_new));
		initializeM(theta);

		printf("*** BEFORE: Processor %1.2d ***\n", myRank);
		printf("numdata = %1.2d\n", numdata);
//		outputMatrix(theta);
	}

	// Scattering information to rest of the processes
	if (MPI_Scatterv(theta, nrow*NX, int *displs, MPI_DOUBLE, localdata, nrow*NX, MPI_DOUBLE, MASTER, MPI_Comm comm)

//MPI_Scatter(theta, nrow*NX, MPI_DOUBLE, localdata, nrow*NX, MPI_DOUBLE, 0, MPI_COMM_WORLD)
	    != MPI_SUCCESS) {
		perror("Scatter error");
		exit(1);
	}

        if (myRank == testRank) {
                printf("\n");
                printf("myRank=%d myStart=%d myEnd=%d my_nrow=%d\n", myRank, start, end, nrow);
                printf("myRank=%d nGhostLayers=%d mySource=%d myDestination=%d\n", myRank, nGhostLayers,
                       src, dest);
                printf("\n");
                printf("After Scatter\n");
                print2Display(localdata, start, end);
                printf("================\n");
                printf("\n");
        }

/*
	int iter = 0;
	while (iter < MAXITER) {

//		SolveHeatEQ(theta, theta_new);
//		theta = theta_new;
		MPI_Barrier(MPI_COMM_WORLD); // Blocks all process until all have reach this routine
		iter++;
	}
*/

	// Gathering information from the rest of the processes
	if (MPI_Gather(localdata, nrow*NX, MPI_DOUBLE, theta_new, nrow*NX, MPI_DOUBLE, MASTER, MPI_COMM_WORLD)
	    != MPI_SUCCESS) {
		perror("Scatter error");
		exit(1);
	}

	// Deallocating and freeing memory on special to root
	if (myRank == MASTER) {
//		printf("*** AFTER: Processor %1.2d ***\n", myRank);
//		outputMatrix(theta_new);

		free(theta);
		free(theta_new);
		theta_new = NULL;
		theta     = NULL;
	}

	// Deallocating memory in all processes
	free(localdata);
	localdata = NULL;

	MPI_Finalize();
	return 0;
}

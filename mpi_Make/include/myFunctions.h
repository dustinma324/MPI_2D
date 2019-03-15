#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

#include "definitions.h"

void initializeM(REAL *in, const int nrow, const int nGhostLayers);
void decomposeMesh_1D(const int N, const int nProcs, const int myRank, int *start, int *end);
void SolveHeatEQ(const REAL *now, REAL *out, const int nrow, const int myRank, const int nProcs);
void exchange_Send_and_Receive(REAL *in, const int src, const int dest, const int nrow, const int myRank);
void exchange_SendRecv(REAL *in, const int src, const int dest, const int nrow, const int myRank);
void outputMatrix(const REAL *in);
void print2Display(const REAL *in, const int start, const int end, const int nrow, const int myRank, const int nProcs);

#endif


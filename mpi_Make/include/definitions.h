#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"

#define FILE "file.h5"

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
#define N ( INT ) 0
#define S ( INT ) 1
#define E ( INT ) 2
#define W ( INT ) 3

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
#define IC i + j *(ncol + nGhostLayers)
#define IP1 (i + 1) + j *(ncol + nGhostLayers)
#define IM1 (i - 1) + j *(ncol + nGhostLayers)
#define JP1 i + (j + 1) * (ncol + nGhostLayers)
#define JM1 i + (j - 1) * (ncol + nGhostLayers)

// Process
#define MASTER 0

// Debug
#define DEBUG 0
#define OUTPUT 0 
#endif

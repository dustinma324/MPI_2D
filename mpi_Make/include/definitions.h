#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef double       REAL;
typedef int          INT;

// Spacial
#define LX ( REAL ) 20.0f
#define LY ( REAL ) 20.0f
#define NX ( INT ) 100
#define NY ( INT ) 10
#define DX LX / (( REAL ) NX - 1.0f)
#define DY LY / (( REAL ) NY - 1.0f)

// Temperature
#define TMAX ( REAL ) 100.0f
#define TMIN ( REAL ) 0.0f

// Time
#define DT ( REAL ) 0.25f * DY *DY
#define MAXITER LX*LX

// Calculation index
#define IC i + j *NX
#define IP1 (i + 1) + j *NX
#define IM1 (i - 1) + j *NX
#define JP1 i + (j + 1) * NX
#define JM1 i + (j - 1) * NX

// Process
#define MASTER 0

#endif

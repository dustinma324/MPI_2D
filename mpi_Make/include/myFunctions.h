#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

#include "definitions.h"

void boundaryConditions(REAL *phi, const INT ncol, const INT nrow, const int nGhostLayers,
                        const int *direction);
void decomposeMesh_2D(INT *coord_2D, INT *direction, INT *location, INT *p_location, INT *nrow,
                      INT *ncol);
void exchange_ISend_and_IRecv(REAL *phi, const int *direction, const INT ncol, const INT nrow,
                              const int nGhostLayers);
void SolveHeatEQ(const REAL *phi, REAL *phi_new, const int *p_location, const INT ncol,
                 const INT nrow, const INT nGhostLayers);
void setAlltoValue(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, REAL value);
#if (OUTPUT)
void outputMatrix(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, char *name);
void outputMatrix1(REAL *phi, INT nrow, INT ncol, INT nGhostLayers, char *name);
#endif

#endif

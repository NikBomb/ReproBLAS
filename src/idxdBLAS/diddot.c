#include "idxdBLAS.h"

/**
 * @brief Add to indexed double precision Z the dot product of double precision vectors X and Y
 *
 * Add to Z the indexed sum of the pairwise products of X and Y.
 *
 * @param N vector length
 * @param X double precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param Y double precision vector
 * @param incY Y vector stride (use every incX'th element)
 * @param Z indexed scalar Z
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void idxdBLAS_diddot(const int fold, const int N, const double *X, const int incX, const double *Y, const int incY, double_indexed *Z){
  idxdBLAS_dmddot(fold, N, X, incX, Y, incY, Z, 1, Z + fold, 1);
}

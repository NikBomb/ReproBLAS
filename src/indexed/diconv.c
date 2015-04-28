/*
 *  Created   13/10/25   H.D. Nguyen
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "indexed.h"
#include "../Common/Common.h"

/**
 * @brief Convert double precision to manually specified indexed double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X scalar X
 * @param repY Y's rep vector
 * @param increpY stride within Y's rep vector (use every increpY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void dmdconv(const int fold, double X, double* repY, int increpY, double* carY, int inccarY) {
  int i;
  double q;
  double s;
  if (X == 0.0) {
    for (i = 0; i < fold; i++) {
      repY[i*increpY] = 0.0;
      carY[i*inccarY] = 0.0;
    }
    return;
  }
  dmbound(fold, dindex(fabs(X)), repY, increpY);
  for (i = 0; i < fold; i++, repY += increpY, carY += inccarY) {
    carY[0] = 0.0;

    s = repY[0];
    q = s + X;
    repY[0] = s;
    q -= s;
    X -= q;
  }
}

/**
 * @brief Convert double precision to indexed double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X scalar X
 * @param Y indexed scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void didconv(const int fold, double X, double_indexed *Y) {
  dmdconv(fold, X, Y, 1, Y + fold, 1);
}

/**
 * @brief Convert complex double precision to manually specified indexed complex double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X scalar X
 * @param repY Y's rep vector
 * @param increpY stride within Y's rep vector (use every increpY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void zmzconv(const int fold, void *X, double *repY, int increpY, double *carY, int inccarY) {
  dmdconv(fold, ((double*)X)[0], repY, increpY * 2, carY, inccarY * 2);
  dmdconv(fold, ((double*)X)[1], repY + 1, increpY * 2, carY + 1, inccarY * 2);
}

/**
 * @brief Convert complex double precision to indexed complex double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X scalar X
 * @param Y indexed scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void zizconv(const int fold, void *X, double_complex_indexed *Y) {
  zmzconv(fold, X, Y, 1, Y + 2 * fold, 1);
}

/**
 * @brief Convert manually specified indexed double precision to double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param repX X's rep vector
 * @param increpX stride within X's rep vector (use every increpX'th element)
 * @param carX X's carry vector
 * @param inccarX stride within X's carry vector (use every inccarX'th element)
 * @return scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
double ddmconv(const int fold, double* repX, int increpX, double* carX, int inccarX) {
  int i;
  double Y = 0.0;

  if (isinf(repX[0]) || isnan(repX[0]))
    return repX[0];

  if (repX[0] == 0.0) {
    return 0.0;
  }

  // TODO: SCALING TO AVOID OVERFLOW

  for (i = 0; i < fold; i++, repX += increpX, carX += inccarX) {
    Y += (repX[0] + (carX[0] - 6) * ufp(repX[0]) * 0.25);
  }

  return Y;
}

/**
 * @brief Convert indexed double precision to double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X indexed scalar X
 * @return scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
double ddiconv(const int fold, double_indexed *X) {
  return ddmconv(fold, X, 1, X + fold, 1);
}

/**
 * @brief Convert manually specified indexed complex double precision to complex double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param repX X's rep vector
 * @param increpX stride within X's rep vector (use every increpX'th element)
 * @param carX X's carry vector
 * @param inccarX stride within X's carry vector (use every inccarX'th element)
 * @param Y scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void zzmconv_sub(const int fold, double *repX, int increpX, double *carX, int inccarX, void *Y) {
  ((double*)Y)[0] = ddmconv(fold, repX, increpX * 2, carX, inccarX + 1);
  ((double*)Y)[1] = ddmconv(fold, repX + 1, increpX * 2, carX + 1, inccarX + 1);
}

/**
 * @brief Convert indexed complex double precision to complex double precision (X -> Y)
 *
 * @param fold the fold of the indexed types
 * @param X indexed scalar X
 * @param Y scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void zziconv_sub(const int fold, double_complex_indexed *X, void *Y) {
  zzmconv_sub(fold, X, 1, X + 2 * fold, 1, Y);
}

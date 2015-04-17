#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "reproBLAS.h"
#include "indexedBLAS.h"

float snrm2I(
	int N,
	float* x, int incx,
	I_float* dot
) {
	return snrm2I1(N, x, incx, DEFAULT_FOLD, dot->m, dot->c);
}

float rsnrm2(int N, float* v, int inc) {
	I_float sum;
	double scale;
	float sqrt_sum;

	sISetZero(sum);

	scale = snrm2I1(N, v, inc, DEFAULT_FOLD, sum.m, sum.c);

	if (isnan(scale))
		return scale;

	if (isinf(sum.m[0]))
		return sum.m[0];

	sqrt_sum = ssiconv(&sum, DEFAULT_FOLD);
	sqrt_sum = sqrt(sqrt_sum);
	return scale * sqrt_sum;
}


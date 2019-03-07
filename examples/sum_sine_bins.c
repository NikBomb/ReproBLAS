#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif


static struct timeval start;
static struct timeval end;

void tic(void){
  gettimeofday( &start, NULL );
}

double toc(void){
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void doubledouble_plus_double(double* a, double b){
  double bv;
  double s1, s2, t1, t2;

  // Add two hi words
  s1 = a[0] + b;
  bv = s1 - a[0];
  s2 = ((b - bv) + (a[0] - (s1 - bv)));

  t1 = a[1] + s2;
  bv = t1 - a[1];
  t2 = ((s2 - bv) + (a[1] - (t1 - bv)));

  s2 = t1;

  // Renormalize (s1, s2)  to  (t1, s2)
  t1 = s1 + s2;
  t2 += s2 - (t1 - s1);

  // Renormalize (t1, t2)
  a[0] = t1 + t2;
  a[1] = t2 - (a[0] - t1);
}

int main(int argc, char** argv){
  
  int n = 1000000;
  double *x = malloc(n * sizeof(double));
  int n_bins = 100;
  int n_more_bins = n;
  int bin_size;
  int more_bin_size;
  
  double *partial_bins = malloc(n_bins * sizeof(double));
  double *partial_more_bins = malloc(n_more_bins * sizeof(double));
 
  double sum_bins;
  double sum_more_bins;
  double elapsed_time;
  
  // Set x to be a sine wave
  for(int i = 0; i < n; i++){
    x[i] = sin(2 * M_PI * (i / (double)n - 0.5));
  }
  
  // Make a header
  printf("%15s : Time (s) : |Sum - Sum of Bins| = ?\n", "Sum Method");
  
  // First, we sum x using double precision
  tic();
  
  sum_bins = 0;
  sum_more_bins = 0;
  bin_size  = n/n_bins;
  more_bin_size  = n/n_more_bins;
  
  for(int i = 0; i < n_bins; i++){
	partial_bins[i] = 0;
  }
  for(int i = 0; i < n_more_bins; i++){
	partial_more_bins[i] = 0;
  }
  
  for(int i = 0; i < n_bins; i++){
	  for (int j = i * bin_size; j < (i + 1) * bin_size; j++)
      partial_bins[i] += x[j];
  }
  elapsed_time = toc();
  
  for(int i = 0; i < n_more_bins; i++){
	  for (int j = i * more_bin_size; j < (i + 1) * more_bin_size; j++)
		partial_more_bins[i] += x[j];
  }
  
  for (int i = 0; i < n_bins; i++){
	  sum_bins += partial_bins[i];
  }
  
  for (int i = 0; i < n_more_bins; i++){
	  sum_more_bins += partial_more_bins[i];
  }
  
  printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "double", elapsed_time, sum_bins, sum_more_bins, fabs(sum_bins - sum_more_bins));
  
  
  // Sum using ReproBlas
  
  tic();
  sum_bins = reproBLAS_dsum(n_bins, partial_bins, 1);
  elapsed_time = toc();

  tic();
  sum_more_bins = reproBLAS_dsum(n_more_bins, partial_more_bins, 1);
  elapsed_time = toc();

  printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "Reproblas sum", elapsed_time, sum_bins, sum_more_bins, fabs(sum_bins - sum_more_bins));
  
  
  double_binned **isum = malloc(n_more_bins * sizeof(binned_dballoc(3)));
  binned_dbdconv(3, x[0], isum[0]);
  
  free(x);
  free(partial_bins);
  free(partial_more_bins);
 
 
 //Sum using Reproblas Primitives
 
  
}

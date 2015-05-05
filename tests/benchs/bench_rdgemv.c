#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <reproBLAS.h>

#include "../common/test_opt.h"
#include "../common/test_time.h"
#include "../common/test_metric.h"

#include "bench_matvec_fill_header.h"

static opt_option incY;
static opt_option FillY;
static opt_option ScaleY;
static opt_option CondY;
static opt_option alpha;
static opt_option beta;

static void bench_rdgemv_options_initialize(void){
  incY._int.header.type       = opt_int;
  incY._int.header.short_name = 'y';
  incY._int.header.long_name  = "incY";
  incY._int.header.help       = "Y vector increment";
  incY._int.required          = 0;
  incY._int.min               = 1;
  incY._int.max               = INT_MAX;
  incY._int.value             = 1;

  FillY._named.header.type       = opt_named;
  FillY._named.header.short_name = 'j';
  FillY._named.header.long_name  = "FillY";
  FillY._named.header.help       = "Y fill type";
  FillY._named.required          = 0;
  FillY._named.n_names           = (int)util_vec_fill_n_names;
  FillY._named.names             = (char**)util_vec_fill_names;
  FillY._named.descs             = (char**)util_vec_fill_descs;
  FillY._named.value             = 0;

  ScaleY._double.header.type       = opt_double;
  ScaleY._double.header.short_name = 'v';
  ScaleY._double.header.long_name  = "ScaleY";
  ScaleY._double.header.help       = "Y scale";
  ScaleY._double.required          = 0;
  ScaleY._double.min               = 0;
  ScaleY._double.max               = DBL_MAX;
  ScaleY._double.value             = 1.0;

  CondY._double.header.type       = opt_double;
  CondY._double.header.short_name = 'e';
  CondY._double.header.long_name  = "CondY";
  CondY._double.header.help       = "Y condition number";
  CondY._double.required          = 0;
  CondY._double.min               = 1.0;
  CondY._double.max               = DBL_MAX;
  CondY._double.value             = 1e3;

  alpha._double.header.type       = opt_double;
  alpha._double.header.short_name = 'l';
  alpha._double.header.long_name  = "alpha";
  alpha._double.header.help       = "alpha";
  alpha._double.required          = 0;
  alpha._double.min               = 1.0;
  alpha._double.max               = DBL_MAX;
  alpha._double.value             = 1e3;

  beta._double.header.type       = opt_double;
  beta._double.header.short_name = 'm';
  beta._double.header.long_name  = "beta";
  beta._double.header.help       = "beta";
  beta._double.required          = 0;
  beta._double.min               = 1.0;
  beta._double.max               = DBL_MAX;
  beta._double.value             = 1e3;
}

int bench_matvec_fill_show_help(void){
  bench_rdgemv_options_initialize();

  opt_show_option(incY);
  opt_show_option(FillY);
  opt_show_option(ScaleY);
  opt_show_option(CondY);
  opt_show_option(alpha);
  opt_show_option(beta);
  return 0;
}

const char* bench_matvec_fill_name(int argc, char** argv){
  bench_rdgemv_options_initialize();

  static char name_buffer[MAX_LINE];
  snprintf(name_buffer, MAX_LINE * sizeof(char), "Benchmark [rdgemv]");
  return name_buffer;
}

int bench_matvec_fill_test(int argc, char** argv, char Order, char TransA, int M, int N, int FillA, double ScaleA, double CondA, int lda, int FillX, double ScaleX, double CondX, int incX, int trials){
  int rc = 0;

  bench_rdgemv_options_initialize();

  opt_eval_option(argc, argv, &incY);
  opt_eval_option(argc, argv, &FillY);
  opt_eval_option(argc, argv, &ScaleY);
  opt_eval_option(argc, argv, &CondY);
  opt_eval_option(argc, argv, &alpha);
  opt_eval_option(argc, argv, &beta);

  util_random_seed();

  rblas_order_t o;
  rblas_transpose_t t;
  int NX;
  int NY;
  char NTransA;
  switch(Order){
    case 'r':
    case 'R':
      o = rblas_Row_Major;
      break;
    default:
      o = rblas_Col_Major;
      break;
  }
  switch(TransA){
    case 'n':
    case 'N':
      t = rblas_No_Trans;
      NX = N;
      NY = M;
      NTransA = 'T';
      break;
    default:
      NX = M;
      NY = N;
      NTransA = 'N';
      t = rblas_Trans;
      break;
  }

  double *A  = util_dmat_alloc(Order, M, N, lda);
  double *X  = util_dvec_alloc(NX, incX);
  double *Y  = util_dvec_alloc(NY, incY._int.value);

  util_dmat_fill(Order, 'n', M, N, A, lda, FillA, ScaleA, CondA);
  util_dvec_fill(NX, X, incX, FillX, ScaleX, CondX);
  util_dvec_fill(NY, Y, incY._int.value, FillY._named.value, ScaleY._double.value, CondY._double.value);
  double *res  = (double*)malloc(NY * incY._int.value * sizeof(double));
  double *ref  = (double*)malloc(NY * incY._int.value * sizeof(double));
  memcpy(ref, Y, NY * incY._int.value * sizeof(double));
  rdgemv(o, t, M, N, A, lda, X, incX, ref, incY._int.value);

  for(int i = 0; i < trials; i++){
    memcpy(res, Y, NY * incY._int.value * sizeof(double));
    time_tic();
    rdgemv(o, t, M, N, A, lda, X, incX, res, incY._int.value);
    time_toc();
  }

  //TODO make generic fold testing
  int fold = 3;
  metric_load_double("time", time_read());
  metric_load_long_long("trials", (long long)trials);
  metric_load_long_long("input", (long long)N * M + N + M);
  metric_load_long_long("output", (long long)NY);
  metric_load_long_long("d_mul", (long long)N * M);
  metric_load_long_long("d_add", (long long)(3 * fold - 2) * N * M);
  metric_load_long_long("d_orb", (long long)fold * N * M);
  metric_dump();

  free(X);
  free(res);
  free(ref);
  return rc;
}

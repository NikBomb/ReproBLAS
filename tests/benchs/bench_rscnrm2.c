#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <reproBLAS.h>
#include "../common/test_opt.h"
#include "../common/test_time.h"
#include "../common/test_metric.h"

#include "bench_vecvec_fill_header.h"

int bench_vecvec_fill_show_help(void){
  return 0;
}

const char* bench_vecvec_fill_name(int argc, char** argv){
  (void)argc;
  (void)argv;
  static char name_buffer[MAX_LINE];
  snprintf(name_buffer, MAX_LINE * sizeof(char), "Benchmark [rscnrm2]");
  return name_buffer;
}

int bench_vecvec_fill_test(int argc, char** argv, int N, int FillX, double RealScaleX, double ImagScaleX, int incX, int FillY, double RealScaleY, double ImagScaleY, int incY, int trials){
  (void)argc;
  (void)argv;
  (void)FillY;
  (void)RealScaleY;
  (void)ImagScaleY;
  (void)incY;
  int rc = 0;
  int i;
  float res = 0.0;

  util_random_seed();

  float complex *X = util_cvec_alloc(N, incX);

  //fill X
  util_cvec_fill(N, X, incX, FillX, RealScaleX, ImagScaleX);

  time_tic();
  for(i = 0; i < trials; i++){
    res = rscnrm2(N, X, incX);
  }
  time_toc();

  //TODO make generic fold testing
  int fold = 3;
  metric_load_double("time", time_read());
  metric_load_float("res", res);
  metric_load_long_long("trials", (long long)trials);
  metric_load_long_long("input", (long long)N);
  metric_load_long_long("output", (long long)1);
  metric_load_long_long("s_mul", (long long)4 * N);
  metric_load_long_long("s_add", (long long)(3 * fold - 2) * 2 * N);
  metric_load_long_long("s_orb", (long long)fold * 2 * N);
  metric_dump();

  free(X);
  return rc;
}

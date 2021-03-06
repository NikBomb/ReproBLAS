#ifndef SAUGSUM_WRAPPER_H
#define SAUGSUM_WRAPPER_H

#include <reproBLAS.h>
#include <binnedBLAS.h>
#include <binned.h>
#include "../../config.h"

#include "../common/test_util.h"

typedef enum wrap_saugsum_func {
  wrap_saugsum_RSSUM = 0,
  wrap_saugsum_RSASUM,
  wrap_saugsum_RSNRM2,
  wrap_saugsum_RSDOT,
  wrap_saugsum_SBSBADD,
  wrap_saugsum_SISADD,
  wrap_saugsum_SISDEPOSIT
} wrap_saugsum_func_t;

typedef float (*wrap_saugsum)(int, int, float*, int, float*, int);
typedef void (*wrap_siaugsum)(int, int, float*, int, float*, int, float_binned*);
static const int wrap_saugsum_func_n_names = 7;
static const char* wrap_saugsum_func_names[] = {"rssum",
                                                "rsasum",
                                                "rsnrm2",
                                                "rsdot",
                                                "sbsbadd",
                                                "sbsadd",
                                                "sbsdeposit"};
static const char* wrap_saugsum_func_descs[] = {"rssum",
                                                "rsasum",
                                                "rsnrm2",
                                                "rsdot",
                                                "sbsbadd",
                                                "sbsadd",
                                                "sbsdeposit"};

float wrap_rssum(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  if(fold == SIDEFAULTFOLD){
    return reproBLAS_ssum(N, x, incx);
  }else{
    return reproBLAS_rssum(fold, N, x, incx);
  }
}

void wrap_sbssum(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  binnedBLAS_sbssum(fold, N, x, incx, z);
}

float wrap_rsasum(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  if(fold == SIDEFAULTFOLD){
    return reproBLAS_sasum(N, x, incx);
  }else{
    return reproBLAS_rsasum(fold, N, x, incx);
  }
}

void wrap_sbsasum(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  binnedBLAS_sbsasum(fold, N, x, incx, z);
}

float wrap_rsnrm2(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  if(fold == SIDEFAULTFOLD){
    return reproBLAS_snrm2(N, x, incx);
  }else{
    return reproBLAS_rsnrm2(fold, N, x, incx);
  }
}

void wrap_sbsssq(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  binnedBLAS_sbsssq(fold, N, x, incx, 0.0, z);
}

float wrap_rsdot(int fold, int N, float *x, int incx, float *y, int incy) {
  if(fold == SIDEFAULTFOLD){
    return reproBLAS_sdot(N, x, incx, y, incy);
  }else{
    return reproBLAS_rsdot(fold, N, x, incx, y, incy);
  }
}

void wrap_sbsdot(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  binnedBLAS_sbsdot(fold, N, x, incx, y, incy, z);
}

float wrap_rsbsbadd(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  float_binned *ires = binned_sballoc(fold);
  float_binned *itmp = binned_sballoc(fold);
  binned_sbsetzero(fold, ires);
  int i;
  for(i = 0; i < N; i++){
    binned_sbsconv(fold, x[i * incx], itmp);
    binned_sbsbadd(fold, itmp, ires);
  }
  float res = binned_ssbconv(fold, ires);
  free(ires);
  free(itmp);
  return res;
}

void wrap_sbsbadd(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  float_binned *itmp = binned_sballoc(fold);
  int i;
  for(i = 0; i < N; i++){
    binned_sbsconv(fold, x[i * incx], itmp);
    binned_sbsbadd(fold, itmp, z);
  }
  free(itmp);
}

float wrap_rsbsadd(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  float_binned *ires = binned_sballoc(fold);
  binned_sbsetzero(fold, ires);
  int i;
  for(i = 0; i < N; i++){
    binned_sbsadd(fold, x[i * incx], ires);
  }
  float res = binned_ssbconv(fold, ires);
  free(ires);
  return res;
}

void wrap_sbsadd(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  int i;
  for(i = 0; i < N; i++){
    binned_sbsadd(fold, x[i * incx], z);
  }
}

float wrap_rsbsdeposit(int fold, int N, float *x, int incx, float *y, int incy) {
  (void)y;
  (void)incy;
  float_binned *ires = binned_sballoc(fold);
  binned_sbsetzero(fold, ires);
  float amax = binnedBLAS_samax(N, x, incx);
  binned_sbsupdate(fold, amax, ires);
  int i;
  int j = 0;
  for(i = 0; i < N; i++){
    if(j >= binned_SBENDURANCE){
      binned_sbrenorm(fold, ires);
      j = 0;
    }
    binned_sbsdeposit(fold, x[i * incx], ires);
    j++;
  }
  binned_sbrenorm(fold, ires);
  float res = binned_ssbconv(fold, ires);
  free(ires);
  return res;
}

void wrap_sbsdeposit(int fold, int N, float *x, int incx, float *y, int incy, float_binned *z) {
  (void)y;
  (void)incy;
  float amax = binnedBLAS_samax(N, x, incx);
  binned_sbsupdate(fold, amax, z);
  int i;
  int j = 0;
  for(i = 0; i < N; i++){
    if(j >= binned_SBENDURANCE){
      binned_sbrenorm(fold, z);
      j = 0;
    }
    binned_sbsdeposit(fold, x[i * incx], z);
    j++;
  }
  binned_sbrenorm(fold, z);
}

wrap_saugsum wrap_saugsum_func(wrap_saugsum_func_t func) {
  switch(func){
    case wrap_saugsum_RSSUM:
      return wrap_rssum;
    case wrap_saugsum_RSASUM:
      return wrap_rsasum;
    case wrap_saugsum_RSNRM2:
      return wrap_rsnrm2;
    case wrap_saugsum_RSDOT:
      return wrap_rsdot;
    case wrap_saugsum_SBSBADD:
      return wrap_rsbsbadd;
    case wrap_saugsum_SISADD:
      return wrap_rsbsadd;
    case wrap_saugsum_SISDEPOSIT:
      return wrap_rsbsdeposit;
  }
  return NULL;
}

wrap_siaugsum wrap_siaugsum_func(wrap_saugsum_func_t func) {
  switch(func){
    case wrap_saugsum_RSSUM:
      return wrap_sbssum;
    case wrap_saugsum_RSASUM:
      return wrap_sbsasum;
    case wrap_saugsum_RSNRM2:
      return wrap_sbsssq;
    case wrap_saugsum_RSDOT:
      return wrap_sbsdot;
    case wrap_saugsum_SBSBADD:
      return wrap_sbsbadd;
    case wrap_saugsum_SISADD:
      return wrap_sbsadd;
    case wrap_saugsum_SISDEPOSIT:
      return wrap_sbsdeposit;
  }
  return NULL;
}

float wrap_saugsum_result(int N, wrap_saugsum_func_t func, util_vec_fill_t FillX, double RealScaleX, double ImagScaleX, util_vec_fill_t FillY, double RealScaleY, double ImagScaleY){
  float small = 1.0 / (1024.0 * 4.0); // 2^-12
  float big   = 1024.0 * 8.0;  // 2^13
  switch(func){
    case wrap_saugsum_RSSUM:
    case wrap_saugsum_SBSBADD:
    case wrap_saugsum_SISADD:
    case wrap_saugsum_SISDEPOSIT:
      switch(FillX){
        case util_Vec_Constant:
          return N * RealScaleX;
        case util_Vec_Mountain:
          return 0;
        case util_Vec_Pos_Inf:
        case util_Vec_Pos_Pos_Inf:
          return RealScaleX * INFINITY;
        case util_Vec_Pos_Neg_Inf:
        case util_Vec_NaN:
        case util_Vec_Pos_Inf_NaN:
        case util_Vec_Pos_Pos_Inf_NaN:
        case util_Vec_Pos_Neg_Inf_NaN:
          return NAN;
        case util_Vec_Pos_Big:
          return (N - 1) * (RealScaleX * small) + RealScaleX * big;
        case util_Vec_Pos_Pos_Big:
          return (N - 2) * (RealScaleX * small) + (RealScaleX * big + RealScaleX * big);
        case util_Vec_Pos_Neg_Big:
          return (N - 2) * (RealScaleX * small);
        case util_Vec_Sine:
          return 0.0;
        default:
          fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX);
          exit(125);
      }

    case wrap_saugsum_RSASUM:
      switch(FillX){
        case util_Vec_Constant:
          return N * fabs(RealScaleX);
        case util_Vec_Pos_Inf:
        case util_Vec_Pos_Pos_Inf:
        case util_Vec_Pos_Neg_Inf:
          return fabs(RealScaleX) * INFINITY;
        case util_Vec_NaN:
        case util_Vec_Pos_Inf_NaN:
        case util_Vec_Pos_Pos_Inf_NaN:
        case util_Vec_Pos_Neg_Inf_NaN:
          return NAN;
        case util_Vec_Pos_Big:
          return (N - 1) * (fabs(RealScaleX) * small) + fabs(RealScaleX) * big;
        case util_Vec_Pos_Pos_Big:
        case util_Vec_Pos_Neg_Big:
          return (N - 2) * (fabs(RealScaleX) * small) + (fabs(RealScaleX) * big + fabs(RealScaleX) * big);
        default:
          fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX);
          exit(125);
      }

    case wrap_saugsum_RSNRM2:
      {
        float new_scale;
        switch(FillX){
          case util_Vec_Constant:
            new_scale = binned_sscale(RealScaleX);
            RealScaleX /= new_scale;
            return sqrtf(N * (RealScaleX * RealScaleX)) * new_scale;
          case util_Vec_Pos_Inf:
          case util_Vec_Pos_Pos_Inf:
          case util_Vec_Pos_Neg_Inf:
            return fabs(RealScaleX) * INFINITY;
          case util_Vec_NaN:
          case util_Vec_Pos_Inf_NaN:
          case util_Vec_Pos_Pos_Inf_NaN:
          case util_Vec_Pos_Neg_Inf_NaN:
            return NAN;
          case util_Vec_Pos_Big:
            new_scale = binned_sscale(RealScaleX * big);
            small *= RealScaleX;
            small /= new_scale;
            big *= RealScaleX;
            big /= new_scale;
            return sqrtf((N - 1) * (small * small) + big * big) * new_scale;
          case util_Vec_Pos_Pos_Big:
          case util_Vec_Pos_Neg_Big:
            new_scale = binned_sscale(RealScaleX * big);
            small *= RealScaleX;
            small /= new_scale;
            big *= RealScaleX;
            big /= new_scale;
            return sqrtf((N - 2) * (small * small) + (big * big + big * big)) * new_scale;
          default:
            fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX);
            exit(125);
        }
      }

    case wrap_saugsum_RSDOT:
      switch(FillX){
        case util_Vec_Mountain:
          switch(FillY){
            case util_Vec_Constant:
              return 0;
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Constant:
          switch(FillY){
            case util_Vec_Constant:
              return N * RealScaleX * RealScaleY;
            case util_Vec_Mountain:
              return 0;
            case util_Vec_Pos_Inf:
            case util_Vec_Pos_Pos_Inf:
              return (RealScaleX * RealScaleY) * INFINITY;
            case util_Vec_Pos_Neg_Inf:
            case util_Vec_NaN:
            case util_Vec_Pos_Inf_NaN:
            case util_Vec_Pos_Pos_Inf_NaN:
            case util_Vec_Pos_Neg_Inf_NaN:
              return NAN;
            case util_Vec_Pos_Big:
              return (N - 1) * (RealScaleX * RealScaleY * small) + RealScaleX * RealScaleY * big;
            case util_Vec_Pos_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small) + (RealScaleX * RealScaleY * big + RealScaleX * RealScaleY * big);
            case util_Vec_Pos_Neg_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small);
            case util_Vec_Sine:
              return 0.0;
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Pos_Inf:
        case util_Vec_Pos_Pos_Inf:
          switch(FillY){
            case util_Vec_Constant:
            case util_Vec_Pos_Inf:
            case util_Vec_Pos_Pos_Inf:
              return (RealScaleX * RealScaleY) * INFINITY;
            case util_Vec_Pos_Neg_Inf:
            case util_Vec_NaN:
            case util_Vec_Pos_Inf_NaN:
            case util_Vec_Pos_Pos_Inf_NaN:
            case util_Vec_Pos_Neg_Inf_NaN:
              return NAN;
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Pos_Neg_Inf:
          switch(FillY){
            case util_Vec_Constant:
            case util_Vec_Pos_Inf:
            case util_Vec_Pos_Pos_Inf:
            case util_Vec_NaN:
            case util_Vec_Pos_Inf_NaN:
            case util_Vec_Pos_Pos_Inf_NaN:
            case util_Vec_Pos_Neg_Inf_NaN:
              return NAN;
            case util_Vec_Pos_Neg_Inf:
              return (RealScaleX * RealScaleY) * INFINITY;
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_NaN:
        case util_Vec_Pos_Inf_NaN:
        case util_Vec_Pos_Pos_Inf_NaN:
        case util_Vec_Pos_Neg_Inf_NaN:
          return NAN;
        case util_Vec_Pos_Big:
          switch(FillY){
            case util_Vec_Constant:
              return (N - 1) * (RealScaleX * RealScaleY * small) + RealScaleX * RealScaleY * big;
            case util_Vec_Pos_Big:
              return (N - 1) * (RealScaleX * RealScaleY * small * small) + RealScaleX * RealScaleY * big * big;
            case util_Vec_Pos_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) + (RealScaleX * RealScaleY * big * small + RealScaleX * RealScaleY * big * big);
            case util_Vec_Pos_Neg_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) - (RealScaleX * RealScaleY * big * small - RealScaleX * RealScaleY * big * big);
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Pos_Pos_Big:
          switch(FillY){
            case util_Vec_Constant:
              return (N - 2) * (RealScaleX * RealScaleY * small) + (RealScaleX * RealScaleY * big + RealScaleX * RealScaleY * big);
            case util_Vec_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) + (RealScaleX * RealScaleY * big * small + RealScaleX * RealScaleY * big * big);
            case util_Vec_Pos_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) + (RealScaleX * RealScaleY * big * big + RealScaleX * RealScaleY * big * big);
            case util_Vec_Pos_Neg_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small);
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Pos_Neg_Big:
          switch(FillY){
            case util_Vec_Constant:
              return (N - 2) * (RealScaleX * RealScaleY * small);
            case util_Vec_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) - (RealScaleX * RealScaleY * big * small - RealScaleX * RealScaleY * big * big);
            case util_Vec_Pos_Pos_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small);
            case util_Vec_Pos_Neg_Big:
              return (N - 2) * (RealScaleX * RealScaleY * small * small) + (RealScaleX * RealScaleY * big * big + RealScaleX * RealScaleY * big * big);
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        case util_Vec_Sine:
          switch(FillY){
            case util_Vec_Constant:
              return 0.0;
            default:
              fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
              exit(125);
          }
        default:
          fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
          exit(125);
      }
    default:
      fprintf(stderr, "ReproBLAS error: unknown result for %s(%s * %g, %s * %g)\n", wrap_saugsum_func_descs[func], util_vec_fill_descs[FillX], RealScaleX, util_vec_fill_descs[FillY], RealScaleY);
      exit(125);
  }
}

float wrap_saugsum_bound(int fold, int N, wrap_saugsum_func_t func, float *X, int incX, float *Y, int incY, float res, float ref){
  switch(func){
    case wrap_saugsum_RSSUM:
    case wrap_saugsum_SBSBADD:
    case wrap_saugsum_SISADD:
    case wrap_saugsum_SISDEPOSIT:
    case wrap_saugsum_RSASUM:
      return binned_sbbound(fold, N, binnedBLAS_samax(N, X, incX), res);
    case wrap_saugsum_RSNRM2:
      {
        float amax = binnedBLAS_samax(N, X, incX);
        float scale = binned_sscale(amax);
        if (amax == 0.0){
          return 0.0;
        }
        return binned_sbbound(fold, N, (amax/scale) * (amax/scale), (res/scale) * (res/scale)) * (scale / (res + ref)) * scale;
      }
    case wrap_saugsum_RSDOT:
      return binned_sbbound(fold, N, binnedBLAS_samaxm(N, X, incX, Y, incY), res);
  }
  fprintf(stderr, "ReproBLAS error: unknown bound for %s\n", wrap_saugsum_func_descs[func]);
  exit(125);
}


#endif

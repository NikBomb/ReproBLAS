#include <indexedBLAS.h>
#include <indexed.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/test_util.h"
#include "../common/test_opt.h"
#include "../common/test_file.h"
#include "rzblas1_wrapper.h"

#include "../common/test_file_header.h"

static opt_option func_type = {._named.header.type       = opt_named,
                               ._named.header.short_name = 'w',
                               ._named.header.long_name  = "w_type",
                               ._named.header.help       = "wrapped function type",
                               ._named.required          = 1,
                               ._named.n_names           = wrap_rzblas1_n_names,
                               ._named.names             = (char**)wrap_rzblas1_names,
                               ._named.descs             = (char**)wrap_rzblas1_descs,
                               ._named.value             = wrap_RZSUM};

static opt_option record    = {._flag.header.type       = opt_flag,
                               ._flag.header.short_name = 'r',
                               ._flag.header.long_name  = "record",
                               ._flag.header.help       = "record the run insted of testing"};

int file_show_help(void){
  opt_show_option(func_type);
  opt_show_option(record);
  return 0;
}

const char* file_name(int argc, char** argv) {
  static char name_buffer[MAX_LINE];

  opt_eval_option(argc, argv, &func_type);
  snprintf(name_buffer, MAX_LINE, "Validate %s external", wrap_rzblas1_names[func_type._named.value]);
  return name_buffer;
}

int file_test(int argc, char** argv, char *fname) {
  int N;
  char ref_fname[MAX_NAME];
  char Iref_fname[MAX_NAME];

  opt_eval_option(argc, argv, &func_type);
  opt_eval_option(argc, argv, &record);

  double complex *X;
  double complex *Y;

  double complex ref;
  double_complex_indexed *Iref = zialloc(DEFAULT_FOLD);
  zisetzero(DEFAULT_FOLD, Iref);

  double complex res;
  double_complex_indexed *Ires = zialloc(DEFAULT_FOLD);
  zisetzero(DEFAULT_FOLD, Ires);

  file_read_vector(fname, &N, (void**)&X, sizeof(double complex));
  Y = util_zvec_alloc(N, 1);
  //fill Y with -i where necessary
  util_zvec_fill(N, Y, 1, util_Vec_Constant, -_Complex_I, 1.0);

  ((char*)file_ext(fname))[0] = '\0';
  snprintf(ref_fname, MAX_NAME, "%s__%s.dat", fname, wrap_rzblas1_names[func_type._named.value]);
  snprintf(Iref_fname, MAX_NAME, "%s__I%s.dat", fname, wrap_rzblas1_names[func_type._named.value]);

  res = (wrap_rzblas1_func(func_type._named.value))(N, X, 1, Y, 1);
  (wrap_ziblas1_func(func_type._named.value))(N, X, 1, Y, 1, Ires);

  if(record._flag.exists){
    ref = res;
    Iref = Ires;

    file_write_vector(ref_fname, 1, &ref, sizeof(ref));
    file_write_vector(Iref_fname, 1, Iref, zisize(DEFAULT_FOLD));
  } else {
    void *data;
    int unused0;
    file_read_vector(ref_fname, &unused0, &data, sizeof(ref));
    ref = *(double complex*)data;
    free(data);
    file_read_vector(Iref_fname, &unused0, &data, zisize(DEFAULT_FOLD));
    free(Iref);
    Iref = data;
    if(ref != res){
      printf("%s(%s) = %g + %gi != %g + %gi\n", wrap_rzblas1_names[func_type._named.value], fname, ZREAL_(res), ZIMAG_(res), ZREAL_(ref), ZIMAG_(ref));
      return 1;
    }
    if(memcmp(&Iref, &Ires, sizeof(Iref)) != 0){
      zziconv_sub(DEFAULT_FOLD, Ires, &res);
      zziconv_sub(DEFAULT_FOLD, Iref, &ref);
      printf("I%s(%s) = %g + %gi != %g + %gi\n", wrap_rzblas1_names[func_type._named.value], fname, ZREAL_(res), ZIMAG_(res), ZREAL_(ref), ZIMAG_(ref));
      printf("Ref I_double_Complex:\n");
      ziprint(DEFAULT_FOLD, Iref);
      printf("\nRes I_double_Complex:\n");
      ziprint(DEFAULT_FOLD, Ires);
      printf("\n");
      return 1;
    }
  }

  free(Iref);
  free(Ires);
  free(X);
  free(Y);
  return 0;
}

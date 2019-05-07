#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
//#include <binnedMPI.h>
#include <binned.h>
#include <mpi.h>


#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

static void binnedMPI_dbdbadd_3(void *invec, void *inoutvec, int *len, MPI_Datatype* datatype){
  binned_dbdbaddv(3, *len, (double_binned*)invec, 1, (double_binned*)inoutvec, 1);
}

MPI_Op binnedMPI_DBDBADD() {
    printf("Defining new Operation\n");
    MPI_Op myOp;
    int rc;
    rc = MPI_Op_create(&binnedMPI_dbdbadd_3, 1, &myOp);
    if(rc != MPI_SUCCESS){
      if (rc == MPI_ERR_TYPE) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_Op_Create error: MPI_ERR_TYPE\n", __FILE__, __LINE__);
      } else if (rc == MPI_ERR_COUNT) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_Op_Create error: MPI_ERR_COUNT\n", __FILE__, __LINE__);
      } else if (rc == MPI_ERR_INTERN) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_Op_Create error: MPI_ERR_INTERN\n", __FILE__, __LINE__);
      } else {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_Op_Create error: %d\n", __FILE__, __LINE__, rc);
      }
      MPI_Abort(MPI_COMM_WORLD, rc);
      return 0;
    }
    return myOp;
}

MPI_Datatype binnedMPI_DOUBLE_BINNED_mine(const int fold){
  int rc;
  
  MPI_Datatype myBinned;
  
  printf("Defining new type\n");
      rc = MPI_Type_contiguous(binned_dbnum(fold), MPI_DOUBLE, &myBinned);
    if(rc != MPI_SUCCESS){
      if (rc == MPI_ERR_TYPE) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_contiguous error: MPI_ERR_TYPE\n", __FILE__, __LINE__);
      } else if (rc == MPI_ERR_COUNT) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_contiguous error: MPI_ERR_COUNT\n", __FILE__, __LINE__);
      } else if (rc == MPI_ERR_INTERN) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_contiguous error: MPI_ERR_INTERN\n", __FILE__, __LINE__);
      } else {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_contiguous error: %d\n", __FILE__, __LINE__, rc);
      }
      MPI_Abort(MPI_COMM_WORLD, rc);
      return 0;
    }
    rc = MPI_Type_commit(&myBinned);
    if(rc != MPI_SUCCESS){
      if (rc == MPI_ERR_TYPE) {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_commit error: MPI_ERR_TYPE\n", __FILE__, __LINE__);
      } else {
        fprintf(stderr, "[%s.%d] ReproBLAS error: MPI_Type_commit error: %d\n", __FILE__, __LINE__, rc);
      }
      MPI_Abort(MPI_COMM_WORLD, rc);
      return 0;
    }
  return myBinned;
}

static struct timeval start;
static struct timeval end;

void tic(void){
  gettimeofday( &start, NULL );
}

double toc(void){
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

int main(int argc, char** argv){
  int rank;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 24000000;
  // Let's make n a multiple of size, for simplicity
  //n = n + size - (n % size); 
  int local_n = n / size;

  double *x = NULL;
  double *x_shuffled = NULL;
  double *local_x = malloc(local_n * sizeof(double));
  double *local_x_shuffled = malloc(local_n * sizeof(double));

  if(rank == 0){
    x = malloc(n * sizeof(double));
    x_shuffled = malloc(n * sizeof(double));

    // Set x to be a sine wave
    for(int i = 0; i < n; i++){
      x[i] = sin(2 * M_PI * (i / (double)n - 0.5));
    }

    // Shuffle x into x_shuffled
    for(int i = 0; i < n; i++){
      x_shuffled[i] = x[i];
    }
    double t;
    int r;
    for(int i = 0; i < n; i++){
      r = rand();
      t = x_shuffled[i];
      x_shuffled[i] = x_shuffled[i + (r % (n - i))];
      x_shuffled[i + (r % (n - i))] = t;
    }
  }

  // Distribute the x arrays among processors
  MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(x_shuffled, local_n, MPI_DOUBLE, local_x_shuffled, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sum;
  double sum_shuffled;
  double elapsed_time;

  if(rank == 0){
    printf("Sum of sin(2* M_PI * (i / (double)n - 0.5)).  n = %d.\n\n", n);

    // Make a header
    printf("%15s : Time (s) : |Sum - Sum of Shuffled| = ?\n", "Sum Method");
  }

  // First, we sum x using double precision
  double local_sum;
  tic();
  local_sum = 0;
  sum = 0;
  // Performing local summation
  for(int i = 0; i < local_n; i++){
    local_sum += local_x[i];
  }
  // Reduce
  MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  elapsed_time = toc();

  // Next, we sum the shuffled x
  local_sum = 0;
  sum_shuffled = 0;
  for(int i = 0; i < local_n; i++){
    local_sum += local_x_shuffled[i];
  }
  MPI_Reduce(&local_sum, &sum_shuffled, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank == 0){
    printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "double", elapsed_time, sum, sum_shuffled, fabs(sum - sum_shuffled));
  }

  // We now sum x using binned summation
  tic();
  double_binned *isum = NULL;
  double_binned *local_isum = binned_dballoc(3);
  binned_dbsetzero(3, local_isum);
  if(rank == 0){
    isum = binned_dballoc(3);
    binned_dbsetzero(3, isum);
  }
  // Performing local summation
  binnedBLAS_dbdsum(3, local_n, local_x, 1, local_isum);
  // Reduce
  MPI_Reduce(local_isum, isum, 1, binnedMPI_DOUBLE_BINNED_mine(3), binnedMPI_DBDBADD(), 0, MPI_COMM_WORLD);
  if(rank == 0){
     sum = binned_ddbconv(3, isum);
  }
  elapsed_time = toc();
  

  // Next, we sum the shuffled x
  binned_dbsetzero(3, local_isum);
  if(rank == 0){
    binned_dbsetzero(3, isum);
  }
  binnedBLAS_dbdsum(3, local_n, local_x_shuffled, 1, local_isum);
  MPI_Reduce(local_isum, isum, 1,  binnedMPI_DOUBLE_BINNED_mine(3), binnedMPI_DBDBADD(), 0, MPI_COMM_WORLD);
  if(rank == 0){
    sum_shuffled = binned_ddbconv(3, isum);
  }

  if(rank == 0){
    free(isum);
  }
  free(local_isum);

  if(rank == 0){
    printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "binnedMPI_DBDBADD", elapsed_time, sum, sum_shuffled, fabs(sum - sum_shuffled));
  }

  if(rank == 0){
    free(x);
    free(x_shuffled);
  }
  free(local_x);
  free(local_x_shuffled);
  MPI_Finalize();
}

#include "gpuError.h"
#include <stdio.h>
#include <stdlib.h>

void hipError(
      hipError_t err,
      const char *file,
      int line)
{
   if (err != hipSuccess) {
       printf("*** CUDA Error in %s at line %i\n", file, line );
       printf("%s\n",hipGetErrorString( err ));
       exit( EXIT_FAILURE );
   }
}

void cublasError(
      hipblasStatus_t err,
      const char *file,
      int line)
{
   if (err != HIPBLAS_STATUS_SUCCESS)  {
      printf("*** CUBLAS Error in %s at line %i\n", file, line);
      switch(err) {
         case HIPBLAS_STATUS_NOT_INITIALIZED:
            printf ("CUBLAS was not initialized\n"); break;
         case HIPBLAS_STATUS_ALLOC_FAILED:
            printf ("CUBLAS allociation failed\n"); break;
         case HIPBLAS_STATUS_INVALID_VALUE:
            printf ("CUBLAS unsupported numerical value passed\n"); break;
         case HIPBLAS_STATUS_MAPPING_ERROR:
            printf ("CUBLAS access to GPU memory space failed\n"); break;
         case HIPBLAS_STATUS_EXECUTION_FAILED:
            printf ("CUBLAS GPU program failed to execute\n"); break;
         case HIPBLAS_STATUS_INTERNAL_ERROR:
            printf ("CUBLAS internal operation failed\n"); break;
         case HIPBLAS_STATUS_ARCH_MISMATCH:
            printf ("CUBLAS architecture mismatch\n"); break;
         // This will never happen due to if statement, but is here to remove
         // compiler warning
         case HIPBLAS_STATUS_SUCCESS:
            printf ("I WAS SUCCESFULL\n"); break;
      }
      exit (EXIT_FAILURE);
   }
}

void checkError(
      const char* file,
      int line)
{
   hipError_t error = hipGetLastError();
   if (error != hipSuccess) {
      printf("*** CUDA Error in %s, line %i\n",file, line);
      printf("%s\n", hipGetErrorString(error));
      exit (EXIT_FAILURE);
   }
}

void promptError(
      const char* condition,
      const char* file,
      int line)
{
   printf("\n***ERROR in %s, line %i !\n",file,line);
   printf("%s check failed!\n\n",condition);
   exit( EXIT_FAILURE );
}

void promptError(
      const char* condition,
      const char* file,
      const char* info,
      int line)
{
   printf("\n***ERROR in %s, line %i !\n",file,line);
   printf("%s check failed!\n\n",condition);
   printf("Additional information:  %s\n\n",info);
   exit( EXIT_FAILURE );
}

void promptErrorVar(
      const char* condition,
      const int val,
      const char* file,
      int line)
{
   printf("\n***ERROR in %s, line %i !\n",file,line);
   printf("%s, Var = %i\n\n",condition, val);
   exit( EXIT_FAILURE );
}

void promptErrorVars(
      const char* condition,
      const int val1,
      const int val2,
      const char* file,
      int line)
{
   printf("\n***ERROR in %s, line %i !\n",file,line);
   printf("%s, Var1 = %i, Var2 = %i\n\n",condition, val1, val2);
   exit( EXIT_FAILURE );
}

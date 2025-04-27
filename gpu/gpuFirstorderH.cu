// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iomanip>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
// includes, project
#include "gpuMacro.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "gpuFirstorderH.h"

#define HIP_CHECK(command)                                                                                                \
    {                                                                                                                     \
        \ 
    hipError_t status = command;                                                                                          \
        if (status != hipSuccess)                                                                                         \
        {                                                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " Error: Hip reports " << hipGetErrorString(status) << std::endl; \
            std::abort();                                                                                                 \
        }                                                                                                                 \
    }
#define CHECK_MAGMA_ERROR(err)                                                                             \
    do                                                                                                     \
    {                                                                                                      \
        if ((err) != MAGMA_SUCCESS)                                                                        \
        {                                                                                                  \
            fprintf(stdout, "Error at line %d in file %s: %s\n", __LINE__, __FILE__, magma_strerror(err)); \
            exit(EXIT_FAILURE);                                                                            \
        }                                                                                                  \
    } while (0)

/*
本程序实现对于H中矩阵乘法dgemm的vbatch以及多流化操作
乘法操作为A*B=C
*/
namespace FirstOrderH
{
    double gflops = 0, magma_perf = 0, magma_time = 0;

    // 最大mnk
    int max_M = 0, max_N = 0, max_K = 0;
    // 总数组的大小
    int total_size_A = 0, total_size_B = 0, total_size_C = 0;
    // 总数组cpu
    double *h_A, *h_B, *h_C;
    // 总数组gpu
    double *d_A, *d_B, *d_C;

    double alpha = 1.0;
    double beta = 0.0;

    // cpu和gpu的指针数组
    double **h_A_array = NULL;
    double **h_B_array = NULL;
    double **h_C_array = NULL;
    double **d_A_array = NULL;
    double **d_B_array = NULL;
    double **d_C_array = NULL;

    // cpu和gpu维度数组，同时也可以作为主维数组
    int *h_M, *h_N;
    int *d_M, *d_N;

    // for memcpy
    double *ptr_hA;
    double *ptr_hB;

    // map
    int *map_all;
    int *d_map_all;
    int SUM = 0;
    int indexid = 0;
    // stream
    int streamNum = 20;
    magma_queue_t queue[20];
    hipStream_t stream[20];

    // 用于记录每个流对应的小batchsize
    int *batchSize;
    // 用于记录每个小的batch的起始位置
    int *indexpos;
    // 用于记录每个小batch的矩阵的起始位置
    int *indexmatA;
    int *indexmatB;
    int *indexmatC;
    // 用于记录每个小batch的矩阵的大小
    int *matsizeA;
    int *matsizeB;
    int *matsizeC;
    std::ofstream outfile;
    std::ofstream outfile1;
    std::ofstream outfile2;
    int myrank;
    // std::string filename = "dcuResultH.csv";

}

void FORTRAN(H_create_gpu)(int batchCount, int total_size_A_dev, int total_size_B_dev, int total_size_C_dev, int n_max_compute_ham)
{
    using namespace FirstOrderH;
    // size arrays on the GPU should be at least of size (batchCount+1)
    CHECK_MAGMA_ERROR(magma_imalloc(&d_M, batchCount + 1));
    CHECK_MAGMA_ERROR(magma_imalloc(&d_N, batchCount + 1));

    // pointer arrays gpu 在多流情况下只接收一部分的大小
    CHECK_MAGMA_ERROR(magma_malloc((void **)&d_A_array, batchCount * sizeof(double *)));
    CHECK_MAGMA_ERROR(magma_malloc((void **)&d_B_array, batchCount * sizeof(double *)));
    CHECK_MAGMA_ERROR(magma_malloc((void **)&d_C_array, batchCount * sizeof(double *)));

    CHECK_MAGMA_ERROR(magma_dmalloc(&d_A, total_size_A_dev));
    CHECK_MAGMA_ERROR(magma_dmalloc(&d_B, total_size_B_dev));
    CHECK_MAGMA_ERROR(magma_dmalloc(&d_C, total_size_C_dev));
    // for mapping
    // CHECK_MAGMA_ERROR(magma_imalloc(&d_map_all, batchCount * n_max_compute_ham));
}

void FORTRAN(prepare_for_vbatched)(
    int *rank,                                                                     // mpi rank for set device
    int *batchId, int *batchCount,                                                 // for malloc and control
    int *n_compute_c, int *n_points, int *n_max_compute_ham, int *n_max_batchsize) // get matrix mnk info

{
    using namespace FirstOrderH;
    int id = (*batchId) - 1;
    if (*batchId == 1)
    {
        // 分配设备
        myrank = (*rank);
        CHECK_MAGMA_ERROR(magma_init());
        magma_setdevice((*rank) % 4);
        int deviceid;
        magma_getdevice(&deviceid);
        std::cout << "my mpi id is: " << *rank << " my device id is: " << deviceid << "\n";
        // 为多流申请锁页内存
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&h_M, *batchCount));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&h_N, *batchCount));

        // pointer arrays
        CHECK_MAGMA_ERROR(magma_malloc_pinned((void **)&h_A_array, (*batchCount + 1) * sizeof(double *)));
        CHECK_MAGMA_ERROR(magma_malloc_pinned((void **)&h_B_array, (*batchCount + 1) * sizeof(double *)));
        CHECK_MAGMA_ERROR(magma_malloc_pinned((void **)&h_C_array, (*batchCount + 1) * sizeof(double *)));
        // 预先分配的大内存，之后再分配锁页小内存
        int sizeAB = (*batchCount) * (*n_max_batchsize) * (*n_max_compute_ham);
        CHECK_MAGMA_ERROR(magma_dmalloc_cpu(&h_A, sizeAB));
        CHECK_MAGMA_ERROR(magma_dmalloc_cpu(&h_B, sizeAB));
        // for mapping
        CHECK_MAGMA_ERROR(magma_imalloc_cpu(&map_all, (*batchCount) * (*n_max_compute_ham)));
    }
    h_M[id] = *n_compute_c;
    h_N[id] = *n_points;
    // for mapping
    SUM += (*n_compute_c);
    // h_M = n_compute_c ,h_N = n_points 但注意在H中M = h_M , N=h_M, K=h_N

    max_M = max(max_M, h_M[id]);
    max_N = max(max_N, h_M[id]);
    max_K = max(max_K, h_N[id]);

    // // 用于计算GFLOPs
    // gflops += 2 * h_M[id] * h_N[id] * h_M[id] / 1e9;
    // 拷贝数组元素
    // memcpy(h_A + total_size_A, contract, h_M[id] * h_N[id] * sizeof(double));
    // memcpy(h_B + total_size_B, wave_t, h_N[id] * (*n_max_compute_ham) * sizeof(double));
    // 计算大小及offset
    total_size_A += h_M[id] * h_N[id];
    total_size_B += h_N[id] * (*n_max_compute_ham);
    total_size_C += h_M[id] * h_M[id];
    // 在这里已经获得了想要的各种值
    if (*batchId == *batchCount)
    {
        HIP_CHECK(hipHostRegister(h_A, total_size_A * sizeof(double), hipHostRegisterPortable));
        HIP_CHECK(hipHostRegister(h_B, total_size_B * sizeof(double), hipHostRegisterPortable));
        CHECK_MAGMA_ERROR(magma_dmalloc_pinned(&h_C, total_size_C));
        // 创建gpu内存空间
        H_create_gpu_(*batchCount, total_size_A, total_size_B, total_size_C, *n_max_compute_ham);
        // 提前分配用于内存复制的主机指针数组
        h_A_array[0] = d_A;
        h_B_array[0] = d_B;
        h_C_array[0] = d_C;
        for (int i = 1; i < *batchCount + 1; i++)
        {
            h_A_array[i] = h_A_array[i - 1] + h_M[i - 1] * h_N[i - 1];
            h_B_array[i] = h_B_array[i - 1] + (*n_max_compute_ham) * h_N[i - 1];
            h_C_array[i] = h_C_array[i - 1] + h_M[i - 1] * h_M[i - 1];
        }
    }
}

void FORTRAN(get_vbatched_array)(
    int *batchId, // for malloc and control
    int *n_max_compute_ham,
    double *contract, // matrix A
    double *wave_t)   // matrix B
{
    using namespace FirstOrderH;
    if (*batchId == 1)
    {
        ptr_hA = h_A;
        ptr_hB = h_B;
    }
    int id = *batchId - 1;

    for (int cnt = 0; cnt < h_M[id] * h_N[id]; cnt++)
    {
        *ptr_hA = *(contract + cnt);
        ptr_hA++;
    }
    // memcpy(ptr_hA, contract, h_M[id] * h_N[id] * sizeof(double));
    // memcpy(ptr_hB, wave_t, h_N[id] * (*n_max_compute_ham) * sizeof(double));
    for (int cnt = 0; cnt < (*n_max_compute_ham) * h_N[id]; cnt++)
    {
        *ptr_hB = *(wave_t + cnt);
        ptr_hB++;
    }

    // if (*batchId == 1)
    // {
    //     std::string filename;
    //     filename = "dcuResultC.csv";
    //     outfile.open(filename.c_str(), std::ios::out | std::ios::app);
    //     outfile << "------------------cpuA1 is--------------\n";
    //     for (int cnt = 0; cnt < h_M[id] * h_N[id]; cnt++)
    //     {
    //         outfile << std::fixed << std::setprecision(5) << *(contract + cnt) << ",";
    //     }
    //     outfile << "\n";
    //     // outfile << "------------------dcuA1 is--------------\n";
    //     // for (int cnt = 0; cnt < h_M[id] * h_N[id]; cnt++)
    //     // {
    //     //     outfile << h_A[cnt] << ",";
    //     // }
    //     // outfile << "\n";
    //     outfile.close();
    // }
}

void FORTRAN(get_map_array)(int *map, int *n_compute_c, int *n_max_compute_ham, int *batchId, int *batchCount)
{
    using namespace FirstOrderH;
    if ((*batchId) == 1)
    {
        indexid = 0;
    }
    for (int i = 0; i < (*n_compute_c); i++)
    {
        *(map_all + indexid) = *(map + i);
        indexid++;
    }
    // indexid += ((*n_max_compute_ham) - (*n_compute_c));
    indexid = (*batchId) * (*n_max_compute_ham);
}

void FORTRAN(index_back_nbp)(int *n_max_compute_ham, int *batchCount, double *first_order_H, int *mpi_id)
{
    using namespace FirstOrderH;
    // int j = 0;
    // int k = 0;
    double *my_h_C = h_C;
    // 用于切换不同map
    for (int i = 0; i < (*batchCount); i++)
    {

        int id = i * (*n_max_compute_ham);
        int *map_local = map_all + id;
        int my_n_compute_c = h_M[i];
        for (int j = 0; j < my_n_compute_c; j++)
        {
            int i_off = (map_local[j] * (map_local[j] - 1)) / 2;
            for (int k = 0; k <= j; k++)
            {
                // 插入大矩阵的位置
                int insertId = map_local[k] + i_off - 1;
                // 上三角部分id
                int uploId = my_n_compute_c * j + k;
                // int uploId = j * my_n_compute_c + k - j * (j + 1) / 2;
                first_order_H[insertId] += my_h_C[uploId];
            }
        }
        my_h_C += my_n_compute_c * my_n_compute_c;
    }
}

void FORTRAN(perform_vbatched)(int *batchCount, int *n_max_compute_ham, int *first_scf)
{
    // 开始创建多流
    using namespace FirstOrderH;
    // 设置设备id
    int deviceid;
    magma_getdevice(&deviceid);
    if (*first_scf == 1)
    {
        // 用于记录每个流对应的小batchsize
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&batchSize, streamNum));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&indexpos, streamNum));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&matsizeA, streamNum));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&matsizeB, streamNum));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&matsizeC, streamNum));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&indexmatA, streamNum + 1));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&indexmatB, streamNum + 1));
        CHECK_MAGMA_ERROR(magma_imalloc_pinned(&indexmatC, streamNum + 1));
        indexmatA[0] = 0;
        indexmatB[0] = 0;
        indexmatC[0] = 0;
        for (int streamId = 0; streamId < streamNum; streamId++)
        {
            hipStreamCreate(&stream[streamId]);
            // magma_queue_create(world_rank, &queue[streamId]);
            magma_queue_create_from_hip(deviceid, stream[streamId], NULL, NULL, &queue[streamId]);
            batchSize[streamId] = (*batchCount) / streamNum;
            indexpos[streamId] = streamId * batchSize[streamId];
            int begin = indexpos[streamId];
            // 除不尽的流则对最后一个流的batch大小进行补充
            if (streamId == streamNum - 1)
            {
                batchSize[streamId] += (*batchCount) % streamNum;
            }
            int end = indexpos[streamId] + batchSize[streamId];
            matsizeA[streamId] = h_A_array[end] - h_A_array[begin];
            matsizeB[streamId] = h_B_array[end] - h_B_array[begin];
            matsizeC[streamId] = h_C_array[end] - h_C_array[begin];
            indexmatA[streamId + 1] = indexmatA[streamId] + matsizeA[streamId];
            indexmatB[streamId + 1] = indexmatB[streamId] + matsizeB[streamId];
            indexmatC[streamId + 1] = indexmatC[streamId] + matsizeC[streamId];
        }
        for (int streamId = 0; streamId < streamNum; streamId++)
        {
            // 设置每个流计算的大小，其中不可整除的部分放到最后一个流中

            // 将主机端的维度和主维矩阵复制到设备端,多流下只复制一部分
            magma_setvector_async(batchSize[streamId], sizeof(int), h_M + indexpos[streamId], 1, d_M + indexpos[streamId], 1, queue[streamId]);
            magma_setvector_async(batchSize[streamId], sizeof(int), h_N + indexpos[streamId], 1, d_N + indexpos[streamId], 1, queue[streamId]);

            // 把设备端的总矩阵指针放到主机端的指针数组中，再复制到设备端

            magma_setvector_async(batchSize[streamId], sizeof(double *), h_A_array + indexpos[streamId], 1, d_A_array + indexpos[streamId], 1, queue[streamId]);
            magma_setvector_async(batchSize[streamId], sizeof(double *), h_B_array + indexpos[streamId], 1, d_B_array + indexpos[streamId], 1, queue[streamId]);
            magma_setvector_async(batchSize[streamId], sizeof(double *), h_C_array + indexpos[streamId], 1, d_C_array + indexpos[streamId], 1, queue[streamId]);
        }
    }

    for (int streamId = 0; streamId < streamNum; streamId++)
    {
        // 设置每个流计算的大小，其中不可整除的部分放到最后一个流中

        // 将多个小矩阵的复制改为大矩阵的复制
        magma_setvector_async(matsizeA[streamId], sizeof(double), h_A + indexmatA[streamId], 1, d_A + indexmatA[streamId], 1, queue[streamId]);
        magma_setvector_async(matsizeB[streamId], sizeof(double), h_B + indexmatB[streamId], 1, d_B + indexmatB[streamId], 1, queue[streamId]);

        magmablas_dgemm_vbatched_max_nocheck(MagmaTrans, MagmaNoTrans,
                                             d_M + indexpos[streamId], d_M + indexpos[streamId], d_N + indexpos[streamId],
                                             alpha, d_A_array + indexpos[streamId], d_N + indexpos[streamId],
                                             d_B_array + indexpos[streamId], d_N + indexpos[streamId],
                                             FirstOrderH::beta, d_C_array + indexpos[streamId], d_M + indexpos[streamId],
                                             batchSize[streamId], max_M, max_N, max_K,
                                             queue[streamId]);
        magma_getvector_async(matsizeC[streamId], sizeof(double), d_C + indexmatC[streamId], 1, h_C + indexmatC[streamId], 1, queue[streamId]);
    }
    hipDeviceSynchronize();
}

void FORTRAN(free_spaces)()
{
    using namespace FirstOrderH;
    // 锁页内存的销毁
    hipHostUnregister(h_A);
    hipHostUnregister(h_B);
    magma_free_pinned(h_C);
    magma_free_pinned(h_M);
    magma_free_pinned(h_N);
    magma_free_pinned(h_A_array);
    magma_free_pinned(h_B_array);
    magma_free_pinned(h_C_array);
    magma_free_cpu(h_A);
    magma_free_cpu(h_B);
    magma_free_cpu(map_all);
    // 释放new出来的内存
    magma_free_pinned(batchSize);
    magma_free_pinned(indexpos);
    magma_free_pinned(indexmatA);
    magma_free_pinned(indexmatB);
    magma_free_pinned(indexmatC);
    magma_free_pinned(matsizeA);
    magma_free_pinned(matsizeB);
    magma_free_pinned(matsizeC);

    magma_free(d_A);
    magma_free(d_B);
    magma_free(d_C);
    magma_free(d_M);
    magma_free(d_N);
    magma_free(d_A_array);
    magma_free(d_B_array);
    magma_free(d_C_array);
    // magma_free(d_map_all);

    for (int streamId = 0; streamId < streamNum; streamId++)
    {
        hipStreamDestroy(stream[streamId]);
        magma_queue_destroy(queue[streamId]);
    }
    CHECK_MAGMA_ERROR(magma_finalize());
}

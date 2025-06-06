#include "hip_util_mpi.h"
#include "gpuVars.h"
#include "gpuUtils.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <vector>
#include "gpuAll.h"
#include "gpuSum.h"
#include <fstream>
#include <unistd.h>
#include <ctime>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <functional>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mpi.h>
#include <functional>
#include <numeric>
#define LOCAL_SIZE_H 256
#define MERGE_BATCH 3

extern void mpi_sync();
extern int MV(mpi_tasks, myid);
extern int MV(opencl_util, mpi_platform_relative_id);
extern int MV(load_balancing, my_batch_off);

#define myid MV(mpi_tasks, myid)
#define mpi_platform_relative_id MV(opencl_util, mpi_platform_relative_id)
#define number_of_files 1
#define number_of_kernels 2
#define DEVICE_ID 0
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
// const int S = 128;
const int S = 64;
const int NBATCHMAX = 100010;
int count_batches[S] = {0};
int block[NBATCHMAX];
int count_batches_rho[S] = {0};
int block_rho[NBATCHMAX];
std::unordered_map<std::string, double> time_map;
double *center_all_batches;
double gpu_batches_times_RHO = 0;
double gpu_batches_times_H = 0;
int h_outer_cnt = 0;
int rho_outer_cnt = 0;
int sumup_outer_cnt = 0;
double h_write_time = 0;
double rho_write_time = 0;
double sumup_write_time = 0;


std::vector<int> reduceMatrix(int *diverge_matrix, int row_size)
{
    std::vector<int> reduced_matrix(4 * row_size, 0);

    // 计算各部分的起始索引和大小
    int part1_size = n_max_compute_ham;
    int part2_4_size = n_max_compute_atoms * n_basis_fns;
    int part5_6_size = n_max_compute_ham;
    int part7_11_size = n_max_compute_atoms;

    std::vector<int> part_starts = {
        0,
        part1_size,
        part1_size + part2_4_size,
        part1_size + 2 * part2_4_size,
        part1_size + 3 * part2_4_size,
        part1_size + 3 * part2_4_size + part5_6_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 2 * part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 3 * part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 4 * part7_11_size};

    for (int group = 0; group < 4; ++group)
    {
        int group_start = group * 64;
        int group_end = group_start + 64;

        // 对每一部分进行操作
        for (int col = 0; col < row_size; ++col)
        {
            bool has_one = false;
            for (int row = group_start; row < group_end; ++row)
            {
                if (diverge_matrix[row * row_size + col] == 1)
                {
                    has_one = true;
                    break;
                }
            }
            reduced_matrix[group * row_size + col] = has_one ? 1 : 0;
        }
    }

    return reduced_matrix;
}

std::vector<int> reduceMatrixSecondStep(const std::vector<int> &first_step_matrix, int row_size)
{
    // 假设部分的起始索引和大小已经定义
    int part1_size = n_max_compute_ham;
    int part2_4_size = n_max_compute_atoms * n_basis_fns;
    int part5_6_size = n_max_compute_ham;
    int part7_11_size = n_max_compute_atoms;

    std::vector<int> part_starts = {
        0,
        part1_size,
        part1_size + part2_4_size,
        part1_size + 2 * part2_4_size,
        part1_size + 3 * part2_4_size,
        part1_size + 3 * part2_4_size + part5_6_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 2 * part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 3 * part7_11_size,
        part1_size + 3 * part2_4_size + 2 * part5_6_size + 4 * part7_11_size};

    std::vector<int> reduced_matrix(4 * 11, 0);

    for (int group = 0; group < 4; ++group)
    {
        int group_start_index = group * row_size;
        int group_end_index = group_start_index + row_size; // 每组的结束索引

        for (int part = 0; part < 11; ++part)
        {
            int part_start_index = group_start_index + part_starts[part];
            int part_end_index = (part == 10) ? group_end_index : group_start_index + part_starts[part + 1];
            reduced_matrix[group * 11 + part] = std::accumulate(first_step_matrix.begin() + part_start_index, first_step_matrix.begin() + part_end_index, 0);
        }
    }

    return reduced_matrix;
}

void get_info_(double *centers)
{
    center_all_batches = centers;
}


// void output_times_fortran_h_(double *time_h_all, double *time_hf, double *time_comm, double *time_pre, double *time_others)
// {
//     std::ofstream outFileHF;
//     std::string filename;
//     outFileHF.open("Htimes_outer.csv", std::ios::app);
//     if (outFileHF.is_open() && h_outer_cnt >= 2)
//     {
//         if (myid == 0)
//         {
//             outFileHF << "myid"
//                       << ","
//                       << "h_outer_cnt"
//                       << ","
//                       << "time_comm"
//                       << ","
//                       << "time_pre"
//                       << ","
//                       << "time_hf"
//                       << ","
//                       << "time_others"
//                       << ","
//                       << "time_h_write"
//                       << ","
//                       << "time_h_all" << std::endl;
//             outFileHF << myid << "," << h_outer_cnt << "," << *time_comm << "," << *time_pre << "," << *time_hf << "," << *time_others << "," << h_write_time << "," << *time_h_all << std::endl;
//             outFileHF.close();
//         }
//         else
//         {
//             sleep(1);
//             outFileHF << myid << "," << h_outer_cnt << "," << *time_comm << "," << *time_pre << "," << *time_hf << "," << *time_others << "," << h_write_time << "," << *time_h_all << std::endl;
//             outFileHF.close();
//         }
//     }
//     // else
//     // {
//     //     std::cout << "Failed to open file." << std::endl;
//     // }
//     h_outer_cnt++;
// }

void output_times_fortran_h_(double *time_h_all, double *time_hf, double *time_comm, double *time_pre, double *time_others)
{
    double time_cpscf, time_rho, time_sumup, time_h, time_dm;
    time_cpscf = *time_h_all;
    time_rho = *time_hf;
    time_sumup = *time_comm;
    time_h = *time_pre;
    time_dm = *time_others;
    std::ofstream outFileHF;
    std::string filename;
    outFileHF.open("CPSCF_time.csv", std::ios::app);
    if (outFileHF.is_open() && h_outer_cnt >= 1)
    {
        if (myid == 0)
        {
            outFileHF << "myid"
                      << ","
                      << "h_outer_cnt"
                      << ","
                      << "time_cpscf"
                      << ","
                      << "time_rho"
                      << ","
                      << "time_sumup"
                      << ","
                      << "time_h"
                      << ","
                      << "time_dm"
                      << std::endl;
            outFileHF << myid << "," << h_outer_cnt << "," << time_cpscf << "," << time_rho << "," << time_sumup << "," << time_h << "," << time_dm << std::endl;
            outFileHF.close();
        }
        else
        {
            sleep(1);
            outFileHF << myid << "," << h_outer_cnt << "," << time_cpscf << "," << time_rho << "," << time_sumup << "," << time_h << "," << time_dm << std::endl;
            outFileHF.close();
        }
    }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    h_outer_cnt++;
}

void output_times_fortran_rho_(double *time_h_all, double *time_hf, double *time_comm, double *time_pre, double *time_others)
{
    std::ofstream outFileHRHO;
    std::string filename;
    outFileHRHO.open("RHOtimes_outer.csv", std::ios::app);
    if (outFileHRHO.is_open() && rho_outer_cnt >= 1)
    {
        if (myid == 0)
        {
            outFileHRHO << "myid"
                        << ","
                        << "rho_outer_cnt"
                        << ","
                        << "time_comm"
                        << ","
                        << "time_pre"
                        << ","
                        << "time_rhof"
                        << ","
                        << "time_others"
                        << ","
                        << "time_rho_write"
                        << ","
                        << "time_rho_all" << std::endl;
            outFileHRHO << myid << "," << rho_outer_cnt << "," << *time_comm << "," << *time_pre << "," << *time_hf << "," << *time_others << "," << rho_write_time << "," << *time_h_all << std::endl;
            outFileHRHO.close();
        }
        else
        {
            sleep(1);
            outFileHRHO << myid << "," << rho_outer_cnt << "," << *time_comm << "," << *time_pre << "," << *time_hf << "," << *time_others << "," << rho_write_time << "," << *time_h_all << std::endl;
            outFileHRHO.close();
        }
    }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    rho_outer_cnt++;
}

void output_times_fortran_sumup_(double *time_up, double *time_sum, double *time_all)
{
    std::ofstream outFileHSUMUP;
    std::string filename;
    outFileHSUMUP.open("SUMUPtimes_outer.csv", std::ios::app);
    if (outFileHSUMUP.is_open() && sumup_outer_cnt >= 1)
    {
        if (myid == 0)
        {
            outFileHSUMUP << "myid"
                          << ","
                          << "sumup_outer_cnt"
                          << ","
                          << "time_up"
                          << ","
                          << "time_sum"
                          << ","
                          << "time_sumup_write"
                          << ","
                          << "time_all"
                          << std::endl;
            outFileHSUMUP << myid << "," << sumup_outer_cnt << "," << *time_up << "," << *time_sum << "," << sumup_write_time << "," << *time_all << std::endl;
            outFileHSUMUP.close();
        }
        else
        {
            sleep(1);
            outFileHSUMUP << myid << "," << sumup_outer_cnt << "," << *time_up << "," << *time_sum << "," << sumup_write_time << "," << *time_all << std::endl;
            outFileHSUMUP.close();
        }
    }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    sumup_outer_cnt++;
}

// void balanceBuckets(std::vector<Bucket> &buckets)
// {
//     double totalSum = std::accumulate(buckets.begin(), buckets.end(), 0.0,
//                                       [](double sum, const Bucket &b)
//                                       { return sum + b.sum; });
//     double averageSum = totalSum / buckets.size();
//     double threshold = averageSum * 0.01; // 设置阈值为平均总和�? 1%

//     auto compMax = [](const Bucket &a, const Bucket &b)
//     { return a.sum < b.sum; };
//     auto compMin = [](const Bucket &a, const Bucket &b)
//     { return a.sum > b.sum; };

//     std::priority_queue<Bucket, std::vector<Bucket>, decltype(compMax)> maxHeap(compMax, buckets);
//     std::priority_queue<Bucket, std::vector<Bucket>, decltype(compMin)> minHeap(compMin, buckets);

//     int maxIterations = 1000;
//     int iteration = 0;

//     while (iteration < maxIterations)
//     {
//         Bucket maxBucket = maxHeap.top();
//         Bucket minBucket = minHeap.top();
//         maxHeap.pop();
//         minHeap.pop();

//         double diff = maxBucket.sum - minBucket.sum;
//         if (diff < threshold)
//             break;

//         bool found = false;
//         for (int i = 0; i < maxBucket.batches.size(); ++i)
//         {
//             Batch batch = maxBucket.batches[i];
//             if (maxBucket.sum - batch.num >= minBucket.sum + batch.num)
//             {
//                 maxBucket.sum -= batch.num;
//                 minBucket.sum += batch.num;
//                 minBucket.batches.push_back(batch);
//                 maxBucket.batches.erase(maxBucket.batches.begin() + i);

//                 found = true;
//                 break;
//             }
//         }

//         if (!found)
//             break;

//         maxHeap.push(maxBucket);
//         minHeap.push(minBucket);

//         iteration++;
//     }

//     // 重构原始桶列�?
//     while (!maxHeap.empty())
//     {
//         Bucket bucket = maxHeap.top();
//         maxHeap.pop();
//         buckets[bucket.tid] = bucket;
//     }
// }
struct Batch
{
    int id;
    double num;
};

struct Bucket
{
    int tid;
    double sum;
    std::vector<Batch> batches;

    bool operator<(const Bucket &other) const
    {
        return sum > other.sum;
    }
};
std::vector<Bucket> distributeBatchesBalanced(std::vector<Batch> &batches)
{
    std::vector<Bucket> buckets(S);
    for (int i = 0; i < S; ++i)
    {
        buckets[i].tid = i;
    }

    std::sort(batches.begin(), batches.end(), [](const Batch &a, const Batch &b)
              { return a.num > b.num; });

    std::priority_queue<Bucket> pq(buckets.begin(), buckets.end());

    for (const auto &batch : batches)
    {
        Bucket minBucket = pq.top();
        pq.pop();

        minBucket.batches.push_back(batch);
        minBucket.sum += batch.num;

        pq.push(minBucket);
    }
    for (int i = 0; i < S; ++i)
    {
        buckets[i] = pq.top();
        pq.pop();
    }
    return buckets;
}
// std::vector<Bucket> distributeBatchesBalanced(std::vector<Batch> &batches)
// {

//     std::sort(batches.begin(), batches.end(), [](const Batch &a, const Batch &b)
//               { return a.num > b.num; });

//     // 使用 lambda 表达式来定义比较逻辑
//     auto comp = [](const Bucket &a, const Bucket &b)
//     {
//         return a.sum < b.sum; // 最小堆的比较逻辑
//     };

//     // 创建最小堆
//     std::priority_queue<Bucket, std::vector<Bucket>, decltype(comp)> pq(comp);

//     // 初始�? S �? Bucket，其 sum 均为 0，并加入到最小堆�?
//     for (int i = 0; i < S; ++i)
//     {
//         pq.push(Bucket{i, 0.0, {}});
//     }

//     for (const auto &batch : batches)
//     {
//         Bucket minBucket = pq.top();
//         pq.pop();

//         minBucket.batches.push_back(batch);
//         minBucket.sum += batch.num;

//         pq.push(minBucket);
//     }
//     std::vector<Bucket> buckets(S);
//     for (int i = 0; i < S; ++i)
//     {
//         buckets[i] = pq.top();
//         pq.pop();
//     }
//     // balanceBuckets(buckets);
//     return buckets;
// }

// void read_csv_to_map_(const char *filename)
// {
//     std::ifstream file(filename);
//     std::string line, key;
//     double value;

//     while (std::getline(file, line))
//     {
//         std::istringstream iss(line);
//         if (!std::getline(iss, key, ','))
//             continue; // 读取第一列作为key

//         std::string valueStr;
//         if (!std::getline(iss, valueStr, ','))
//             continue; // 读取第二列作为value的字符串形式

//         value = std::stod(valueStr); // 将字符串转换为double类型
//         time_map[key] = value;
//     }
//     std::cout << "success reading csv to map" << std::endl;
//     std::ofstream output_file("map_info.csv");
//     if (!output_file.is_open())
//     {
//         std::cerr << "Error opening output file" << std::endl;
//         return;
//     }
//     for (const auto &pair : time_map)
//     {
//         output_file << pair.first << "," << pair.second << std::endl;
//     }
//     output_file.close();
//     std::cout << "success writing map to file" << std::endl;
// }
void read_csv_to_map_(const char *filename)
{
    std::ifstream file(filename);
    std::string line;
    time_map.clear();
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string key;
        double value;
        if (std::getline(ss, key, ',') && ss >> value)
        {
            time_map[key] = value;
        }
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (myid == 0)
    // {
    //     std::ofstream output_file("map_info.csv");
    //     if (!output_file.is_open())
    //     {
    //         std::cerr << "Error opening output file" << std::endl;
    //         return;
    //     }
    //     for (const auto &pair : time_map)
    //     {
    //         output_file << pair.first << "," << pair.second << std::endl;
    //     }
    //     output_file.close();
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
}
std::string generate_id_concatenation(double *centers)
{
    std::hash<std::string> hasher;
    std::size_t hash = 0;
    std::string concatenated = "";
    for (int j = 0; j < 3; ++j)
    {
        // 对每个元素进行链�?
        concatenated = concatenated + std::to_string(centers[j]) + "_";
    }

    // 最终返回哈希值的字符串形�?
    hash = hasher(concatenated);
    return std::to_string(hash);
}
void get_my_id_map_(int *procid, int *n_batches, double *centers)
{

    std::ofstream outFiletest;
    std::string filename1;
    filename1 = "testdatas" + std::to_string(*procid) + ".csv";
    outFiletest.open(filename1, std::ios::app);
    for (int i = 0; i < *n_batches; i++)
    {
        std::string unique_id = generate_id_concatenation(&centers[i * 3]);
        outFiletest << unique_id << std::endl;
    }
    outFiletest.close();
    MPI_Barrier(MPI_COMM_WORLD);
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    // std::cout << "numProcs: " << numProcs << "\n";
    if (myid == 0)
    {
        std::ofstream combinedFile("combined_data_test.csv");
        for (int i = 0; i < numProcs; ++i)
        {
            std::ostringstream fname;
            fname << "testdatas" + std::to_string(i) + ".csv";
            std::ifstream inFile(fname.str());
            combinedFile << inFile.rdbuf();
            inFile.close();
            // 删除原始文件
            std::remove(fname.str().c_str());
        }
        combinedFile.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void get_gpu_time_(double *batch_times, int *i_my_batch, double *centers)
{

    std::string batch_key = generate_id_concatenation(&centers[(*i_my_batch - 1) * 3]);
    if (time_map.find(batch_key) == time_map.end())
    {
        std::cerr << "Error: Key not found in time_map, *i_my_batch is : " << *i_my_batch << "batch key is : " << batch_key << std::endl;
        exit(1);
        return; // 或者处理错误的其它方式
    }
    double gpu_batch_time = time_map[batch_key];
    batch_times[*i_my_batch - 1] = gpu_batch_time;
}

void get_merged_batch_weight_(double *batch_desc, int *batch_start, int *batch_nums)
{
    double centers[3];
    int start = *batch_start;
    int len = *batch_nums;
    // std::ofstream outFile312;
    // std::string filename312;
    // filename312 = "weightdata" + std::to_string(myid) + ".csv";
    // outFile312.open(filename312, std::ios::app);
    for (int i_batch = start; i_batch < start + len; i_batch++)
    {

        for (int i_coord = 0; i_coord < 3; i_coord++)
        {
            centers[i_coord] = batch_desc[i_batch * 5 + i_coord];
        }
        std::string batch_key = generate_id_concatenation(centers);
        if (time_map.find(batch_key) == time_map.end())
        {
            std::cerr << "Error1: Key not found in time_map, i_batch is : " << i_batch << " batch key is : " << batch_key << std::endl;
            exit(1);
            return; // 或者处理错误的其它方式
        }
        double merged_batch_weight = time_map[batch_key];
        batch_desc[i_batch * 5 + 3] = merged_batch_weight;
        double batch_id = batch_desc[i_batch * 5 + 4];
        // outFile312 << myid << "," << i_batch << "," << merged_batch_weight << "," << batch_id << std::endl;
    }
    // outFile312.close();
}

void sort_batch_desc_mod_(double *batch_desc_mod, int *len)
{
    int length = *len; // 获取子数组的长度
    // 由于Fortran中传入的是列优先，我们需要特别注意这一�?

    // 创建一个包含索引和值的向量
    std::vector<std::pair<double, double>> pairs(length);

    // 填充向量，每个元素包含值（用于排序的依据）和原始索�?
    for (int i = 0; i < length; i++)
    {
        pairs[i] = std::make_pair(batch_desc_mod[i * 2], batch_desc_mod[i * 2 + 1]);
    }

    // 对向量进行排序，基于每对中的第二个元素（即排序依据）
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b)
              { return a.second < b.second; });

    // 创建一个新的数组来保存排序后的结果
    std::vector<double> sorted_batch_desc_mod(length * 2);

    // 根据排序结果重排原数�?
    for (int i = 0; i < length; i++)
    {
        sorted_batch_desc_mod[i * 2] = pairs[i].first;
        sorted_batch_desc_mod[i * 2 + 1] = pairs[i].second;
    }

    // 将排序后的结果复制回原数�?
    for (int i = 0; i < length * 2; i++)
    {
        batch_desc_mod[i] = sorted_batch_desc_mod[i];
    }
}

double get_gpu_time_local(double *centers, int i_my_batch)
{
    std::string batch_key;
    batch_key = generate_id_concatenation(&centers[i_my_batch * 3]);
    double gpu_batch_time = time_map[batch_key];
    return gpu_batch_time;
}

// 写入数组到文件中的函�?
template <typename T>
void writeArrayToFile(T *arr, int arrSize, std::string arrName)
{
    using namespace std;
    string filename = "./out_datas/test_data" + to_string(myid) + ".txt";
    // 打开文件并定位到文件末尾
    ofstream outFile(filename, ios::out | ios::app);

    // 写入数组名称和元素数�?
    outFile << arrName << "," << arrSize << endl;

    // 写入数组的元�?
    for (int i = 0; i < arrSize; i++)
    {
        outFile << arr[i] << " ";
    }
    outFile << endl;

    // 关闭文件
    outFile.close();
}

size_t platformId = 0; // INFO choose a platform
int n_coeff_hartree = 2;
int init_sum_up = 0;
// const char *file_names[] = {"/public/home/aicao/jiazifan/submit-bak/fhi-aims-dcu-mod/src/gpu/gpuSum.cu",
//                             "/public/home/aicao/jiazifan/submit-bak/fhi-aims-dcu-mod/src/gpu/gpuAll.cu"};
const char *file_names[] = {"/public/home/aicao/jiazifan/submit-bak/fhi-aims-dcu-mod/src/gpu/gpuSum.cu"};
const char *kernel_names[] = {"sum_up_whole_potential_shanghui_sub_t_", "sum_up_whole_potential_shanghui_pre_proc_"};
hipFunction_t kernels[number_of_kernels];
char *buffer[number_of_files];
size_t sizes[number_of_files];
int numOfGpus;

size_t localSize[] = {128};        // 可能被重新设�? !!!
size_t globalSize[] = {128 * 128}; // 可能被重新设�? !!!

size_t globalSize_sum_up_pre_proc[1];
size_t localSize_sum_up_pre_proc[1] = {64};
const int i_center_tile_size_default = 256;

// extern double *Fp_function_spline_slice;
// extern double *Fpc_function_spline_slice;
// extern double *Fp;

double *Fp_function_spline_slice = NULL;
double *Fpc_function_spline_slice = NULL;
double *Fp = NULL;

RHO_PARAM rho_param;
H_PARAM H_param;

static int hip_init_finished = 0;
static int hip_init_1_finished = 0;
static int hip_common_buffer_init_finished = 0;
static int remember_arg_index_1 = -10;
static int remember_arg_Fp_function_spline = -10;
static int remember_arg_n_my_batches_work_sumup = -10;
static int remember_arg_Fp_max_grid = -10;
static int sum_up_first_begin_finished = 0;
static int sum_up_begin_0_finished = 0;
static int rho_first_begin_finished = 0;
static int H_first_begin_finished = 0;
static int h_begin_0_finished = 0;
static int htimes = 0;
static int rhotimes = 0;
static int hcnt = 0;
static int rhocnt = 0;
static int sumupcnt = 0;

struct HIP_BUF_COM_T
{
    hipDeviceptr_t species;
    hipDeviceptr_t empty;
    hipDeviceptr_t centers_hartree_potential;
    hipDeviceptr_t center_to_atom;
    hipDeviceptr_t species_center;
    hipDeviceptr_t coords_center;
    hipDeviceptr_t l_hartree;
    hipDeviceptr_t n_grid;
    hipDeviceptr_t n_radial;
    hipDeviceptr_t r_grid_min;
    hipDeviceptr_t r_grid_inc;
    hipDeviceptr_t log_r_grid_inc;
    hipDeviceptr_t scale_radial;
    hipDeviceptr_t r_radial;
    hipDeviceptr_t r_grid;
    hipDeviceptr_t n_cc_lm_ijk;
    hipDeviceptr_t index_cc;
    hipDeviceptr_t index_ijk_max_cc;
    hipDeviceptr_t b0;
    hipDeviceptr_t b2;
    hipDeviceptr_t b4;
    hipDeviceptr_t b6;
    hipDeviceptr_t a_save;
    hipDeviceptr_t Fp_function_spline_slice;
    hipDeviceptr_t Fpc_function_spline_slice;
    // ---
    hipDeviceptr_t rho_multipole_index;
    hipDeviceptr_t compensation_norm;
    hipDeviceptr_t compensation_radius;
    hipDeviceptr_t rho_multipole_h_p_s;
    hipDeviceptr_t multipole_radius_free;
    // rho global
    hipDeviceptr_t perm_basis_fns_spl;
    hipDeviceptr_t outer_radius_sq;
    hipDeviceptr_t basis_fn;
    hipDeviceptr_t basis_l;
    hipDeviceptr_t atom_radius_sq;
    hipDeviceptr_t basis_fn_start_spl;
    hipDeviceptr_t basis_fn_atom;
    hipDeviceptr_t basis_wave_ordered;
    hipDeviceptr_t basis_kinetic_ordered;
    hipDeviceptr_t Cbasis_to_basis;
    hipDeviceptr_t Cbasis_to_center;
    hipDeviceptr_t centers_basis_integrals; // 可能因为宏展开出问�?
    hipDeviceptr_t index_hamiltonian;
    hipDeviceptr_t position_in_hamiltonian;
    hipDeviceptr_t column_index_hamiltonian;
    // pbc_lists_coords_center �? coords_center，只是为了规避一点点重名
    hipDeviceptr_t center_to_cell;
    // // loop helper
    hipDeviceptr_t point_to_i_batch;
    hipDeviceptr_t point_to_i_index;
    hipDeviceptr_t valid_point_to_i_full_point;
    hipDeviceptr_t index_cc_aos;
    hipDeviceptr_t i_center_to_centers_index;
    // sum up
    // hipDeviceptr_t partition_tab_std;
    // hipDeviceptr_t delta_v_hartree;               // (n_full_points_work)
    // hipDeviceptr_t rho_multipole;                 // (n_full_points_work)
    hipDeviceptr_t centers_rho_multipole_spl;     // (l_pot_max+1)**2, n_max_spline, n_max_radial+2, n_atoms)
    hipDeviceptr_t centers_delta_v_hart_part_spl; // (l_pot_max+1)**2, n_coeff_hartree, n_hartree_grid, n_atoms)
    // hipDeviceptr_t adap_outer_radius_sq;          // (n_atoms)
    // hipDeviceptr_t multipole_radius_sq;           // (n_atoms)
    // hipDeviceptr_t l_hartree_max_far_distance;    // (n_atoms)
    // hipDeviceptr_t outer_potential_radius;        // (0:l_pot_max, n_atoms)
    // hipDeviceptr_t multipole_c;                   // (n_cc_lm_ijk(l_pot_max), n_atoms)
    // sum up tmp
    hipDeviceptr_t angular_integral_log; // per block (l_pot_max + 1) * (l_pot_max + 1) * n_max_grid
    hipDeviceptr_t Fp;                   // global_size * (l_pot_max + 2) * n_centers_hartree_potential)
    hipDeviceptr_t coord_c;
    hipDeviceptr_t coord_mat;
    hipDeviceptr_t rest_mat;
    hipDeviceptr_t vector;
    hipDeviceptr_t delta_v_hartree_multipole_component;
    hipDeviceptr_t rho_multipole_component;
    hipDeviceptr_t ylm_tab;
    // sum_up batches
    // hipDeviceptr_t batches_size_sumup;
    // hipDeviceptr_t batches_points_coords_sumup;
    // rho tmp
    hipDeviceptr_t dist_tab_sq__;
    hipDeviceptr_t dist_tab__;
    hipDeviceptr_t dir_tab__;
    hipDeviceptr_t atom_index__;
    hipDeviceptr_t atom_index_inv__;
    hipDeviceptr_t i_basis_fns__;
    hipDeviceptr_t i_basis_fns_inv__;
    hipDeviceptr_t i_atom_fns__;
    hipDeviceptr_t spline_array_start__;
    hipDeviceptr_t spline_array_end__;
    hipDeviceptr_t one_over_dist_tab__;
    hipDeviceptr_t rad_index__;
    hipDeviceptr_t wave_index__;
    hipDeviceptr_t l_index__;
    hipDeviceptr_t l_count__;
    hipDeviceptr_t fn_atom__;
    hipDeviceptr_t zero_index_point__;
    hipDeviceptr_t wave__;
    hipDeviceptr_t first_order_density_matrix_con__;
    hipDeviceptr_t i_r__;
    hipDeviceptr_t trigonom_tab__;
    hipDeviceptr_t radial_wave__;
    hipDeviceptr_t spline_array_aux__;
    hipDeviceptr_t aux_radial__;
    hipDeviceptr_t ylm_tab__;
    hipDeviceptr_t dylm_dtheta_tab__;
    hipDeviceptr_t scaled_dylm_dphi_tab__;
    hipDeviceptr_t kinetic_wave__;
    hipDeviceptr_t grid_coord__;
    hipDeviceptr_t H_times_psi__;
    hipDeviceptr_t T_plus_V__;
    hipDeviceptr_t contract__;
    hipDeviceptr_t wave_t__;
    hipDeviceptr_t first_order_H_dense__;
    // // rho batches
    hipDeviceptr_t batches_size_rho;
    hipDeviceptr_t batches_batch_n_compute_rho;
    hipDeviceptr_t batches_batch_i_basis_rho;
    hipDeviceptr_t batches_points_coords_rho;
    // H batches
    hipDeviceptr_t batches_size_H;
    hipDeviceptr_t batches_batch_n_compute_H;
    hipDeviceptr_t batches_batch_i_basis_H;
    hipDeviceptr_t batches_points_coords_H;
    // rho
    hipDeviceptr_t basis_l_max__;
    hipDeviceptr_t n_points_all_batches__;
    hipDeviceptr_t n_batch_centers_all_batches__;
    hipDeviceptr_t batch_center_all_batches__;
    hipDeviceptr_t batch_point_to_i_full_point__;
    hipDeviceptr_t ins_idx_all_batches__;
    hipDeviceptr_t first_order_rho__;
    hipDeviceptr_t first_order_density_matrix__;
    hipDeviceptr_t partition_tab__;
    // hipDeviceptr_t tmp_rho__; // only for swcl
    // H
    hipDeviceptr_t batches_batch_i_basis_h__;
    hipDeviceptr_t partition_all_batches__;
    hipDeviceptr_t first_order_H__;
    hipDeviceptr_t local_potential_parts_all_points__;
    hipDeviceptr_t local_first_order_rho_all_batches__;
    hipDeviceptr_t local_first_order_potential_all_batches__;
    hipDeviceptr_t local_dVxc_drho_all_batches__;
    hipDeviceptr_t local_rho_gradient__;
    hipDeviceptr_t first_order_gradient_rho__;
    // distribute
    hipDeviceptr_t d_block_h;
    hipDeviceptr_t d_count_batches_h;
    hipDeviceptr_t d_block_rho;
    hipDeviceptr_t d_count_batches_rho;
    hipDeviceptr_t d_new_batch_count;
    hipDeviceptr_t d_new_batch_i_start;
    // cpy
    hipDeviceptr_t n_points_all_batches_H__;
    hipDeviceptr_t n_batch_centers_all_batches_H__;
    hipDeviceptr_t batch_center_all_batches_H__;
    hipDeviceptr_t ins_idx_all_batches_H__;
    hipDeviceptr_t d_count_if_h;
    hipDeviceptr_t diverge_matrix;
} hip_buf_com;

typedef struct CL_BUF_SUMUP_T
{
    // sum up
    hipDeviceptr_t partition_tab_std;
    hipDeviceptr_t delta_v_hartree; // (n_full_points_work)
    hipDeviceptr_t rho_multipole;   // (n_full_points_work)
    // hipDeviceptr_t centers_rho_multipole_spl;     // (l_pot_max+1)**2, n_max_spline, n_max_radial+2, n_atoms)
    // hipDeviceptr_t centers_delta_v_hart_part_spl; // (l_pot_max+1)**2, n_coeff_hartree, n_hartree_grid, n_atoms)
    hipDeviceptr_t adap_outer_radius_sq;       // (n_atoms)
    hipDeviceptr_t multipole_radius_sq;        // (n_atoms)
    hipDeviceptr_t l_hartree_max_far_distance; // (n_atoms)
    hipDeviceptr_t outer_potential_radius;     // (0:l_pot_max, n_atoms)
    hipDeviceptr_t multipole_c;                // (n_cc_lm_ijk(l_pot_max), n_atoms)
    // sum_up batches
    hipDeviceptr_t batches_size_sumup;
    hipDeviceptr_t batches_points_coords_sumup;

    hipDeviceptr_t point_to_i_batch;
    hipDeviceptr_t point_to_i_index;
    hipDeviceptr_t valid_point_to_i_full_point;
} CL_BUF_SUMUP;

CL_BUF_SUMUP cl_buf_sumup[8];

static int rho_pass_vars_count = -1;
void rho_pass_vars_(
    int *l_ylm_max,
    int *n_local_matrix_size,
    int *n_basis_local,
    int *perm_n_full_points,
    int *first_order_density_matrix_size,
    int *basis_l_max,
    int *n_points_all_batches,
    int *n_batch_centers_all_batches,
    int *batch_center_all_batches,
    int *batch_point_to_i_full_point,
    int *ins_idx_all_batches,
    double *first_order_rho,
    double *first_order_density_matrix,
    double *partition_tab)
{
    rho_pass_vars_count++;

    rho_param.l_ylm_max = *l_ylm_max;
    rho_param.n_local_matrix_size = *n_local_matrix_size;
    rho_param.n_basis_local = *n_basis_local;
    rho_param.perm_n_full_points = *perm_n_full_points;
    rho_param.first_order_density_matrix_size = *first_order_density_matrix_size;
    rho_param.basis_l_max = basis_l_max;
    rho_param.n_points_all_batches = n_points_all_batches;
    rho_param.n_batch_centers_all_batches = n_batch_centers_all_batches;
    rho_param.batch_center_all_batches = batch_center_all_batches;
    rho_param.batch_point_to_i_full_point = batch_point_to_i_full_point;
    rho_param.ins_idx_all_batches = ins_idx_all_batches;
    rho_param.first_order_rho = first_order_rho;
    rho_param.first_order_density_matrix = first_order_density_matrix;
    rho_param.partition_tab = partition_tab;
    //   char save_file_name[64];
    //   if((myid == 0 || myid == 3) && rho_pass_vars_count <= 1){
    //     sprintf(save_file_name, "mdata_outer_rank%d_%d.bin", myid, rho_pass_vars_count);
    //     m_save_load_rho(save_file_name, 0, 0);
    //   }
}
static int H_pass_vars_count = -1;
void h_pass_vars_(
    int *j_coord_,
    int *n_spin_,
    int *l_ylm_max_,
    int *n_basis_local_,
    int *n_matrix_size_,
    int *basis_l_max,
    int *n_points_all_batches,
    int *n_batch_centers_all_batches,
    int *batch_center_all_batches,
    int *ins_idx_all_batches,
    int *batches_batch_i_basis_h,
    double *partition_all_batches,
    double *first_order_H,
    double *local_potential_parts_all_points,
    double *local_first_order_rho_all_batches,
    double *local_first_order_potential_all_batches,
    double *local_dVxc_drho_all_batches,
    double *local_rho_gradient,
    double *first_order_gradient_rho)
{
    H_pass_vars_count++;
    H_param.j_coord = *j_coord_;
    H_param.n_spin = *n_spin_;
    H_param.l_ylm_max = *l_ylm_max_;
    H_param.n_basis_local = *n_basis_local_;
    H_param.n_matrix_size = *n_matrix_size_;
    H_param.basis_l_max = basis_l_max;
    H_param.n_points_all_batches = n_points_all_batches;
    H_param.n_batch_centers_all_batches = n_batch_centers_all_batches;
    H_param.batch_center_all_batches = batch_center_all_batches;
    H_param.ins_idx_all_batches = ins_idx_all_batches;
    H_param.batches_batch_i_basis_h = batches_batch_i_basis_h;
    H_param.partition_all_batches = partition_all_batches;
    H_param.first_order_H = first_order_H;
    H_param.local_potential_parts_all_points = local_potential_parts_all_points;
    H_param.local_first_order_rho_all_batches = local_first_order_rho_all_batches;
    H_param.local_first_order_potential_all_batches = local_first_order_potential_all_batches;
    H_param.local_dVxc_drho_all_batches = local_dVxc_drho_all_batches;
    H_param.local_rho_gradient = local_rho_gradient;
    H_param.first_order_gradient_rho = first_order_gradient_rho;
    // char save_file_name[64];
    // if ((myid == 0 || myid == 3) && H_pass_vars_count <= 4)
    // {
    //     sprintf(save_file_name, "mdata_outer_rank%d_%d.bin", myid, H_pass_vars_count);
    //     m_save_load_H(save_file_name, 0, 0);
    // }
}

#define IF_ERROR_EXIT(cond, err_code, str)                                                                              \
    if (cond)                                                                                                           \
    {                                                                                                                   \
        printf("Error! rank%d, %s:%d, %s\nError_code=%d\n%s\n", myid, __FILE__, __LINE__, __FUNCTION__, err_code, str); \
        fflush(stdout);                                                                                                 \
        exit(-1);                                                                                                       \
    }

#define _CHK_(size1, size2)                                                           \
    if ((size1) != (size2))                                                           \
    {                                                                                 \
        printf("Error! rank%d, %s:%d, %s\n", myid, __FILE__, __LINE__, __FUNCTION__); \
        fflush(stdout);                                                               \
        exit(-1);                                                                     \
    }
#define _NL_ status = func(&endl, sizeof(char), 1, file_p);
#define _FWD_(type, var)                              \
    status = func(&var, sizeof(type), 1, file_p);     \
    if (print)                                        \
    {                                                 \
        printf("rank%d, %s = %d\n", myid, #var, var); \
    }

#define _FW_(type, var, size, var_out, hip_mem_flag)                                                 \
    if (hip_mem_flag == hipMemcpyHostToDevice)                                                       \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
        hip_err = hipMemcpy(hip_buf_com.var_out, var, sizeof(type) * (size), hipMemcpyHostToDevice); \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMemcpy failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 0)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 1)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipHostRegister(var, sizeof(type) * (size), hipHostRegisterDefault);    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostRegister failed");                     \
        hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.var_out, var, 0);                    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostGetDevicePointer failed");             \
    }                                                                                                \
    else                                                                                             \
    {                                                                                                \
        fprintf(stderr, "Invalid hip memory flag");                                                  \
        exit(EXIT_FAILURE);                                                                          \
    }

#define _FWF_(type, var, size, var_out, hip_mem_flag, file_flag)                                     \
    if (file_flag == 1)                                                                              \
    {                                                                                                \
        writeArrayToFile(var, size, #var_out);                                                       \
    }                                                                                                \
    if (hip_mem_flag == hipMemcpyHostToDevice)                                                       \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
        hip_err = hipMemcpy(hip_buf_com.var_out, var, sizeof(type) * (size), hipMemcpyHostToDevice); \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMemcpy failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 0)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 1)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipHostRegister(var, sizeof(type) * (size), hipHostRegisterDefault);    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostRegister failed");                     \
        hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.var_out, var, 0);                    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostGetDevicePointer failed");             \
    }                                                                                                \
    else                                                                                             \
    {                                                                                                \
        fprintf(stderr, "Invalid hip memory flag");                                                  \
        exit(EXIT_FAILURE);                                                                          \
    }

#define _FWV_(type, var, size, var_out, hip_mem_flag)                                         \
    {                                                                                         \
        {                                                                                     \
            hipError_t error = hipMalloc((void **)&var_out, sizeof(type) * (size));           \
                                                                                              \
            IF_ERROR_EXIT(error != hipSuccess, error, "hipMalloc failed");                    \
        }                                                                                     \
        if (hip_mem_flag == hipMemcpyHostToDevice)                                            \
        {                                                                                     \
            hipError_t cpyError;                                                              \
            cpyError = hipMemcpy(var_out, var, sizeof(type) * (size), hipMemcpyHostToDevice); \
            IF_ERROR_EXIT(cpyError != hipSuccess, cpyError, "hipMemcpy failed");              \
        }                                                                                     \
    }

#define hipSetupArgumentWithCheck(kernel_name, arg_index, off_set, arg_size, arg_value)                      \
    {                                                                                                        \
        hipError_t error = hipSetupArgument(arg_value, arg_size, off_set);                                   \
        off_set = off_set + arg_size;                                                                        \
        if (error != hipSuccess)                                                                             \
        {                                                                                                    \
            fprintf(stderr, "hipSetupArgument failed at arg %d: %s\n", arg_index, hipGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    }

#define setKernelArgs(args, arg_index, arg_size, arg_value) \
    {                                                       \
        args[arg_index] = arg_value;                        \
    }

void loadProgramSource(const char **files, size_t length, char **buffer, size_t *sizes)
{
    /* Read each source file (*.cl) and store the contents into a temporary datastore */
    for (size_t i = 0; i < length; i++)
    {
        FILE *file = fopen(files[i], "r");
        if (file == NULL)
        {
            printf("Couldn't read the program file\n");
            fflush(stdout);
            exit(1);
        }
        fseek(file, 0, SEEK_END);
        sizes[i] = ftell(file);
        rewind(file); // reset the file pointer so that 'fread' reads from the front
        buffer[i] = (char *)malloc(sizes[i] + 1);
        buffer[i][sizes[i]] = '\0';
        size_t status = fread(buffer[i], sizeof(char), sizes[i], file);
        IF_ERROR_EXIT(status != sizes[i], (int)status, "fread filed, read num != file size");
        fclose(file);
    }
}

// void hip_device_init_()
// {
//     if (hip_init_finished)
//         return;
//     hip_init_finished = 1;
//     hipError_t error;
//     // // 得到平台数量
//     // error = hipGetDeviceCount(&numOfGpus);
//     // IF_ERROR_EXIT(error != hipSuccess, error, "Unable to find any GPUs");
//     // printf("Number of GPUs: %d\n", numOfGpus);
//     // // int device_id = DEVICE_ID;
//     // // 选择指定平台上的指定设备
//     int device_id = mpi_platform_relative_id % 4;
//     // std::cout << "mpi_platform_relative_id is : " << mpi_platform_relative_id << " device_id is : " << device_id << std::endl;
//     // int device_id = myid % 4;
//     // IF_ERROR_EXIT(numOfGpus <= device_id, numOfGpus, "The selected platformId is out of range");
//     error = hipSetDevice(device_id);
//     IF_ERROR_EXIT(error != hipSuccess, error, "Unable to set GPU");
// }

void hip_init_()
{
    if (hip_init_finished)
        return;
    hip_init_finished = 1;
    hipError_t error;
    int device_id = mpi_platform_relative_id % 4;
    error = hipSetDevice(device_id);
    IF_ERROR_EXIT(error != hipSuccess, error, "Unable to set GPU");

    if (JIT_ENABLED)
    {
        // Read one or more source files
        loadProgramSource(file_names, number_of_files, buffer, sizes);

        // if (myid == 0)
        //     std::cout << buffer[0] << std::endl;
        // Create HIP RTC program
        // 将所有源代码文件内容拼接到一个字符串中
        // std::string combined_kernel_code;
        // for (size_t i = 0; i < number_of_files; ++i)
        // {
        //     combined_kernel_code += std::string(buffer[i], sizes[i]);
        // }
        hiprtcProgram prog;
        hiprtcResult ret;
        ret = hiprtcCreateProgram(&prog, buffer[0], "combined_kernels", 0, nullptr, nullptr);
        if (ret != HIPRTC_SUCCESS)
        {
            std::cerr << "Failed to create RTC program: " << hiprtcGetErrorString(ret) << std::endl;
            exit(1);
        }
        // Set build options
        int N_PERIODIC_OR_USE_HARTREE_NON_PERIODIC_EWALD;
        if (n_periodic > 0 || use_hartree_non_periodic_ewald)
        {
            N_PERIODIC_OR_USE_HARTREE_NON_PERIODIC_EWALD = 1;
        }
        else
        {
            N_PERIODIC_OR_USE_HARTREE_NON_PERIODIC_EWALD = 0;
        }

        char option1[64];
        sprintf(option1, "-DL_POT_MAX=%d", l_pot_max);
        char option2[64];
        sprintf(option2, "-DHARTREE_FP_FUNCTION_SPLINES=%d", hartree_fp_function_splines);
        char option3[64];
        sprintf(option3, "-DN_PERIODIC_OR_USE_HARTREE_NON_PERIODIC_EWALD=%d", N_PERIODIC_OR_USE_HARTREE_NON_PERIODIC_EWALD);
        char option4[64];
        sprintf(option4, "-DMACRO_n_centers_integrals=%d", n_centers_integrals);
        char option5[64];
        sprintf(option5, "-DMACRO_n_max_compute_atoms=%d", n_max_compute_atoms);
        char option6[64];
        sprintf(option6, "-DMACRO_n_centers=%d", n_centers);
        char option7[64];
        sprintf(option7, "-DLOCALSIZE_SUM_UP_PRE_PROC=%zu", localSize_sum_up_pre_proc[0]);
        char option8[64];
        sprintf(option8, "-DUSE_JIT");
        const char *options[] = {option1, option2, option3, option4, option5, option6, option7, option8};

        // const char *options[] = {option8};

        std::cout << "buildOptions: " << std::endl;
        for (const auto &option : options)
        {
            std::cout << option << std::endl;
        }

        // Add the kernel name expression for each kernel name
        for (int i = 0; i < number_of_kernels; i++)
        {
            hiprtcResult result = hiprtcAddNameExpression(prog, kernel_names[i]);
            if (result != HIPRTC_SUCCESS)
            {
                std::cerr << "Failed to add name expression for kernel: " << kernel_names[i] << ". Error code: " << result << std::endl;
                exit(1);
            }
        }

        // Compile the program
        hiprtcResult compileResult = hiprtcCompileProgram(prog, sizeof(options) / sizeof(options[0]), options);
        // Check the compilation result
        if (compileResult != HIPRTC_SUCCESS)
        {
            std::cerr << "Failed to compile kernel" << std::endl;

            // Get the log size
            size_t logSize;
            hiprtcGetProgramLogSize(prog, &logSize);

            // Get the log
            std::vector<char> log(logSize);
            hiprtcGetProgramLog(prog, log.data());

            // Print the log
            std::cerr << "Compilation log:" << std::endl;
            std::cerr << log.data() << std::endl;

            exit(1);
        }
        // Get the mangled kernel names
        std::vector<std::string> mangledKernelNames(number_of_kernels);
        for (int i = 0; i < number_of_kernels; i++)
        {
            const char *mangledKernelName;
            hiprtcResult result = hiprtcGetLoweredName(prog, kernel_names[i], &mangledKernelName);
            // std::cout << "mangledKernelName: " << mangledKernelName << std::endl;
            if (result != HIPRTC_SUCCESS)
            {
                std::cerr << "Failed to get mangled kernel name for kernel: " << kernel_names[i] << ". Error code: " << result << std::endl;
                exit(1);
            }
            mangledKernelNames[i] = mangledKernelName;
        }

        // if (myid == 0)
        // {
        //     std::cout << "mangledKernelNames 0: " << mangledKernelNames[0] << std::endl;
        //     std::cout << "mangledKernelNames 1: " << mangledKernelNames[1] << std::endl;
        // }
        // Get the compiled code
        size_t codeSize;
        hiprtcResult result = hiprtcGetCodeSize(prog, &codeSize);
        if (result != HIPRTC_SUCCESS)
        {
            std::cerr << "Failed to get compiled code size. Error code: " << result << std::endl;
            exit(1);
        }

        char *kernelCode = new char[codeSize];
        // std::vector<char> kernelCode(codeSize);
        result = hiprtcGetCode(prog, &kernelCode[0]);
        if (result != HIPRTC_SUCCESS)
        {
            std::cerr << "Failed to get compiled code. Error code: " << result << std::endl;
            exit(1);
        }

        // Load the compiled code into the HIP module
        // std::cout << "Load compiled code" << std::endl;
        hipModule_t module;
        hipError_t hipResult = hipModuleLoadData(&module, &kernelCode[0]);
        if (hipResult != hipSuccess)
        {
            std::cerr << "Failed to load compiled code. Error code: " << hipResult << std::endl;
            exit(1);
        }

        // std::cout << "Get kernel functions" << std::endl;
        // Create kernels
        for (int i = 0; i < number_of_kernels; i++)
        {
            hipFunction_t kernelFunc;
            hipResult = hipModuleGetFunction(&kernelFunc, module, mangledKernelNames[i].c_str());
            // std::cout << mangledKernelNames[i] << " hipResult is : " << hipResult << std::endl;
            if (hipResult != hipSuccess)
            {
                std::cerr << "Failed to get kernel function: " << mangledKernelNames[i] << ". Error code: " << hipResult << std::endl;
                exit(1);
            }
            kernels[i] = kernelFunc;
        }
        hiprtcDestroyProgram(&prog);
        // std::cout << "Get kernel functions finished" << std::endl;

        // std::cout << "Kernel handles:" << std::endl;
        // for (int i = 0; i < number_of_kernels; i++)
        // {
        //     std::cout << "Kernel " << i << ": " << (void *)kernels[i] << std::endl;
        // }

        // Cleanup
        delete[] kernelCode;
    }
}

void hip_device_finish_()
{
    if (!hip_init_1_finished)
        return;
    hip_init_1_finished = 0;
    hipDeviceReset();
}

void hip_common_buffer_init_()
{
    if (hip_common_buffer_init_finished)
        return;
    hip_common_buffer_init_finished = 1;

    _FW_(int, MV(geometry, species), n_atoms, species, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, centers_hartree_potential), n_centers_hartree_potential, centers_hartree_potential,
         hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, center_to_atom), n_centers, center_to_atom, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, species_center), n_centers, species_center, hipMemcpyHostToDevice);
    _FW_(double, MV(pbc_lists, coords_center), 3 * n_centers, coords_center, hipMemcpyHostToDevice);
    _FW_(int, MV(species_data, l_hartree), n_species, l_hartree, hipMemcpyHostToDevice);
    _FW_(int, MV(grids, n_grid), n_species, n_grid, hipMemcpyHostToDevice);
    _FW_(int, MV(grids, n_radial), n_species, n_radial, hipMemcpyHostToDevice);

    _FW_(double, MV(grids, r_grid_min), n_species, r_grid_min, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, r_grid_inc), n_species, r_grid_inc, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, log_r_grid_inc), n_species, log_r_grid_inc, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, scale_radial), n_species, scale_radial, hipMemcpyHostToDevice);
    _FW_(int, MV(analytic_multipole_coefficients, n_cc_lm_ijk), (l_max_analytic_multipole + 1), n_cc_lm_ijk,
         hipMemcpyHostToDevice);
    _FW_(int, MV(analytic_multipole_coefficients, index_cc), n_cc_lm_ijk(l_max_analytic_multipole) * 6, index_cc,
         hipMemcpyHostToDevice);
    _FW_(int, MV(analytic_multipole_coefficients, index_ijk_max_cc), 3 * (l_max_analytic_multipole + 1), index_ijk_max_cc,
         hipMemcpyHostToDevice);
    _FW_(double, MV(hartree_potential_real_p0, b0), pmaxab + 1, b0, hipMemcpyHostToDevice);
    _FW_(double, MV(hartree_potential_real_p0, b2), pmaxab + 1, b2, hipMemcpyHostToDevice);
    _FW_(double, MV(hartree_potential_real_p0, b4), pmaxab + 1, b4, hipMemcpyHostToDevice);
    _FW_(double, MV(hartree_potential_real_p0, b6), pmaxab + 1, b6, hipMemcpyHostToDevice);
    _FW_(double, MV(hartree_potential_real_p0, a_save), pmaxab + 1, a_save, hipMemcpyHostToDevice);
    // rho global
    _FW_(int, MV(basis, perm_basis_fns_spl), n_basis_fns, perm_basis_fns_spl, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, outer_radius_sq), n_basis_fns, outer_radius_sq, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn), n_basis, basis_fn, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_l), n_basis, basis_l, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, atom_radius_sq), n_species, atom_radius_sq, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn_start_spl), n_species, basis_fn_start_spl, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn_atom), n_basis_fns *n_atoms, basis_fn_atom, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, basis_wave_ordered), n_basis_fns *n_max_spline *n_max_grid, basis_wave_ordered, hipMemcpyHostToDevice);       // 进程间可能不同
    _FW_(double, MV(basis, basis_kinetic_ordered), n_basis_fns *n_max_spline *n_max_grid, basis_kinetic_ordered, hipMemcpyHostToDevice); // 进程间可能不同

    _FW_(int, MV(pbc_lists, cbasis_to_basis), n_centers_basis_T, Cbasis_to_basis, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, cbasis_to_center), n_centers_basis_T, Cbasis_to_center, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, centers_basis_integrals), n_centers_basis_integrals, centers_basis_integrals, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, index_hamiltonian), 2 * index_hamiltonian_dim2 * n_basis, index_hamiltonian, hipMemcpyHostToDevice);                                   // 进程间可能不同
    _FW_(int, MV(pbc_lists, position_in_hamiltonian), position_in_hamiltonian_dim1 *position_in_hamiltonian_dim2, position_in_hamiltonian, hipMemcpyHostToDevice); // 进程间可能不同
    _FW_(int, MV(pbc_lists, column_index_hamiltonian), column_index_hamiltonian_size, column_index_hamiltonian, hipMemcpyHostToDevice);                            // 进程间可能不同
    _FW_(int, MV(pbc_lists, center_to_cell), n_centers, center_to_cell, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, r_radial), n_max_radial *n_species, r_radial, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, r_grid), n_max_grid *n_species, r_grid, hipMemcpyHostToDevice);
    _FW_(int, NULL, n_atoms, rho_multipole_index, 0);
    if (compensate_multipole_errors)
    {
        _FW_(int, MV(hartree_potential_storage, compensation_norm), n_atoms, compensation_norm, hipMemcpyHostToDevice);
        _FW_(int, MV(hartree_potential_storage, compensation_radius), n_atoms, compensation_radius, hipMemcpyHostToDevice);
    }
    else
    {
        _FW_(int, NULL, 1, compensation_norm, 0);
        _FW_(int, NULL, 1, compensation_radius, 0);
    }

    _FW_(double, MV(species_data, multipole_radius_free), n_species, multipole_radius_free, hipMemcpyHostToDevice);
    // hipError_t hip_err;
    // // 对于 hipMemcpyHostToDevice 的情�?
    // hip_err = hipMalloc(&hip_buf_com.species, sizeof(int) * n_atoms);
    // hip_err = hipMalloc(&hip_buf_com.centers_hartree_potential, sizeof(int) * n_centers_hartree_potential);
    // hip_err = hipMalloc(&hip_buf_com.center_to_atom, sizeof(int) * n_centers);
    // hip_err = hipMalloc(&hip_buf_com.species_center, sizeof(int) * n_centers);
    // hip_err = hipMalloc(&hip_buf_com.coords_center, sizeof(double) * 3 * n_centers);
    // hip_err = hipMalloc(&hip_buf_com.l_hartree, sizeof(int) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.n_grid, sizeof(int) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.n_radial, sizeof(int) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.r_grid_min, sizeof(double) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.r_grid_inc, sizeof(double) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.log_r_grid_inc, sizeof(double) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.scale_radial, sizeof(double) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.n_cc_lm_ijk, sizeof(int) * (l_max_analytic_multipole + 1));
    // hip_err = hipMalloc(&hip_buf_com.index_cc, sizeof(int) * n_cc_lm_ijk(l_max_analytic_multipole) * 6);
    // hip_err = hipMalloc(&hip_buf_com.index_ijk_max_cc, sizeof(int) * 3 * (l_max_analytic_multipole + 1));
    // hip_err = hipMalloc(&hip_buf_com.b0, sizeof(double) * (pmaxab + 1));
    // hip_err = hipMalloc(&hip_buf_com.b2, sizeof(double) * (pmaxab + 1));
    // hip_err = hipMalloc(&hip_buf_com.b4, sizeof(double) * (pmaxab + 1));
    // hip_err = hipMalloc(&hip_buf_com.b6, sizeof(double) * (pmaxab + 1));
    // hip_err = hipMalloc(&hip_buf_com.a_save, sizeof(double) * (pmaxab + 1));
    // hip_err = hipMalloc(&hip_buf_com.perm_basis_fns_spl, sizeof(int) * n_basis_fns);
    // hip_err = hipMalloc(&hip_buf_com.outer_radius_sq, sizeof(double) * n_basis_fns);
    // hip_err = hipMalloc(&hip_buf_com.basis_fn, sizeof(int) * n_basis);
    // hip_err = hipMalloc(&hip_buf_com.basis_l, sizeof(int) * n_basis);
    // hip_err = hipMalloc(&hip_buf_com.atom_radius_sq, sizeof(double) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.basis_fn_start_spl, sizeof(int) * n_species);
    // hip_err = hipMalloc(&hip_buf_com.basis_fn_atom, sizeof(int) * n_basis_fns * n_atoms);
    // hip_err = hipMalloc(&hip_buf_com.basis_wave_ordered, sizeof(double) * n_basis_fns * n_max_spline * n_max_grid);
    // hip_err = hipMalloc(&hip_buf_com.basis_kinetic_ordered, sizeof(double) * n_basis_fns * n_max_spline * n_max_grid);
    // hip_err = hipMalloc(&hip_buf_com.Cbasis_to_basis, sizeof(int) * n_centers_basis_T);
    // hip_err = hipMalloc(&hip_buf_com.Cbasis_to_center, sizeof(int) * n_centers_basis_T);
    // hip_err = hipMalloc(&hip_buf_com.centers_basis_integrals, sizeof(int) * n_centers_basis_integrals);
    // hip_err = hipMalloc(&hip_buf_com.index_hamiltonian, sizeof(int) * 2 * index_hamiltonian_dim2 * n_basis);
    // hip_err = hipMalloc(&hip_buf_com.position_in_hamiltonian, sizeof(int) * position_in_hamiltonian_dim1 * position_in_hamiltonian_dim2);
    // hip_err = hipMalloc(&hip_buf_com.column_index_hamiltonian, sizeof(int) * column_index_hamiltonian_size);
    // hip_err = hipMalloc(&hip_buf_com.center_to_cell, sizeof(int) * n_centers);
    // hip_err = hipMalloc(&hip_buf_com.r_radial, sizeof(double) * n_max_radial * n_species);
    // hip_err = hipMalloc(&hip_buf_com.r_grid, sizeof(double) * n_max_grid * n_species);
    // hip_err = hipMalloc(&hip_buf_com.rho_multipole_index, sizeof(int) * n_atoms);
    // hip_err = hipMalloc(&hip_buf_com.multipole_radius_free, sizeof(double) * n_species);
    // // 对于 hipMemcpyHostToDevice 的情�?
    // hip_err = hipMemcpy(hip_buf_com.species, MV(geometry, species), sizeof(int) * n_atoms, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.centers_hartree_potential, MV(pbc_lists, centers_hartree_potential), sizeof(int) * n_centers_hartree_potential, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.center_to_atom, MV(pbc_lists, center_to_atom), sizeof(int) * n_centers, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.species_center, MV(pbc_lists, species_center), sizeof(int) * n_centers, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.coords_center, MV(pbc_lists, coords_center), sizeof(double) * 3 * n_centers, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.l_hartree, MV(species_data, l_hartree), sizeof(int) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.n_grid, MV(grids, n_grid), sizeof(int) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.n_radial, MV(grids, n_radial), sizeof(int) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.r_grid_min, MV(grids, r_grid_min), sizeof(double) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.r_grid_inc, MV(grids, r_grid_inc), sizeof(double) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.log_r_grid_inc, MV(grids, log_r_grid_inc), sizeof(double) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.scale_radial, MV(grids, scale_radial), sizeof(double) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.n_cc_lm_ijk, MV(analytic_multipole_coefficients, n_cc_lm_ijk), sizeof(int) * (l_max_analytic_multipole + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.index_cc, MV(analytic_multipole_coefficients, index_cc), sizeof(int) * n_cc_lm_ijk(l_max_analytic_multipole) * 6, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.index_ijk_max_cc, MV(analytic_multipole_coefficients, index_ijk_max_cc), sizeof(int) * 3 * (l_max_analytic_multipole + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.b0, MV(hartree_potential_real_p0, b0), sizeof(double) * (pmaxab + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.b2, MV(hartree_potential_real_p0, b2), sizeof(double) * (pmaxab + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.b4, MV(hartree_potential_real_p0, b4), sizeof(double) * (pmaxab + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.b6, MV(hartree_potential_real_p0, b6), sizeof(double) * (pmaxab + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.a_save, MV(hartree_potential_real_p0, a_save), sizeof(double) * (pmaxab + 1), hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.perm_basis_fns_spl, MV(basis, perm_basis_fns_spl), sizeof(int) * n_basis_fns, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.outer_radius_sq, MV(basis, outer_radius_sq), sizeof(double) * n_basis_fns, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_fn, MV(basis, basis_fn), sizeof(int) * n_basis, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_l, MV(basis, basis_l), sizeof(int) * n_basis, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.atom_radius_sq, MV(basis, atom_radius_sq), sizeof(double) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_fn_start_spl, MV(basis, basis_fn_start_spl), sizeof(int) * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_fn_atom, MV(basis, basis_fn_atom), sizeof(int) * n_basis_fns * n_atoms, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_wave_ordered, MV(basis, basis_wave_ordered), sizeof(double) * n_basis_fns * n_max_spline * n_max_grid, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.basis_kinetic_ordered, MV(basis, basis_kinetic_ordered), sizeof(double) * n_basis_fns * n_max_spline * n_max_grid, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.Cbasis_to_basis, MV(pbc_lists, cbasis_to_basis), sizeof(int) * n_centers_basis_T, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.Cbasis_to_center, MV(pbc_lists, cbasis_to_center), sizeof(int) * n_centers_basis_T, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.centers_basis_integrals, MV(pbc_lists, centers_basis_integrals), sizeof(int) * n_centers_basis_integrals, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.index_hamiltonian, MV(pbc_lists, index_hamiltonian), sizeof(int) * 2 * index_hamiltonian_dim2 * n_basis, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.position_in_hamiltonian, MV(pbc_lists, position_in_hamiltonian), sizeof(int) * position_in_hamiltonian_dim1 * position_in_hamiltonian_dim2, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.column_index_hamiltonian, MV(pbc_lists, column_index_hamiltonian), sizeof(int) * column_index_hamiltonian_size, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.center_to_cell, MV(pbc_lists, center_to_cell), sizeof(int) * n_centers, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.r_radial, MV(grids, r_radial), sizeof(double) * n_max_radial * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.r_grid, MV(grids, r_grid), sizeof(double) * n_max_grid * n_species, hipMemcpyHostToDevice);
    // hip_err = hipMemcpy(hip_buf_com.multipole_radius_free, MV(species_data, multipole_radius_free), sizeof(double) * n_species, hipMemcpyHostToDevice);

    // if (compensate_multipole_errors)
    // {
    //     // hipMalloc calls
    //     hip_err = hipMalloc(&hip_buf_com.compensation_norm, sizeof(int) * n_atoms);
    //     hip_err = hipMalloc(&hip_buf_com.compensation_radius, sizeof(int) * n_atoms);

    //     // hipMemcpy calls
    //     hip_err = hipMemcpy(hip_buf_com.compensation_norm, MV(hartree_potential_storage, compensation_norm), sizeof(int) * n_atoms, hipMemcpyHostToDevice);
    //     hip_err = hipMemcpy(hip_buf_com.compensation_radius, MV(hartree_potential_storage, compensation_radius), sizeof(int) * n_atoms, hipMemcpyHostToDevice);
    // }
    // else
    // {
    //     // hipMalloc calls for dummy or placeholder memory allocation
    //     hip_err = hipMalloc(&hip_buf_com.compensation_norm, sizeof(int) * 1);   // Assuming placeholder memory allocation is needed
    //     hip_err = hipMalloc(&hip_buf_com.compensation_radius, sizeof(int) * 1); // Assuming placeholder memory allocation is needed
    //     // No hipMemcpy calls since no data needs to be copied for the false condition
    // }
}

void hip_common_buffer_free_()
{
    if (!hip_common_buffer_init_finished)
        return;
    hip_common_buffer_init_finished = 0;

    unsigned int arg_index = 0;

    hipFree(hip_buf_com.species);
    hipFree(hip_buf_com.centers_hartree_potential);
    hipFree(hip_buf_com.center_to_atom);
    hipFree(hip_buf_com.species_center);
    hipFree(hip_buf_com.coords_center);
    hipFree(hip_buf_com.l_hartree);
    hipFree(hip_buf_com.n_grid);
    hipFree(hip_buf_com.n_radial);

    hipFree(hip_buf_com.r_grid_min);
    hipFree(hip_buf_com.r_grid_inc);
    hipFree(hip_buf_com.log_r_grid_inc);
    hipFree(hip_buf_com.scale_radial);
    hipFree(hip_buf_com.n_cc_lm_ijk);
    hipFree(hip_buf_com.index_cc);
    hipFree(hip_buf_com.index_ijk_max_cc);
    hipFree(hip_buf_com.b0);
    hipFree(hip_buf_com.b2);
    hipFree(hip_buf_com.b4);
    hipFree(hip_buf_com.b6);
    hipFree(hip_buf_com.a_save);

    // rho global
    hipFree(hip_buf_com.perm_basis_fns_spl);
    hipFree(hip_buf_com.outer_radius_sq);
    hipFree(hip_buf_com.basis_fn);
    hipFree(hip_buf_com.basis_l);
    hipFree(hip_buf_com.atom_radius_sq);
    hipFree(hip_buf_com.basis_fn_start_spl);
    hipFree(hip_buf_com.basis_fn_atom);
    hipFree(hip_buf_com.basis_wave_ordered);
    hipFree(hip_buf_com.basis_kinetic_ordered);

    hipFree(hip_buf_com.Cbasis_to_basis);
    hipFree(hip_buf_com.Cbasis_to_center);
    hipFree(hip_buf_com.centers_basis_integrals);
    hipFree(hip_buf_com.index_hamiltonian);
    hipFree(hip_buf_com.position_in_hamiltonian);
    hipFree(hip_buf_com.column_index_hamiltonian);

    hipFree(hip_buf_com.center_to_cell);

    // if(ctrl_use_sumup_pre_c_cl_version){
    hipFree(hip_buf_com.r_radial);
    hipFree(hip_buf_com.r_grid);
    hipFree(hip_buf_com.rho_multipole_index);
    hipFree(hip_buf_com.compensation_norm);
    hipFree(hip_buf_com.compensation_radius);
    // hipFree(hip_buf_com.rho_multipole_h_p_s);
    hipFree(hip_buf_com.multipole_radius_free);
    // }
    // ------
}

void rho_first_begin()
{
    if (rho_first_begin_finished)
        return;
    rho_first_begin_finished = 1;
    hip_init_();               // ok may be wrong !
    hip_common_buffer_init_(); // ok
}

void rho_begin_()
{
    // for (int kk = 0; kk < 1; kk++)
    // {
    // if (rhotimes == 0 && kk == 0)
    // {
    //     pre_run_();
    // }
    struct timeval start, end, start1, end1;
    // gettimeofday(&start, NULL);
    rho_first_begin();
    rho_first_begin_finished = 0;

    hipEvent_t rho_cpy_start,rho_cpy_end,rho_kernel_start,rho_kernel_end;
    hipEventCreate(&rho_cpy_start);
    hipEventCreate(&rho_cpy_end);
    float rho_cpy_time,rho_kernel_time;


    
    long time_uses[2];
    //   char *time_infos[32];
    //   size_t time_index = 0;

    hipError_t error;
    int arg_index;

    size_t localSize[] = {256};      // 覆盖前面的设�?
    size_t globalSize[] = {256 * S}; // 覆盖前面的设�?
    dim3 blockDim(localSize[0], 1, 1);
    dim3 gridDim(globalSize[0] / blockDim.x, 1, 1);
    //   if(myid == 0)
    //     printf("n_max_batch_size=%d\n", n_max_batch_size);
    // rho tmp
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab_sq__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (3 * n_max_compute_atoms), dir_tab__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), atom_index__, 0);
    // _FW_(int, NULL, globalSize[0] * (n_centers_integrals), atom_index_inv__, CL_MEM_READ_WRITE);
    _FW_(int, NULL, 1, atom_index_inv__, 0);
    _FW_(int, NULL, 1, i_basis_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_basis_fns * (n_max_compute_atoms + 1)), i_basis_fns_inv__, 0);
    _FW_(int, NULL, 1, i_atom_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_start__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_end__, 0);
    _FW_(double, NULL, globalSize[0] * n_max_compute_atoms, one_over_dist_tab__, 0);
    _FW_(int, NULL, globalSize[0] * n_max_compute_atoms, rad_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), wave_index__, 0);
    _FW_(int, NULL, 1, l_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), l_count__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), fn_atom__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_ham), zero_index_point__, 0);
    // _FW_(double, NULL, (globalSize[0] * (n_max_compute_ham) + 128), wave__, CL_MEM_READ_WRITE); // 多加 128 为了避免 TILE 后越界
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 256 + 16 * n_max_compute_ham,
         wave__, 0); // 多加 128 为了避免 TILE 后越界, 长宽按 128 对齐
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * (n_max_compute_dens * n_max_compute_dens) + 256 + 16 * n_max_compute_ham,
         first_order_density_matrix_con__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), i_r__, 0);
    _FW_(double, NULL, globalSize[0] * (4 * n_max_compute_atoms), trigonom_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_fns_ham), radial_wave__, 0);
    _FW_(double, NULL, globalSize[0] * (n_basis_fns), spline_array_aux__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms * n_basis_fns), aux_radial__, 0);
    _FW_(double, NULL, globalSize[0] * ((rho_param.l_ylm_max + 1) * (rho_param.l_ylm_max + 1) * n_max_compute_atoms), ylm_tab__, 0);
    _FW_(double, NULL, globalSize[0] * ((rho_param.l_ylm_max + 1) * (rho_param.l_ylm_max + 1) * n_max_compute_atoms), dylm_dtheta_tab__, 0);
    _FW_(double, NULL, ((rho_param.l_ylm_max + 1) * (rho_param.l_ylm_max + 1) * n_max_compute_atoms), scaled_dylm_dphi_tab__, 0); // 反正这个和上面那个暂时无�?
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * ((n_max_batch_size+127)/128*128) * ((n_max_compute_ham+127)/128*128) + 128,
    //               tmp_rho__, CL_MEM_READ_WRITE);  // only for swcl

    // rho param
    // _FW_(int, rho_param.basis_l_max, n_species, basis_l_max__, hipMemcpyHostToDevice);
    // _FW_(int, rho_param.n_points_all_batches, n_my_batches_work_rho, n_points_all_batches__, hipMemcpyHostToDevice);
    // _FW_(int, rho_param.n_batch_centers_all_batches, n_my_batches_work_rho, n_batch_centers_all_batches__, hipMemcpyHostToDevice);
    // _FW_(int, rho_param.batch_center_all_batches, max_n_batch_centers *n_my_batches_work_rho, batch_center_all_batches__, hipMemcpyHostToDevice);
    // _FW_(int, rho_param.batch_point_to_i_full_point, n_max_batch_size *n_my_batches_work_rho, batch_point_to_i_full_point__, hipMemcpyHostToDevice);
    // if (rho_param.n_basis_local > 0)
    // {
    //     _FW_(int, rho_param.ins_idx_all_batches, rho_param.n_basis_local *n_my_batches_work_rho, ins_idx_all_batches__, hipMemcpyHostToDevice);
    //     _FW_(double, rho_param.first_order_rho, rho_param.perm_n_full_points, first_order_rho__, hipMemcpyHostToDevice);
    // }
    // else
    // {
    //     _FW_(int, NULL, 1, ins_idx_all_batches__, 0);
    //     _FW_(double, rho_param.first_order_rho, n_full_points_work_rho, first_order_rho__, hipMemcpyHostToDevice);
    // }
    // _FW_(double, rho_param.first_order_density_matrix, rho_param.first_order_density_matrix_size, first_order_density_matrix__, hipMemcpyHostToDevice);
    // _FW_(double, rho_param.partition_tab, n_full_points_work_rho, partition_tab__, hipMemcpyHostToDevice);

    // // rho batches
    // _FW_(int, MV(opencl_util, batches_size_rho), n_my_batches_work_rho, batches_size_rho, 1);
    // _FW_(int, MV(opencl_util, batches_batch_n_compute_rho), n_my_batches_work_rho, batches_batch_n_compute_rho, 1);
    // _FW_(int, MV(opencl_util, batches_batch_i_basis_rho), n_max_compute_dens *n_my_batches_work_rho, batches_batch_i_basis_rho, 1);
    // _FW_(double, MV(opencl_util, batches_points_coords_rho), 3 * n_max_batch_size * n_my_batches_work_rho, batches_points_coords_rho, 1);

    // 新增的对�? hipMemcpyHostToDevice 的情�?
    hipError_t hip_err;
    hip_err = hipMalloc(&hip_buf_com.basis_l_max__, sizeof(int) * n_species);
    hip_err = hipMalloc(&hip_buf_com.n_points_all_batches__, sizeof(int) * n_my_batches_work_rho);
    hip_err = hipMalloc(&hip_buf_com.n_batch_centers_all_batches__, sizeof(int) * n_my_batches_work_rho);
    hip_err = hipMalloc(&hip_buf_com.batch_center_all_batches__, sizeof(int) * max_n_batch_centers * n_my_batches_work_rho);
    hip_err = hipMalloc(&hip_buf_com.batch_point_to_i_full_point__, sizeof(int) * n_max_batch_size * n_my_batches_work_rho);
    hip_err = hipMalloc(&hip_buf_com.first_order_density_matrix__, sizeof(double) * rho_param.first_order_density_matrix_size);
    hip_err = hipMalloc(&hip_buf_com.partition_tab__, sizeof(double) * n_full_points_work_rho);

    hip_err = hipMemcpy(hip_buf_com.basis_l_max__, rho_param.basis_l_max, sizeof(int) * n_species, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.n_points_all_batches__, rho_param.n_points_all_batches, sizeof(int) * n_my_batches_work_rho, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.n_batch_centers_all_batches__, rho_param.n_batch_centers_all_batches, sizeof(int) * n_my_batches_work_rho, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batch_center_all_batches__, rho_param.batch_center_all_batches, sizeof(int) * max_n_batch_centers * n_my_batches_work_rho, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batch_point_to_i_full_point__, rho_param.batch_point_to_i_full_point, sizeof(int) * n_max_batch_size * n_my_batches_work_rho, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.first_order_density_matrix__, rho_param.first_order_density_matrix, sizeof(double) * rho_param.first_order_density_matrix_size, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.partition_tab__, rho_param.partition_tab, sizeof(double) * n_full_points_work_rho, hipMemcpyHostToDevice);
    if (rho_param.n_basis_local > 0)
    {
        hip_err = hipMalloc(&hip_buf_com.ins_idx_all_batches__, sizeof(int) * rho_param.n_basis_local * n_my_batches_work_rho);
        hip_err = hipMalloc(&hip_buf_com.first_order_rho__, sizeof(double) * rho_param.perm_n_full_points);
        hip_err = hipMemcpy(hip_buf_com.ins_idx_all_batches__, rho_param.ins_idx_all_batches, sizeof(int) * rho_param.n_basis_local * n_my_batches_work_rho, hipMemcpyHostToDevice);
        hip_err = hipMemcpy(hip_buf_com.first_order_rho__, rho_param.first_order_rho, sizeof(double) * rho_param.perm_n_full_points, hipMemcpyHostToDevice);
    }
    else
    {
        hip_err = hipMalloc(&hip_buf_com.ins_idx_all_batches__, sizeof(int) * 1);
        hip_err = hipMalloc(&hip_buf_com.first_order_rho__, sizeof(double) * n_full_points_work_rho);
        hip_err = hipMemcpy(hip_buf_com.first_order_rho__, rho_param.first_order_rho, sizeof(double) * n_full_points_work_rho, hipMemcpyHostToDevice);
    }
    // 对于 hip_mem_flag == 1 的情�?
    if (rhocnt == 0)
    {
        hip_err = hipHostRegister(MV(opencl_util, batches_size_rho), sizeof(int) * n_my_batches_work_rho, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_batch_n_compute_rho), sizeof(int) * n_my_batches_work_rho, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_batch_i_basis_rho), sizeof(int) * n_max_compute_dens * n_my_batches_work_rho, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_points_coords_rho), sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_rho, hipHostRegisterDefault);
        hip_err = hipMalloc(&hip_buf_com.batches_size_rho, sizeof(int) * n_my_batches_work_rho);
        hip_err = hipMalloc(&hip_buf_com.batches_batch_n_compute_rho, sizeof(int) * n_my_batches_work_rho);
        hip_err = hipMalloc(&hip_buf_com.batches_batch_i_basis_rho, sizeof(int) * n_max_compute_dens * n_my_batches_work_rho);
        hip_err = hipMalloc(&hip_buf_com.batches_points_coords_rho, sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_rho);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_size_rho, MV(opencl_util, batches_size_rho), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_batch_n_compute_rho, MV(opencl_util, batches_batch_n_compute_rho), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_batch_i_basis_rho, MV(opencl_util, batches_batch_i_basis_rho), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_points_coords_rho, MV(opencl_util, batches_points_coords_rho), 0);
    }
    hipEventRecord(rho_cpy_start);
    hip_err = hipMemcpy(hip_buf_com.batches_size_rho,MV(opencl_util, batches_size_rho),sizeof(int) * n_my_batches_work_rho,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_batch_n_compute_rho,MV(opencl_util, batches_batch_n_compute_rho),sizeof(int) * n_my_batches_work_rho,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_batch_i_basis_rho,MV(opencl_util, batches_batch_i_basis_rho),sizeof(int) * n_max_compute_dens * n_my_batches_work_rho,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_points_coords_rho,MV(opencl_util, batches_points_coords_rho), sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_rho,hipMemcpyHostToDevice);
    hipEventRecord(rho_cpy_end);
    hipEventSynchronize(rho_cpy_end);
    hipEventElapsedTime(&rho_cpy_time, rho_cpy_start, rho_cpy_end);
    int Data_size = sizeof(int) * n_my_batches_work_rho + sizeof(int) * n_my_batches_work_rho + sizeof(int) * n_max_compute_dens * n_my_batches_work_rho + sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_rho;
    float thery_time = 0.09 + 7.84259e-08 * Data_size + 0.01;

    // rho param
    arg_index = 0;
    void *args_rho[84];
    setKernelArgs(args_rho, arg_index++, sizeof(int), &rho_param.l_ylm_max);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &rho_param.n_local_matrix_size);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &rho_param.n_basis_local);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &rho_param.first_order_density_matrix_size);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l_max__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_points_all_batches__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_batch_centers_all_batches__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_center_all_batches__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_point_to_i_full_point__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ins_idx_all_batches__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_rho__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_density_matrix__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.partition_tab__);
    // IF_ERROR_EXIT(error != hipSuccess, error, "clSetKernelArg failed");

    arg_index = 13;
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_centers);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_centers_integrals);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_compute_fns_ham);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_basis_fns);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_centers_basis_I);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_grid);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_compute_atoms);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_compute_ham);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_compute_dens);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_max_batch_size);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &index_hamiltonian_dim2);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &position_in_hamiltonian_dim1);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &position_in_hamiltonian_dim2);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &column_index_hamiltonian_size);
    //_CHK_(error, hipSuccess);

    arg_index = 27;
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_my_batches_work_rho);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &n_full_points_work_rho);
    // IF_ERROR_EXIT(error != hipSuccess, error, "clSetKernelArg failed");

    arg_index = 29;
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_atom);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species_center);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_cell);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_basis);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_center);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_basis_integrals);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_hamiltonian);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.position_in_hamiltonian);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.column_index_hamiltonian);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coords_center);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_grid);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid_min);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.log_r_grid_inc);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.perm_basis_fns_spl);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.outer_radius_sq);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_radius_sq);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_start_spl);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_atom);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_wave_ordered);
    //_CHK_(error, hipSuccess);

    // rho batches
    arg_index = 50;
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_size_rho);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_n_compute_rho);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_i_basis_rho);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_points_coords_rho);
    // IF_ERROR_EXIT(error != hipSuccess, error, "clSetKernelArg failed");

    // rho tmp
    arg_index = 54;
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab_sq__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dir_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index_inv__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns_inv__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_atom_fns__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_start__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_end__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.one_over_dist_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rad_index__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave_index__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_index__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_count__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.fn_atom__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.zero_index_point__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_density_matrix_con__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_r__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.trigonom_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.radial_wave__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_aux__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.aux_radial__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ylm_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dylm_dtheta_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.scaled_dylm_dphi_tab__);
    setKernelArgs(args_rho, arg_index++, sizeof(int), &max_n_batch_centers);

    // if (rhotimes == 0)
    // {
    //     std::vector<Batch> batches;
    //     // for (int i = 0; i < n_my_batches_work_rho; ++i)
    //     // {
    //     //     batches.push_back({i, batch_time[i]});
    //     // }
    //     for (int i = 0; i < n_my_batches_work_rho; ++i)
    //     {
    //         double t_batch = get_gpu_time_local(center_all_batches, i);
    //         gpu_batches_times_RHO += t_batch;
    //         batches.push_back({i, t_batch});
    //     }
    //     // for (int i = 0; i < n_my_batches_work_rho; ++i)
    //     // {
    //     //     batches.push_back({i, rho_param.n_batch_centers_all_batches[i]});
    //     // }
    //     std::vector<Bucket> buckets = distributeBatchesBalanced(batches);
    //     int new_id = 0;
    //     for (int i = 0; i < S; i++)
    //     {
    //         for (std::vector<Batch>::iterator it = buckets[i].batches.begin(); it != buckets[i].batches.end(); ++it)
    //         {
    //             block_rho[new_id] = it->id;
    //             count_batches_rho[i]++;
    //             new_id++;
    //         }
    //     }
    // }
    // rhotimes++;
    // _FW_(int, block_rho, n_my_batches_work_rho, d_block_rho, hipMemcpyHostToDevice);
    // _FW_(int, count_batches_rho, S, d_count_batches_rho, hipMemcpyHostToDevice);
    // setKernelArgs(args_rho, arg_index++, sizeof(int), &hip_buf_com.d_block_rho);
    // setKernelArgs(args_rho, arg_index++, sizeof(int), &hip_buf_com.d_count_batches_rho);

    // float RHO_time;
    // hipEvent_t rho_start, rho_stop;

    // hipEventCreate(&rho_start);
    // hipEventCreate(&rho_stop);
    // hipEventRecord(rho_start);

    hipEventCreate(&rho_kernel_start);
    hipEventCreate(&rho_kernel_end);

    hipEventRecord(rho_kernel_start);


    error = hipLaunchKernel(reinterpret_cast<const void *>(&integrate_first_order_rho_sub_tmp2_), gridDim, blockDim, args_rho, 0, 0);
    IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel failed")

    hipEventRecord(rho_kernel_end);
    hipEventSynchronize(rho_kernel_end);
    hipEventElapsedTime(&rho_kernel_time, rho_kernel_start, rho_kernel_end);





    // hipEventRecord(rho_stop);
    // hipEventSynchronize(rho_stop);
    // hipEventElapsedTime(&RHO_time, rho_start, rho_stop);

    // if (kk == 0)
    // {
    if (rho_param.n_basis_local > 0)
    {
        hipMemcpy(rho_param.first_order_rho, hip_buf_com.first_order_rho__, sizeof(double) * rho_param.perm_n_full_points, hipMemcpyDeviceToHost);
    }
    else
    {
        hipMemcpy(rho_param.first_order_rho, hip_buf_com.first_order_rho__, sizeof(double) * n_full_points_work_rho, hipMemcpyDeviceToHost);
    }
    if(myid == 0){
    std::ofstream outFile("outputrho.csv",std::ios::app);
        if (!outFile) {  // 检查文件是否成功打开
            std::cerr << "文件打开失败！" << std::endl;
        }
        outFile <<"rho thery_time"<<","<<"rho cpy time"<<","<<"rho kernel time"<< std::endl;
        outFile <<thery_time<<","<<rho_cpy_time<<","<<rho_kernel_time<< std::endl;
        outFile.close();
    }
    // }
    // hipEventDestroy(rho_start);
    // hipEventDestroy(rho_stop);
    // 可能有问�?
    // hipHostUnregister(MV(opencl_util, batches_size_rho));
    // hipFree(hip_buf_com.batches_size_rho);
    // // 可能有问�?
    // hipHostUnregister(MV(opencl_util, batches_batch_n_compute_rho));
    // hipFree(hip_buf_com.batches_batch_n_compute_rho);
    // // 可能有问�?
    // hipHostUnregister(MV(opencl_util, batches_batch_i_basis_rho));
    // hipFree(hip_buf_com.batches_batch_i_basis_rho);
    // // 可能有问�?
    // hipHostUnregister(MV(opencl_util, batches_points_coords_rho));
    // hipFree(hip_buf_com.batches_points_coords_rho);

    hipFree(hip_buf_com.basis_l_max__);
    hipFree(hip_buf_com.n_points_all_batches__);
    hipFree(hip_buf_com.n_batch_centers_all_batches__);
    hipFree(hip_buf_com.batch_center_all_batches__);
    hipFree(hip_buf_com.batch_point_to_i_full_point__);
    hipFree(hip_buf_com.ins_idx_all_batches__);

    hipFree(hip_buf_com.first_order_rho__);
    hipFree(hip_buf_com.first_order_density_matrix__);
    hipFree(hip_buf_com.partition_tab__);

    hipFree(hip_buf_com.dist_tab_sq__);
    hipFree(hip_buf_com.dist_tab__);
    hipFree(hip_buf_com.dir_tab__);
    hipFree(hip_buf_com.atom_index__);
    hipFree(hip_buf_com.atom_index_inv__);
    hipFree(hip_buf_com.i_basis_fns__);
    hipFree(hip_buf_com.i_basis_fns_inv__);
    hipFree(hip_buf_com.i_atom_fns__);
    hipFree(hip_buf_com.spline_array_start__);
    hipFree(hip_buf_com.spline_array_end__);
    hipFree(hip_buf_com.one_over_dist_tab__);
    hipFree(hip_buf_com.rad_index__);
    hipFree(hip_buf_com.wave_index__);
    hipFree(hip_buf_com.l_index__);
    hipFree(hip_buf_com.l_count__);
    hipFree(hip_buf_com.fn_atom__);
    hipFree(hip_buf_com.zero_index_point__);
    hipFree(hip_buf_com.wave__);
    hipFree(hip_buf_com.first_order_density_matrix_con__);
    hipFree(hip_buf_com.i_r__);
    hipFree(hip_buf_com.trigonom_tab__);
    hipFree(hip_buf_com.radial_wave__);
    hipFree(hip_buf_com.spline_array_aux__);
    hipFree(hip_buf_com.aux_radial__);
    hipFree(hip_buf_com.ylm_tab__);
    hipFree(hip_buf_com.dylm_dtheta_tab__);
    hipFree(hip_buf_com.scaled_dylm_dphi_tab__);
    hipFree(hip_buf_com.d_block_rho);
    hipFree(hip_buf_com.d_count_batches_rho);

    // hip_common_buffer_free_();

    //   if (myid == 0)
    //     m_save_check_rho_(rho_param.first_order_rho);

    // printf("End\n");
    // gettimeofday(&end, NULL);

    // time_uses[0] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // }
    // gettimeofday(&start, NULL);
    // std::ofstream outFileRHO;
    // std::string filename;
    // outFileRHO.open("RHOdatas-newdis.csv", std::ios::app);

    // if (outFileRHO.is_open())
    // {
    //     if (myid == 0)
    //     {
    //         outFileRHO << "myid"
    //                    << ","
    //                    << "rhocnt"
    //                    << ","
    //                    << "RHO_kernel_time"
    //                    << ","
    //                    << "kernel_and_others" << std::endl;
    //         outFileRHO << myid << "," << rhocnt << "," << RHO_time << "," << time_uses[0] / 1000.0 << std::endl;
    //         outFileRHO.close();
    //     }
    //     else
    //     {
    //         sleep(1);
    //         outFileRHO << myid << "," << rhocnt << "," << RHO_time << "," << time_uses[0] / 1000.0 << std::endl;
    //         outFileRHO.close();
    //     }
    // }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    // gettimeofday(&end, NULL);
    // rho_write_time = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000000.0;
    rhocnt++;
    // for(size_t i=0; i<time_index; i++){
    //   if(myid < 8)
    //     printf("rank%d, %s: %lf seconds\n", myid, time_infos[i], time_uses[i]/1000000.0);
    // }
}
void hipfree_rho_(){
        hipFree(hip_buf_com.batches_size_rho);
        hipFree(hip_buf_com.batches_batch_n_compute_rho);
        hipFree(hip_buf_com.batches_batch_i_basis_rho);
        hipFree(hip_buf_com.batches_points_coords_rho);
}
// ok
void H_first_begin()
{
    if (H_first_begin_finished)
        return;
    H_first_begin_finished = 1;
    hip_init_(); // ok may be wrong !
    // if (htimes == 0)
    // {
    //     hip_common_buffer_init_();
    // }
    // else
    // {
    //     hip_common_buffer_init_();
    // }
    hip_common_buffer_init_();
    // ok
    // size_t localSize[] = {256};        // 覆盖前面的设�?
    // size_t globalSize[] = {256 * 128}; // 覆盖前面的设�?
    // dim3 blockDim(localSize[0], 1, 1);
    // dim3 gridDim(globalSize[0] / blockDim.x, 1, 1);
    // error = hipConfigureCall(gridDim, blockDim, 0, 0);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipConfigureCall failed");
    // int fast_ylm = 1;
    // int new_ylm = 0;
}

void output_times_fortran_(double *CPU_time,double*H_time){
    std::ofstream outFileHTime;
    std::string filename;
    outFileHTime.open("load_balance_time.csv", std::ios::app);
    if (outFileHTime.is_open())
    {
        if (myid == 0)
        {
            // outFileHTime <<"myid"
            //              <<","
            //              <<"CPU_time"
            //              <<","
            //              <<"H_time"
            //              <<std::endl;
            outFileHTime<<myid<<(*H_time)*1000<<std::endl;   
        }
    }
}
// // Int 类型元素的总数
// int total_int_elements_h = n_species + n_my_batches_work_h * (4 + max_n_batch_centers + H_param.n_basis_local + n_max_compute_dens) + 1;

// // Double 类型元素的总数
// //  int total_double_elements = n_max_batch_size * n_my_batches_work_h + 3 * n_max_batch_size * n_my_batches_work_h + H_param.n_matrix_size * H_param.n_spin + 1 + H_param.n_spin * n_max_batch_size * n_my_batches_work_h + 4 * n_max_batch_size * n_my_batches_work_h + 6 * H_param.n_spin * n_max_batch_size;
// int total_double_elements_h = H_param.n_matrix_size * H_param.n_spin + H_param.n_spin * n_max_batch_size * n_my_batches_work_h + 6 * H_param.n_spin * n_max_batch_size + 8 * n_max_batch_size * n_my_batches_work_h + 1;
// ok
// extern int *MV(opencl_util, n_points_all_batches_H);
float cpytime = 0;
float regtime = 0;
float malloctime = 0;
float getpointertime = 0;
float timefw = 0;
float timesplit = 0;
void h_begin_0_orgin(){
    if (h_begin_0_finished)
        return;
    h_begin_0_finished = 1;

    H_first_begin();
    H_first_begin_finished = 0;

    hipError_t error;

    // H param
    _FW_(int, H_param.basis_l_max, n_species, basis_l_max__, hipMemcpyHostToDevice);

    _FW_(int, H_param.n_points_all_batches, n_my_batches_work_h, n_points_all_batches__, hipMemcpyHostToDevice);
    _FW_(int, H_param.n_batch_centers_all_batches, n_my_batches_work_h, n_batch_centers_all_batches__, hipMemcpyHostToDevice);
    _FW_(int, H_param.batch_center_all_batches, max_n_batch_centers *n_my_batches_work_h, batch_center_all_batches__, hipMemcpyHostToDevice);

    // _FW_(int, H_param.batch_point_to_i_full_point, n_max_batch_size * n_my_batches_work_h, batch_point_to_i_full_point__, hipMemcpyHostToDevice);
    if (H_param.n_basis_local > 0)
    {
        _FW_(int, H_param.ins_idx_all_batches, H_param.n_basis_local *n_my_batches_work_h, ins_idx_all_batches__, hipMemcpyHostToDevice);
    }
    else
    {
        _FW_(int, NULL, 1, ins_idx_all_batches__, 0); // 废弃
    }
    // _FW_(int, H_param.batches_batch_i_basis_h, (n_centers_basis_I * n_my_batches_work_h), batches_batch_i_basis_h__, 0 | CL_MEM_COPY_HOST_PTR);  // 输出
    // _FW_(int, H_param.batches_batch_i_basis_h, 1, batches_batch_i_basis_h__, 0 | CL_MEM_COPY_HOST_PTR);  // 输出
    _FW_(int, NULL, 1, batches_batch_i_basis_h__, 0);
    _FW_(double, H_param.partition_all_batches, (n_max_batch_size * n_my_batches_work_h), partition_all_batches__, hipMemcpyHostToDevice);
    _FW_(double, H_param.first_order_H, (H_param.n_matrix_size * H_param.n_spin), first_order_H__, hipMemcpyHostToDevice);
    // _FW_(double, H_param.local_potential_parts_all_points, (H_param.n_spin * n_full_points_work_h), local_potential_parts_all_points__, hipMemcpyHostToDevice);
    _FW_(double, H_param.local_potential_parts_all_points, 1, local_potential_parts_all_points__, hipMemcpyHostToDevice);
    _FW_(double, H_param.local_first_order_rho_all_batches, (H_param.n_spin * n_max_batch_size * n_my_batches_work_h), local_first_order_rho_all_batches__, 1);
    _FW_(double, H_param.local_first_order_potential_all_batches, (n_max_batch_size * n_my_batches_work_h), local_first_order_potential_all_batches__, 1);
    _FW_(double, H_param.local_dVxc_drho_all_batches, (3 * n_max_batch_size * n_my_batches_work_h), local_dVxc_drho_all_batches__, 1);
    _FW_(double, H_param.local_rho_gradient, (3 * H_param.n_spin * n_max_batch_size), local_rho_gradient__, hipMemcpyHostToDevice);
    _FW_(double, H_param.first_order_gradient_rho, (3 * H_param.n_spin * n_max_batch_size), first_order_gradient_rho__, hipMemcpyHostToDevice);

    // H batches
    _FW_(int, MV(opencl_util, batches_size_h), n_my_batches_work_h, batches_size_H, 1);
    _FW_(int, MV(opencl_util, batches_batch_n_compute_h), n_my_batches_work_h, batches_batch_n_compute_H, 1);
    _FW_(int, MV(opencl_util, batches_batch_i_basis_h), n_max_compute_dens *n_my_batches_work_h, batches_batch_i_basis_H, 1);
    _FW_(double, MV(opencl_util, batches_points_coords_h), 3 * n_max_batch_size * n_my_batches_work_h, batches_points_coords_H, 1);

}
void h_begin_0_()
{
    if (h_begin_0_finished)
        return;
    h_begin_0_finished = 1;

    H_first_begin();
    H_first_begin_finished = 0;
    struct timeval start, end, start1, end1;
    hipError_t error;
    // if (hcnt == 0)
    // {
    // H param
    gettimeofday(&start, NULL);
    float H_time, H_time1;
    hipEvent_t h_start, h_stop;
    hipEventCreate(&h_start);
    hipEventCreate(&h_stop);

    hipError_t hip_err;

    // hip_err = hipHostRegister(&hip_buf_com.basis_l_max__,sizeof(int) * n_species,hipHostRegisterDefault);
    hip_err = hipMalloc(&hip_buf_com.basis_l_max__, sizeof(int) * n_species);
    hip_err = hipMalloc(&hip_buf_com.first_order_H__, sizeof(double) * (H_param.n_matrix_size * H_param.n_spin));
    hip_err = hipMalloc(&hip_buf_com.batches_batch_i_basis_h__, sizeof(int) * 1);
 
    // // cpy
    hip_err = hipMemcpy(hip_buf_com.basis_l_max__, H_param.basis_l_max, sizeof(int) * n_species, hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.first_order_H__, H_param.first_order_H, sizeof(double) * (H_param.n_matrix_size * H_param.n_spin), hipMemcpyHostToDevice);

    if (hcnt == 0)
    {
        hip_err = hipHostRegister(MV(opencl_util, batches_size_h), sizeof(int) * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_batch_n_compute_h), sizeof(int) * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_batch_i_basis_h), sizeof(int) * n_max_compute_dens * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(MV(opencl_util, batches_points_coords_h), sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_h, hipHostRegisterDefault);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_size_H, MV(opencl_util, batches_size_h), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_batch_n_compute_H, MV(opencl_util, batches_batch_n_compute_h), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_batch_i_basis_H, MV(opencl_util, batches_batch_i_basis_h), 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batches_points_coords_H, MV(opencl_util, batches_points_coords_h), 0);

        // hipEventRecord(h_stop);
        // hipEventSynchronize(h_stop);
        // hipEventElapsedTime(&regtime, h_start, h_stop);
        // hipEventRecord(h_start);

        hip_err = hipHostRegister(H_param.local_first_order_rho_all_batches, sizeof(double) * (H_param.n_spin * n_max_batch_size * n_my_batches_work_h), hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.local_first_order_potential_all_batches, sizeof(double) * (n_max_batch_size * n_my_batches_work_h), hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.local_dVxc_drho_all_batches, sizeof(double) * (3 * n_max_batch_size * n_my_batches_work_h), hipHostRegisterDefault);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.local_first_order_rho_all_batches__, H_param.local_first_order_rho_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.local_first_order_potential_all_batches__, H_param.local_first_order_potential_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.local_dVxc_drho_all_batches__, H_param.local_dVxc_drho_all_batches, 0);

        // change memcpy to register
        // hip_err = hipHostRegister(H_param.basis_l_max, sizeof(int) * n_my_batches_work_h, hipHostRegisterDefault);
        // hip_err = hipHostRegister(H_param.first_order_H, sizeof(double) * (H_param.n_matrix_size * H_param.n_spin), hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.n_points_all_batches, sizeof(int) * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.n_batch_centers_all_batches, sizeof(int) * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.batch_center_all_batches, sizeof(int) * max_n_batch_centers * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.ins_idx_all_batches, sizeof(int) * H_param.n_basis_local * n_my_batches_work_h, hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.partition_all_batches, sizeof(double) * (n_max_batch_size * n_my_batches_work_h), hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.local_potential_parts_all_points, sizeof(double) * 1, hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.local_rho_gradient, sizeof(double) * (3 * H_param.n_spin * n_max_batch_size), hipHostRegisterDefault);
        hip_err = hipHostRegister(H_param.first_order_gradient_rho, sizeof(double) * (3 * H_param.n_spin * n_max_batch_size), hipHostRegisterDefault);

        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.basis_l_max__, H_param.basis_l_max, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.first_order_H__, H_param.first_order_H, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.n_points_all_batches_H__, H_param.n_points_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.n_batch_centers_all_batches_H__, H_param.n_batch_centers_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.batch_center_all_batches_H__, H_param.batch_center_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.ins_idx_all_batches_H__, H_param.ins_idx_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.partition_all_batches__, H_param.partition_all_batches, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.local_potential_parts_all_points__, H_param.local_potential_parts_all_points, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.local_rho_gradient__, H_param.local_rho_gradient, 0);
        // hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.first_order_gradient_rho__, H_param.first_order_gradient_rho, 0);
    // }

        hip_err = hipMalloc(&hip_buf_com.batches_size_H,sizeof(int) * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.batches_batch_n_compute_H,sizeof(int) * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.batches_batch_i_basis_H,sizeof(int) * n_max_compute_dens * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.batches_points_coords_H,sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_h);
        
        hip_err = hipMalloc(&hip_buf_com.local_first_order_rho_all_batches__,sizeof(double) * (H_param.n_spin * n_max_batch_size * n_my_batches_work_h));
        hip_err = hipMalloc(&hip_buf_com.local_first_order_potential_all_batches__,sizeof(double) * (n_max_batch_size * n_my_batches_work_h));
        hip_err = hipMalloc(&hip_buf_com.local_dVxc_drho_all_batches__,sizeof(double) * (3 * n_max_batch_size * n_my_batches_work_h));

        hip_err = hipMalloc(&hip_buf_com.n_points_all_batches_H__,sizeof(int) * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.n_batch_centers_all_batches_H__,sizeof(int) * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.batch_center_all_batches_H__,sizeof(int) * max_n_batch_centers * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.ins_idx_all_batches_H__,sizeof(int) * H_param.n_basis_local * n_my_batches_work_h);
        hip_err = hipMalloc(&hip_buf_com.partition_all_batches__,sizeof(double) * (n_max_batch_size * n_my_batches_work_h));
        hip_err = hipMalloc(&hip_buf_com.local_potential_parts_all_points__,sizeof(double) * 1);
        hip_err = hipMalloc(&hip_buf_com.local_rho_gradient__,sizeof(double) * (3 * H_param.n_spin * n_max_batch_size));
        hip_err = hipMalloc(&hip_buf_com.first_order_gradient_rho__,sizeof(double) * (3 * H_param.n_spin * n_max_batch_size));
    }

    //��memcpy���м�ʱ
    hipEvent_t Memcpy_time_start,Memcpy_time_end;
    float Memcpy_time;
    int Data_size;

    hipEventCreate(&Memcpy_time_start);
    hipEventCreate(&Memcpy_time_end);
    hipEventRecord(Memcpy_time_start);

    hip_err = hipMemcpy(hip_buf_com.batches_size_H, MV(opencl_util, batches_size_h),sizeof(int) * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_batch_n_compute_H, MV(opencl_util, batches_batch_n_compute_h),sizeof(int) * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_batch_i_basis_H, MV(opencl_util, batches_batch_i_basis_h),sizeof(int) * n_max_compute_dens * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batches_points_coords_H, MV(opencl_util, batches_points_coords_h),sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.local_first_order_rho_all_batches__, H_param.local_first_order_rho_all_batches, sizeof(double) * (H_param.n_spin * n_max_batch_size * n_my_batches_work_h),hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.local_first_order_potential_all_batches__, H_param.local_first_order_potential_all_batches, sizeof(double) * (n_max_batch_size * n_my_batches_work_h),hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.local_dVxc_drho_all_batches__, H_param.local_dVxc_drho_all_batches,sizeof(double) * (3 * n_max_batch_size * n_my_batches_work_h),hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.n_points_all_batches_H__, H_param.n_points_all_batches,sizeof(int) * n_my_batches_work_h,hipMemcpyHostToDevice);
    
    hip_err = hipMemcpy(hip_buf_com.n_batch_centers_all_batches_H__, H_param.n_batch_centers_all_batches, sizeof(int) * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.batch_center_all_batches_H__, H_param.batch_center_all_batches, sizeof(int) * max_n_batch_centers * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.ins_idx_all_batches_H__, H_param.ins_idx_all_batches, sizeof(int) * H_param.n_basis_local * n_my_batches_work_h,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.partition_all_batches__, H_param.partition_all_batches, sizeof(double) * (n_max_batch_size * n_my_batches_work_h),hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.local_potential_parts_all_points__, H_param.local_potential_parts_all_points, sizeof(double) * 1,hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.local_rho_gradient__, H_param.local_rho_gradient, sizeof(double) * (3 * H_param.n_spin * n_max_batch_size),hipMemcpyHostToDevice);
    hip_err = hipMemcpy(hip_buf_com.first_order_gradient_rho__, H_param.first_order_gradient_rho, sizeof(double) * (3 * H_param.n_spin * n_max_batch_size),hipMemcpyHostToDevice);
    hipEventRecord(Memcpy_time_end);
    hipEventSynchronize(Memcpy_time_end);
    hipEventElapsedTime(&Memcpy_time,Memcpy_time_start,Memcpy_time_end);
    
    Data_size = sizeof(int) * n_my_batches_work_h + sizeof(int) * n_my_batches_work_h + sizeof(int) * n_max_compute_dens * n_my_batches_work_h + 
                sizeof(double) * 3 * n_max_batch_size * n_my_batches_work_h + sizeof(double) * (H_param.n_spin * n_max_batch_size * n_my_batches_work_h) +
                sizeof(double) * (n_max_batch_size * n_my_batches_work_h) + sizeof(double) * (3 * n_max_batch_size * n_my_batches_work_h) + 
                sizeof(int) * n_my_batches_work_h + sizeof(int) * max_n_batch_centers * n_my_batches_work_h + sizeof(int) * H_param.n_basis_local * n_my_batches_work_h + 
                sizeof(double) * (n_max_batch_size * n_my_batches_work_h) + sizeof(double) * 1 + sizeof(double) * (3 * H_param.n_spin * n_max_batch_size) +
                sizeof(double) * (3 * H_param.n_spin * n_max_batch_size);

    float thery_time = 0.09 + 7.84259e-08 * Data_size + 0.01;
    std::ofstream outFileHTime;
    std::string filename;
    outFileHTime.open("outFileHTime.csv", std::ios::app);
     if (outFileHTime.is_open())
    {
        if (myid == 0)
        {
            outFileHTime <<"myid"
                         <<","
                         <<"data_size"
                         <<","
                         <<"actual time"
                         <<","
                         <<"theory_time"
                         <<std::endl;
            outFileHTime<<myid<<","<<Data_size<<","<<Memcpy_time<<","<<thery_time<<std::endl;   
        }
    }
    // batches_size_H_time_theory = cal_theory_time(n_my_batches_work_h,sizeof(int));


    // }

    // hipEventRecord(h_stop);
    // hipEventSynchronize(h_stop);
    // hipEventElapsedTime(&getpointertime, h_start, h_stop);
    // }
    // hipEventRecord(h_stop);
    // hipEventSynchronize(h_stop);
    // hipEventElapsedTime(&cpytime, h_start, h_stop);
    double gh2d = 7.84259e-8;
    double predict_cpy = 0.009 + 0.01 + gh2d * (n_max_batch_size * n_my_batches_work_h * 8);
    // std::ofstream outFileH1;
    // outFileH1.open("Memcpytime.csv", std::ios::app);
    // if (outFileH1.is_open())
    // {
    //     if (myid == 0)
    //     {
    //         outFileH1 << "myid"
    //                   << ","
    //                   << "hcnt"
    //                   << ","
    //                   << "malloctime"
    //                   << ","
    //                   << "cpytime"
    //                   << ","
    //                   << "predict_cpy"
    //                   << ","
    //                   << "getpointertime"
    //                   << std::endl;
    //         outFileH1 << myid << "," << hcnt << "," << malloctime << "," << cpytime << "," << predict_cpy << "," << getpointertime << std::endl;
    //         outFileH1.close();
    //     }
    //     else
    //     {
    //         sleep(1);
    //         outFileH1 << myid << "," << hcnt << "," << malloctime << "," << cpytime << "," << predict_cpy << "," << getpointertime << std::endl;
    //         outFileH1.close();
    //     }
    // }

    // }
    // H batches

    // gettimeofday(&end, NULL);
    // regtime = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000.0;
}
void hipfree_h_(){

        hipHostUnregister(MV(opencl_util, batches_size_h));
        hipHostUnregister(MV(opencl_util, batches_batch_n_compute_h));
        hipHostUnregister(MV(opencl_util, batches_batch_i_basis_h));
        hipHostUnregister(MV(opencl_util, batches_points_coords_h));


        hipHostUnregister(H_param.local_first_order_rho_all_batches);
        hipHostUnregister(H_param.local_first_order_potential_all_batches);
        hipHostUnregister(H_param.local_dVxc_drho_all_batches);
        
        hipHostUnregister(H_param.n_points_all_batches);
        hipHostUnregister(H_param.n_batch_centers_all_batches);
        hipHostUnregister(H_param.batch_center_all_batches);
        hipHostUnregister(H_param.ins_idx_all_batches);
        hipHostUnregister(H_param.partition_all_batches);
        hipHostUnregister(H_param.local_potential_parts_all_points);
        hipHostUnregister(H_param.local_rho_gradient);
        hipHostUnregister(H_param.first_order_gradient_rho);

        hipFree(hip_buf_com.batches_size_H);
        hipFree(hip_buf_com.batches_batch_n_compute_H);
        hipFree(hip_buf_com.batches_batch_i_basis_H);
        hipFree(hip_buf_com.batches_points_coords_H);

        hipFree(hip_buf_com.local_first_order_rho_all_batches__);
        hipFree(hip_buf_com.local_first_order_potential_all_batches__);
        hipFree(hip_buf_com.local_dVxc_drho_all_batches__);

        hipFree(hip_buf_com.n_points_all_batches_H__);
        hipFree(hip_buf_com.n_batch_centers_all_batches_H__);
        hipFree(hip_buf_com.batch_center_all_batches_H__);
        hipFree(hip_buf_com.ins_idx_all_batches_H__);
        hipFree(hip_buf_com.partition_all_batches__);
        hipFree(hip_buf_com.local_potential_parts_all_points__);
        hipFree(hip_buf_com.local_rho_gradient__);
        hipFree(hip_buf_com.first_order_gradient_rho__);
}

// ok
int n_new_batches_nums = 0;
int *new_batch_count;   // 存储每个逻辑batch中有几个原batch
int *new_batch_i_start; // 存储每个逻辑batch中起始原batch的i_batch

void h_begin_()
{

    hipEvent_t data_pre_start,data_pre_end,kernel_start,kernel_end,data_del_start,data_del_end,H_all_start,H_all_end;
    float data_pre_time,kernel_time,data_del_time,H_all_time;
    hipEventCreate(&data_pre_start);
    hipEventCreate(&data_pre_end);
    hipEventCreate(&kernel_start);
    hipEventCreate(&kernel_end);
    hipEventCreate(&data_del_start);
    hipEventCreate(&data_del_end);
    hipEventCreate(&H_all_start);
    hipEventCreate(&H_all_end);
    

    hipEventRecord(data_pre_start);
    hipEventRecord(H_all_start);
    
    // for (int kk = 0; kk < 1; kk++)
    // {
    // if (htimes == 0 && kk == 0)
    // {
    //     pre_run_();
    // }
    struct timeval start, end, start1, end1;
    long time_uses[2];
    // gettimeofday(&start, NULL);

    h_begin_0_();
    h_begin_0_finished = 0;
    // gettimeofday(&end, NULL);
    // time_uses[0] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // double cpytime = time_uses[0] / 1000.0;
    // long time_uses[32];
    // char *time_infos[32];
    // size_t time_index = 0;
    // gettimeofday(&start, NULL);
    hipError_t error;
    int arg_index;
    size_t localSize[] = {LOCAL_SIZE_H};                    // 覆盖前面的设�?
    size_t globalSize[] = {LOCAL_SIZE_H * S * MERGE_BATCH}; // 覆盖前面的设�?
    // size_t globalSize[] = {256 * 1}; // 覆盖前面的设�?
    dim3 blockDim(LOCAL_SIZE_H, 1, 1);
    dim3 gridDim(S, 1, 1);

    
    // printf("n_max_batch_size=%d\n", n_max_batch_size);
    // H tmp

    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab_sq__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (3 * n_max_compute_atoms), dir_tab__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), atom_index__, 0);
    // _FW_(int, NULL, globalSize[0] * (n_centers_integrals), atom_index_inv__, 0);
    _FW_(int, NULL, 1, atom_index_inv__, 0);
    _FW_(int, NULL, 1, i_basis_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_basis_fns * (n_max_compute_atoms + 1)), i_basis_fns_inv__, 0);
    _FW_(int, NULL, 1, i_atom_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_start__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_end__, 0);
    _FW_(double, NULL, globalSize[0] * n_max_compute_atoms, one_over_dist_tab__, 0);
    _FW_(int, NULL, globalSize[0] * n_max_compute_atoms, rad_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), wave_index__, 0);
    _FW_(int, NULL, 1, l_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), l_count__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), fn_atom__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_ham), zero_index_point__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 256 + 16 * n_max_compute_ham,
         wave__, 0); // 多加 128 为了避免 TILE 后越界, 长宽按 128 对齐
    _FW_(double, NULL, 1, first_order_density_matrix_con__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), i_r__, 0);
    _FW_(double, NULL, globalSize[0] * (4 * n_max_compute_atoms), trigonom_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_fns_ham), radial_wave__, 0);
    _FW_(double, NULL, globalSize[0] * (n_basis_fns), spline_array_aux__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms * n_basis_fns), aux_radial__, 0);
    _FW_(double, NULL, globalSize[0] * ((H_param.l_ylm_max + 1) * (H_param.l_ylm_max + 1) * n_max_compute_atoms), ylm_tab__, 0);
    _FW_(double, NULL, globalSize[0] * ((H_param.l_ylm_max + 1) * (H_param.l_ylm_max + 1) * n_max_compute_atoms), dylm_dtheta_tab__, 0);
    _FW_(double, NULL, 100, scaled_dylm_dphi_tab__, 0); // 反正这个和上面那个暂时无用
    _FW_(double, NULL, 1, kinetic_wave__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) + 256, grid_coord__, 0); // 长宽按 128 对齐
    // _FW_(double, NULL, globalSize[0] * n_max_compute_ham * H_param.n_spin, H_times_psi__, 0);
    _FW_(double, NULL, 1, H_times_psi__, 0);
    _FW_(double, NULL, 1, T_plus_V__, 0);
    // _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms * n_basis_fns), T_plus_V__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * ((n_max_batch_size+127)/128*128) * n_max_compute_ham, contract__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * ((n_max_batch_size+127)/128*128) * n_max_compute_ham, wave_t__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * n_max_compute_ham * n_max_compute_ham * H_param.n_spin, first_order_H_dense__, 0);

    // _FW_(double, NULL, 1, contract__, 0);
    // _FW_(double, NULL, 1, wave_t__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 256 + 16 * n_max_compute_ham,
         contract__, 0); // 多加 128 为了避免 TILE 后越界, 长宽按 128 对齐
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 256 + 16 * n_max_compute_ham,
         wave_t__, 0); // 多加 128 为了避免 TILE 后越越界, 长宽按 128 对齐
    _FW_(double, NULL, (globalSize[0] / localSize[0] + 1) * n_max_compute_ham * n_max_compute_ham + 128 * H_param.n_spin, first_order_H_dense__, 0);
    // H param
    arg_index = 0;
    void *args[100];
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.j_coord);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_spin);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.l_ylm_max);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_basis_local);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_matrix_size);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l_max__);
    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_points_all_batches_H__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_batch_centers_all_batches_H__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_center_all_batches_H__);
    //----
    // setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_point_to_i_full_point__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ins_idx_all_batches_H__);

    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_i_basis_h__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.partition_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_H__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_potential_parts_all_points__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_first_order_rho_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_first_order_potential_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_dVxc_drho_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_rho_gradient__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_gradient_rho__);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    arg_index = 19;
    setKernelArgs(args, arg_index++, sizeof(int), &n_centers);
    //----
    setKernelArgs(args, arg_index++, sizeof(int), &n_centers_integrals);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_fns_ham);
    setKernelArgs(args, arg_index++, sizeof(int), &n_basis_fns);
    //----

    setKernelArgs(args, arg_index++, sizeof(int), &n_centers_basis_I);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_grid);

    //----
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_atoms);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_ham);
    //----

    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_dens);

    //----
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_batch_size);
    //----

    setKernelArgs(args, arg_index++, sizeof(int), &index_hamiltonian_dim2);
    setKernelArgs(args, arg_index++, sizeof(int), &position_in_hamiltonian_dim1);
    setKernelArgs(args, arg_index++, sizeof(int), &position_in_hamiltonian_dim2);
    setKernelArgs(args, arg_index++, sizeof(int), &column_index_hamiltonian_size);
    // _CHK_(error, hipSuccess);

    arg_index = 33;
    // int test_batch = 1;
    // setKernelArgs(args, arg_index++, sizeof(int), &test_batch);

    //----
    setKernelArgs(args, arg_index++, sizeof(int), &n_my_batches_work_h);
    //----

    setKernelArgs(args, arg_index++, sizeof(int), &n_full_points_work_h);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    arg_index = 35;
    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_atom);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species_center);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_cell);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_basis);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_center);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_basis_integrals);
    //----

    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_hamiltonian);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.position_in_hamiltonian);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.column_index_hamiltonian);

    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coords_center);
    //----

    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_grid);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid_min);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.log_r_grid_inc);

    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.perm_basis_fns_spl);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.outer_radius_sq);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_radius_sq);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_start_spl);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_atom);
    //----

    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_wave_ordered);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_kinetic_ordered); // new
    // _CHK_(error, hipSuccess);
    // H batches
    arg_index = 57;
    // setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_size_H);
    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_n_compute_H);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_i_basis_H);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_points_coords_H);
    //----
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    // H tmp
    arg_index = 60;
    //----
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab_sq__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dir_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index_inv__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns_inv__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_atom_fns__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_start__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_end__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.one_over_dist_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rad_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_count__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.fn_atom__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.zero_index_point__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_density_matrix_con__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_r__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.trigonom_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.radial_wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_aux__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.aux_radial__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ylm_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dylm_dtheta_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.scaled_dylm_dphi_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.kinetic_wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.grid_coord__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.H_times_psi__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.T_plus_V__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.contract__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave_t__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_H_dense__);
    //----
    setKernelArgs(args, arg_index++, sizeof(int), &max_n_batch_centers);
    // gettimeofday(&end, NULL);
    // time_uses[0] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // double malloc_time = time_uses[0] / 1000.0;
    // if (htimes == 0)
    // {

    //     // std::vector<Batch> batches;
    //     // // for (int i = 0; i < n_my_batches_work_h; ++i)
    //     // // {
    //     // //     batches.push_back({i, MV(opencl_util, batches_batch_n_compute_h)[i]});
    //     // // }
    //     // // for (int i = 0; i < n_my_batches_work_h; ++i)
    //     // // {
    //     // //     batches.push_back({i, static_cast<float>(H_param.n_batch_centers_all_batches[i])});
    //     // // }
    //     // // for (int i = 0; i < n_my_batches_work_h; ++i)
    //     // // {
    //     // //     batches.push_back({i, batch_time[i]});
    //     // // }
    //     // for (int i = 0; i < n_my_batches_work_h; ++i)
    //     // {
    //     //     double t_batch = get_gpu_time_local(center_all_batches, i);
    //     //     gpu_batches_times_H += t_batch;
    //     //     batches.push_back({i, t_batch});
    //     // }
    //     // std::vector<Bucket> buckets = distributeBatchesBalanced(batches);
    //     // std::ofstream outFileH1;
    //     // std::string filename1;
    //     // filename1 = "blocks-newdis.csv";
    //     // outFileH1.open(filename1, std::ios::app);
    //     // for (int i_bk = 0; i_bk < S; i_bk++)
    //     // {
    //     //     outFileH1 << myid << "," << i_bk << "," << buckets[i_bk].sum << std::endl;
    //     // }
    //     // outFileH1.close();
    //     // int new_id = 0;
    //     // for (int i = 0; i < S; i++)
    //     // {
    //     //     for (std::vector<Batch>::iterator it = buckets[i].batches.begin(); it != buckets[i].batches.end(); ++it)
    //     //     {
    //     //         block[new_id] = it->id;
    //     //         count_batches[i]++;
    //     //         new_id++;
    //     //     }
    //     // }
    // }
    // htimes++;

    // _FW_(int, block, n_my_batches_work_h, d_block_h, hipMemcpyHostToDevice);
    // _FW_(int, count_batches, S, d_count_batches_h, hipMemcpyHostToDevice);
    // setKernelArgs(args, arg_index++, sizeof(int), &hip_buf_com.d_block_h);
    // setKernelArgs(args, arg_index++, sizeof(int), &hip_buf_com.d_count_batches_h);
    // int n_new_batches_nums = 0;
    if (hcnt == 0)
    {
        new_batch_count = new int[n_my_batches_work_h];   // 存储每个逻辑batch中有几个原batch
        new_batch_i_start = new int[n_my_batches_work_h]; // 存储每个逻辑batch中起始原batch的i_batch

        for (int i_batch = 1; i_batch <= n_my_batches_work_h;)
        {
            int n_points_sum = H_param.n_points_all_batches[i_batch - 1];
            int count = 1;
            int i_batch_tmp = i_batch + 1; // 从下一个batch开始累�?

            while (i_batch_tmp <= n_my_batches_work_h && (n_points_sum + H_param.n_points_all_batches[i_batch_tmp - 1]) <= 256 && count < MERGE_BATCH)
            {
                n_points_sum += H_param.n_points_all_batches[i_batch_tmp - 1];
                count++;
                i_batch_tmp++;
            }

            // 记录当前逻辑batch的信息
            new_batch_count[n_new_batches_nums] = count;
            new_batch_i_start[n_new_batches_nums] = i_batch;

            // 移动到下一个未处理的batch
            i_batch += count;

            // 增加逻辑batch的计数
            n_new_batches_nums++;
        }
        _FW_(int, new_batch_count, n_my_batches_work_h, d_new_batch_count, hipMemcpyHostToDevice);
        _FW_(int, new_batch_i_start, n_my_batches_work_h, d_new_batch_i_start, hipMemcpyHostToDevice);
    }

    setKernelArgs(args, 95, sizeof(hipDeviceptr_t), &hip_buf_com.d_new_batch_count);
    setKernelArgs(args, 96, sizeof(hipDeviceptr_t), &hip_buf_com.d_new_batch_i_start);
    setKernelArgs(args, 97, sizeof(int), &n_new_batches_nums);

    // float H_time, H_time1;
    // hipEvent_t h_start, h_stop;
    // hipEventCreate(&h_start);
    // hipEventCreate(&h_stop);

    // std::ofstream outFileH1;
    // std::string filename1;
    // filename1 = "Hdatas-newdis.csv";
    // hipEventRecord(h_start);
    hipEventRecord(data_pre_end);
    hipEventSynchronize(data_pre_end);
    hipEventElapsedTime(&data_pre_time, data_pre_start, data_pre_end);
    std::cout<<"myID:"<<myid<<" H_kernel running started"<<std::endl;
    if(hcnt == -1){
        size_t localSize_pre_run[] = {256};               // 覆盖前面的设置
        size_t globalSize_pre_run[] = {256 * 1}; // 覆盖前面的设置
        dim3 blockDim_pre(localSize_pre_run[0], 1, 1);
        dim3 gridDim_pre(globalSize_pre_run[0] / blockDim.x, 1, 1);
        int row_size = 3 * n_max_compute_ham + 3 * n_max_compute_atoms * (n_basis_fns + 1) + 2 * n_max_compute_atoms;
        _FW_(int, NULL, localSize_pre_run[0] * row_size, diverge_matrix, 0);
        setKernelArgs(args, 98, sizeof(int), &hip_buf_com.diverge_matrix);
        for (int i_batch = 1; i_batch <= n_new_batches_nums; i_batch += 1)
        {
            setKernelArgs(args, 99, sizeof(int), &i_batch);
            hipMemset(hip_buf_com.diverge_matrix, 0, localSize[0] * row_size * sizeof(int));
            hipEvent_t h_first_begin,h_first_end;
            hipEventCreate(&h_first_begin);
            hipEventCreate(&h_first_end);
            float H_first_time;
        
            hipEventRecord(h_first_begin);
            error = hipLaunchKernel(reinterpret_cast<const void *>(&integrate_first_order_h_sub_tmp2_pre_), gridDim_pre, blockDim_pre, args, 0, 0);
            IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel failed");
            hipEventRecord(h_first_end);
            hipEventSynchronize(h_first_end);
            hipEventElapsedTime(&H_first_time, h_first_begin, h_first_end);

            int *diverge_matrix = new int[localSize[0] * row_size];
            hipMemcpy(diverge_matrix, hip_buf_com.diverge_matrix, sizeof(int) * localSize[0] * row_size, hipMemcpyDeviceToHost);
            std::vector<int> reduced_matrix1(4 * row_size, 0);
            std::vector<int> reduced_matrix2(4 * 11, 0);
            reduced_matrix1 = reduceMatrix(diverge_matrix, row_size);
            reduced_matrix2 = reduceMatrixSecondStep(reduced_matrix1, row_size);
            double args[] = {0.13, 0.16, 0.09, 0.06, 0.15, 0.07, 0.10, 0.08, 0.14, 0.09, 0.04, 0.11, 0.05, 0.08, 0.07, 0.03, 0.04, 0.06, 0.10, 0.06, 0.13, 0.09, 0.10, 0.12, 0.08, 0.05, 0.07, 0.03, 0.10, 0.06, 0.08, 0.07, 0.04, 0.11, 0.08, 0.05, 0.12, 0.09, 0.10, 0.06, 0.08, 0.07, 0.04, 0.11};
            std::ofstream outFileIFMatrix;
            outFileIFMatrix.open("outFileIFMatrix1.csv",std::ios::app);
            std::string unique_id = generate_id_concatenation(&center_all_batches[(i_batch - 1) * 3]);
                double thery_time = 0;
                int outer_loop = (n_max_compute_ham/16*4)*(n_max_compute_ham/16*8);
                long long data_scale = outer_loop*256*4+outer_loop*256*8 + outer_loop*256*16*4 +  outer_loop*256*16*8
                                    + outer_loop*4*8 + n_max_compute_ham;
                outFileIFMatrix<<"unique_id"<<","<<"actual_time"<<","<<"thery_time"<<","<<"deviation"<<","<<"datascale"<<"\n";
                outFileIFMatrix<<unique_id<<","<<H_first_time<<","<<thery_time<<","<<abs(thery_time-H_first_time)/H_first_time<<","<<n_max_compute_ham<<"\n";
                for (int i = 0; i < 4; ++i)
                { // 现在每行有 row_size 个元素
                    for (int j = 0; j < 11; ++j)
                    {
                        // 计算转置矩阵中元素的索引
                        int index = i * 11 + j;
                        // outFileH1 << reduced_matrix1[index];
                        outFileIFMatrix << reduced_matrix2[index];
                        thery_time += args[index]*reduced_matrix2[index];
                        // outFileH1 << diverge_matrix[index];
                        if (j < 10)
                        {
                            outFileIFMatrix << ",";
                        }
                    }
                    outFileIFMatrix << "\n";
                }
              
                outFileIFMatrix.close();
        }
            
    }
    else{
    
        hipEventRecord(kernel_start);
        error = hipLaunchKernel(reinterpret_cast<const void *>(&integrate_first_order_h_sub_tmp2_), gridDim, blockDim, args, 0, 0);
        IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel failed");
        hipEventRecord(kernel_end);
        hipEventSynchronize(kernel_end);
        hipEventElapsedTime(&kernel_time, kernel_start, kernel_end);
        // hipEventRecord(data_del_start)
    }

    // hipEventDestroy(data_pre_start);
    // hipEventDestroy(data_pre_end);
    // hipEventDestroy(kernel_start);
    // hipEventDestroy(kernel_end);
    // hipEventDestroy(data_del_start);
    // hipEventDestroy(H_all_start);
    // hipEventDestroy(H_all_end);
    std::cout<<"myID:"<<myid<<" H_kernel running succeed"<<std::endl;


    
    // hipEventRecord(h_stop);
    // hipEventSynchronize(h_stop);
    // hipEventElapsedTime(&H_time1, h_start, h_stop);
    // outFileH1.open(filename1, std::ios::app);

    // if (kk == 0)
    // {
    hipMemcpy(H_param.first_order_H, hip_buf_com.first_order_H__, sizeof(double) * (H_param.n_matrix_size * H_param.n_spin), hipMemcpyDeviceToHost);
    // }

    // time_infos[time_index++] = "H kernel and readbuf";

    // delete[] new_batch_count;
    // delete[] new_batch_i_start;
    // hipFree(hip_buf_com.d_new_batch_count);
    // hipFree(hip_buf_com.d_new_batch_i_start);
    // hipEventDestroy(h_start);
    // hipEventDestroy(h_stop);
    // 可能有问�?
    {
        // if (hcnt == 0)
        // {
        // hipHostUnregister(MV(opencl_util, batches_size_h));
        // hipFree(hip_buf_com.batches_size_H);
        // // 可能有问�?
        // hipHostUnregister(MV(opencl_util, batches_batch_n_compute_h));
        // hipFree(hip_buf_com.batches_batch_n_compute_H);
        // // 可能有问�?
        // hipHostUnregister(MV(opencl_util, batches_batch_i_basis_h));
        // hipFree(hip_buf_com.batches_batch_i_basis_H);
        // // 可能有问�?
        // hipHostUnregister(MV(opencl_util, batches_points_coords_h));
        // hipFree(hip_buf_com.batches_points_coords_H);
        // 可能有问�?
        // hipHostUnregister(H_param.local_first_order_rho_all_batches);
        // hipFree(hip_buf_com.local_first_order_rho_all_batches__);

        // // 可能有问�?
        // hipHostUnregister(H_param.local_first_order_potential_all_batches);
        // hipFree(hip_buf_com.local_first_order_potential_all_batches__);

        // // 可能有问�?
        // hipHostUnregister(H_param.local_dVxc_drho_all_batches);
        // hipFree(hip_buf_com.local_dVxc_drho_all_batches__);
        // }

        // cpyfree
        //  // hipFree(hip_buf_com.batch_point_to_i_full_point__);
        hipFree(hip_buf_com.basis_l_max__);
        hipFree(hip_buf_com.batches_batch_i_basis_h__);
        hipFree(hip_buf_com.first_order_H__);
        // hipFree(hip_buf_com.n_points_all_batches__);
        // hipFree(hip_buf_com.n_batch_centers_all_batches__);
        // hipFree(hip_buf_com.batch_center_all_batches__);
        // hipFree(hip_buf_com.ins_idx_all_batches__);
        // hipFree(hip_buf_com.partition_all_batches__);
        // hipFree(hip_buf_com.local_potential_parts_all_points__);
        // hipFree(hip_buf_com.local_rho_gradient__);
        // hipFree(hip_buf_com.first_order_gradient_rho__);
    }

    hipFree(hip_buf_com.dist_tab_sq__);
    hipFree(hip_buf_com.dist_tab__);
    hipFree(hip_buf_com.dir_tab__);
    hipFree(hip_buf_com.atom_index__);
    hipFree(hip_buf_com.atom_index_inv__);
    hipFree(hip_buf_com.i_basis_fns__);
    hipFree(hip_buf_com.i_basis_fns_inv__);
    hipFree(hip_buf_com.i_atom_fns__);
    hipFree(hip_buf_com.spline_array_start__);
    hipFree(hip_buf_com.spline_array_end__);
    hipFree(hip_buf_com.one_over_dist_tab__);
    hipFree(hip_buf_com.rad_index__);
    hipFree(hip_buf_com.wave_index__);
    hipFree(hip_buf_com.l_index__);
    hipFree(hip_buf_com.l_count__);
    hipFree(hip_buf_com.fn_atom__);
    hipFree(hip_buf_com.zero_index_point__);
    hipFree(hip_buf_com.wave__);
    hipFree(hip_buf_com.first_order_density_matrix_con__);
    hipFree(hip_buf_com.i_r__);
    hipFree(hip_buf_com.trigonom_tab__);
    hipFree(hip_buf_com.radial_wave__);
    hipFree(hip_buf_com.spline_array_aux__);
    hipFree(hip_buf_com.aux_radial__);
    hipFree(hip_buf_com.ylm_tab__);
    hipFree(hip_buf_com.dylm_dtheta_tab__);
    hipFree(hip_buf_com.scaled_dylm_dphi_tab__);
    hipFree(hip_buf_com.kinetic_wave__);
    hipFree(hip_buf_com.grid_coord__);
    hipFree(hip_buf_com.H_times_psi__);
    hipFree(hip_buf_com.T_plus_V__);
    hipFree(hip_buf_com.contract__);
    hipFree(hip_buf_com.wave_t__);
    hipFree(hip_buf_com.first_order_H_dense__);

    



    // hipFree(hip_buf_com.d_block_h);
    // hipFree(hip_buf_com.d_count_batches_h);

    // if (myid == 0)
    //     m_save_check_h_(H_param.first_order_H, &(H_param.n_spin), &(H_param.n_matrix_size));

    // printf("End\n");

    // hip_common_buffer_free_();
    // // Int 类型元素的总数
    // size_t total_int_elements_h = n_species + n_my_batches_work_h * (4 + max_n_batch_centers + H_param.n_basis_local + n_max_compute_dens) + 1;

    // // Double 类型元素的总数
    // //  int total_double_elements = n_max_batch_size * n_my_batches_work_h + 3 * n_max_batch_size * n_my_batches_work_h + H_param.n_matrix_size * H_param.n_spin + 1 + H_param.n_spin * n_max_batch_size * n_my_batches_work_h + 4 * n_max_batch_size * n_my_batches_work_h + 6 * H_param.n_spin * n_max_batch_size;
    // size_t total_double_elements_h = H_param.n_matrix_size * H_param.n_spin + H_param.n_spin * n_max_batch_size * n_my_batches_work_h + 6 * H_param.n_spin * n_max_batch_size + 8 * n_max_batch_size * n_my_batches_work_h + 1;
    // size_t total_int_bytes_h = 4 * total_int_elements_h;
    // size_t total_double_bytes_h = 8 * total_double_elements_h;
    // double gh2d = 7.84259e-8;
    // double predict_cpy = 0.009 + 0.01 + gh2d * (total_int_bytes_h + total_double_bytes_h);
    // gettimeofday(&start, NULL);
    // if (outFileH1.is_open())
    // {
    //     if (myid == 0)
    //     {
    //         outFileH1 << "myid"
    //                   << ","
    //                   << "hcnt"
    //                   << ","
    //                   << "n_my_batches_work_h"
    //                   << ","
    //                   << "n_new_batches_nums"
    //                   << ","
    //                   << "n_max_batch_size"
    //                   << ","
    //                   << "n_max_compute_dens"
    //                   << ","
    //                   << "H_kernel_time"
    //                   << ","
    //                   << "cpytime"
    //                   << ","
    //                   << "regtime" << std::endl;
    //         outFileH1 << myid << "," << hcnt << "," << n_my_batches_work_h << "," << n_new_batches_nums << "," << n_max_batch_size << "," << n_max_compute_dens << "," << H_time1 << "," << cpytime << "," << regtime << "," << predict_cpy << "," << total_int_bytes_h << "," << total_double_elements_h << "," << n_species << "," << n_my_batches_work_h << "," << max_n_batch_centers << "," << n_max_compute_dens << "," << H_param.n_basis_local << "," << H_param.n_matrix_size << "," << H_param.n_spin << std::endl;
    //         outFileH1.close();
    //     }
    //     else
    //     {
    //         sleep(1);
    //         outFileH1 << myid << "," << hcnt << "," << n_my_batches_work_h << "," << n_new_batches_nums << "," << n_max_batch_size << "," << n_max_compute_dens << "," << H_time1 << "," << cpytime << "," << regtime << "," << predict_cpy << "," << total_int_bytes_h << "," << total_double_elements_h << "," << n_species << "," << n_my_batches_work_h << "," << max_n_batch_centers << "," << n_max_compute_dens << "," << H_param.n_basis_local << "," << H_param.n_matrix_size << "," << H_param.n_spin << std::endl;
    //         outFileH1.close();
    //     }
    // }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    // gettimeofday(&end, NULL);
    // h_write_time = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000000.0;
    hcnt++;
    // if(hcnt == 6){
    //     hipfree_h_();
    // }
    hipEventRecord(data_del_end);
    hipEventSynchronize(data_del_end);
    hipEventElapsedTime(&data_del_time, data_del_start, data_del_end);

    hipEventRecord(H_all_end);
    hipEventSynchronize(H_all_end);
    hipEventElapsedTime(&H_all_time, H_all_start, H_all_end);

    std::ofstream load_balance_file;
    load_balance_file.open("load_balance_h.csv", std::ios::app);

    if (load_balance_file.is_open())
    {
        if(myid == 0 && hcnt != 1){
        // outFileHTime<<"H prepare time"<<","<<"H kernel time"<<","<<"data delete time"<<"hcnt"<<std::endl;
        // outFileHTime<<data_pre_time<<","<<kernel_time<<","<<data_del_time<<","<<hcnt<<std::endl;
            load_balance_file<<H_all_time<<"\n";
        }
        
    }
    load_balance_file.close();

    std::ofstream outFileHTime;
    std::string filename;
    outFileHTime.open("outFileHTime.csv", std::ios::app);

    if (outFileHTime.is_open())
    {
        if(myid == 0){
        outFileHTime<<"H prepare time"<<","<<"H kernel time"<<","<<"data delete time"<<"hcnt"<<std::endl;
        outFileHTime<<data_pre_time<<","<<kernel_time<<","<<data_del_time<<","<<hcnt<<std::endl;
        }
        
    }
    outFileHTime.close();


    std::cout<<"myID:"<<myid<<" H_kernel delete succeed"<<std::endl;
}

namespace sum_up_pre
{
    void *args[30];
    int arg_index;
}

void sum_up_pre_processing_init_()
{
    hip_init_();
    // if (JIT_ENABLED)
    // {
    //     hip_init_();
    // }
    // else
    // {
    //     hip_device_init_();
    // }
    hip_common_buffer_init_();

    globalSize_sum_up_pre_proc[0] = 256 * localSize_sum_up_pre_proc[0]; // static !
                                                                        // globalSize_sum_up_pre_proc[0] = n_atoms * localSize_sum_up_pre_proc[0];  // dynamic !

    hipError_t error;

    _FW_(double, NULL, (l_pot_max + 1) * (l_pot_max + 1) * n_max_grid * (globalSize_sum_up_pre_proc[0] / localSize_sum_up_pre_proc[0]),
         angular_integral_log, 0);

    sum_up_pre::arg_index = 0;
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &n_max_radial);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &l_pot_max);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &n_max_spline);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &n_hartree_grid);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &n_atoms);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &n_max_grid);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &Adams_Moulton_integrator);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), &compensate_multipole_errors);

    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_hartree);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.multipole_radius_free);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_grid);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_radial);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid_inc);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.scale_radial);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_radial);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rho_multipole_h_p_s);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rho_multipole_index);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.compensation_norm);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.compensation_radius);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_hartree_potential);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_atom);

    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.angular_integral_log);
}

void sum_up_pre_processing_part_(int *n_coeff_hartree_, int *i_center_begin_, int *i_center_end_,
                                 hipDeviceptr_t *centers_rho_multipole_spl, hipDeviceptr_t *centers_delta_v_hart_part_spl, hipDeviceptr_t *i_center_to_centers_index,
                                 int debug)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // hip_buf_com.centers_rho_multipole_spl;
    // _FW_(double, NULL, (l_pot_max + 1) * (l_pot_max + 1) * n_max_spline * (n_max_radial + 2) * centers_tile_size, centers_rho_multipole_spl, CL_MEM_READ_WRITE);
    // _FW_(double, NULL, (l_pot_max + 1) * (l_pot_max + 1) * n_coeff_hartree * n_hartree_grid * centers_tile_size, centers_delta_v_hart_part_spl, CL_MEM_READ_WRITE);
    hipError_t error;
    sum_up_pre::arg_index = 24;

    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), centers_rho_multipole_spl);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), centers_delta_v_hart_part_spl);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), n_coeff_hartree_);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), i_center_begin_);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(int), i_center_end_);
    setKernelArgs(sum_up_pre::args, sum_up_pre::arg_index++, sizeof(hipDeviceptr_t), i_center_to_centers_index);

    dim3 blockDim(localSize_sum_up_pre_proc[0], 1, 1);
    dim3 gridDim(globalSize_sum_up_pre_proc[0] / blockDim.x, 1, 1);
    // dim3 blockDim(64, 1, 1);
    // dim3 gridDim(256, 1, 1);
    // if (myid == 0)
    //     std::cout << "---------------grid dims" << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;

    if (JIT_ENABLED)
    {
        error = hipModuleLaunchKernel(kernels[1], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, nullptr, sum_up_pre::args, nullptr);
    }
    else
    {
        error = hipLaunchKernel(reinterpret_cast<const void *>(&sum_up_whole_potential_shanghui_pre_proc_), gridDim, blockDim, sum_up_pre::args, 0, 0);
    }

    const char *errorString = hipGetErrorString(error);
    // std::cout << "Error code " << error << " corresponds to the following HIP error: " << errorString << std::endl;
    IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel sumup pre failed");
    hipDeviceSynchronize();

    // clEnqueueReadBuffer(cQ, centers_rho_multipole_spl, CL_TRUE, 0,
    //   sizeof(double) * (l_pot_max + 1) * (l_pot_max + 1) * n_max_spline * (n_max_radial + 2) * (*i_center_end_ - *i_center_begin_),
    //   param_centers_rho_multipole_spl, 1, &event, NULL);
    // clEnqueueReadBuffer(cQ, centers_delta_v_hart_part_spl, CL_TRUE, 0,
    //   sizeof(double) * (l_pot_max + 1) * (l_pot_max + 1) * n_coeff_hartree * n_hartree_grid * (*i_center_end_ - *i_center_begin_),
    //   param_centers_delta_v_hart_part_spl, 1, &event, NULL);

    gettimeofday(&end, NULL);
    long time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // if(myid == 0 && debug)
    //   printf("    rank%d, %s: %lf seconds\n", myid, __func__, time_use/1000000.0);
}

void sum_up_pre_processing_finish()
{
    hipFree(hip_buf_com.angular_integral_log);
}

namespace sum_up
{
    void *args[75];
    int arg_index;
}

void sum_up_first_begin()
{
    if (sum_up_first_begin_finished)
        return;
    sum_up_first_begin_finished = 1;

    hip_init_();
    // if (JIT_ENABLED)
    // {
    //     hip_init_();
    // }
    // else
    // {
    //     hip_device_init_();
    // }

    hip_common_buffer_init_();

    hipError_t error;
    int fast_ylm = 1;
    int new_ylm = 0;

    // TODO 这两没有释放
    // TODO WARNING 进程间协作可能没有处理这�?
    if (Fp_max_grid == 0)
    {
        _FW_(double, NULL, 1, Fp_function_spline_slice, 0);
        _FW_(double, NULL, 1, Fpc_function_spline_slice, 0);
    }
    else
    {
        _FW_(double, MV(hartree_f_p_functions, fp_function_spline), (lmax_Fp + 1) * n_max_spline * (Fp_max_grid), Fp_function_spline_slice,
             hipMemcpyHostToDevice);
        _FW_(double, MV(hartree_f_p_functions, fpc_function_spline), (lmax_Fp + 1) * n_max_spline * (Fp_max_grid), Fpc_function_spline_slice,
             hipMemcpyHostToDevice);
    }

    sum_up::arg_index = 11;
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_centers_hartree_potential);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_periodic);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_max_radial);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &l_pot_max);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_max_spline);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_hartree_grid);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_species);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_atoms);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_centers);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_max_batch_size);
    remember_arg_n_my_batches_work_sumup = sum_up::arg_index;
    sum_up::arg_index += 2;
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_my_batches_work_sumup);
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_full_points_work_sumup);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &use_hartree_non_periodic_ewald);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &hartree_fp_function_splines);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &fast_ylm);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &new_ylm);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &l_max_analytic_multipole);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &n_hartree_atoms);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &hartree_force_l_add);
    remember_arg_Fp_max_grid = sum_up::arg_index;
    sum_up::arg_index += 5;
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &Fp_max_grid);
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &lmax_Fp);
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(double), &Fp_grid_min);
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(double), &Fp_grid_inc);
    // =~error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(double), &Fp_grid_max);
    sum_up::arg_index = 35;
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_hartree_potential);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_atom);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species_center);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coords_center);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_hartree);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_grid);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_radial);
    remember_arg_index_1 = sum_up::arg_index;
    // error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_size_s);
    // error = setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_points_coords_s);
    sum_up::arg_index += 2;
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid_min);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.log_r_grid_inc);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.scale_radial);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_cc_lm_ijk);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_cc);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_ijk_max_cc);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.b0);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.b2);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.b4);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.b6);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.a_save);
    remember_arg_Fp_function_spline = sum_up::arg_index;
}

void sum_up_begin_0_()
{
    if (sum_up_begin_0_finished)
        return;
    sum_up_begin_0_finished = 1;

    sum_up_first_begin();

    // _FW_(double, sum_up_param.centers_rho_multipole_spl, (l_pot_max + 1) * (l_pot_max + 1) * n_max_spline * (n_max_radial + 2) * n_atoms,
    //      centers_rho_multipole_spl, hipMemcpyHostToDevice);
    // _FW_(double, sum_up_param.centers_delta_v_hart_part_spl, (l_pot_max + 1) * (l_pot_max + 1) * n_coeff_hartree * n_hartree_grid * n_atoms,
    //      centers_delta_v_hart_part_spl, hipMemcpyHostToDevice);  // CL_MEM_USE_HOST_PTR
}

void sum_up_begin_()
{
    struct timeval start, end, startall, endall;
    gettimeofday(&start, NULL);
    // gettimeofday(&startall, NULL);
    sum_up_begin_0_();
    sum_up_begin_0_finished = 0;

    long time_uses[32];
    long time_all = 0;
    char *time_infos[32];
    size_t time_index = 0;

    hipError_t error;
    unsigned int arg_index_centers_rho_multipole_spl = 0;
    unsigned int arg_index_i_center_begin = 0;

    // int i_center_tile_size = n_centers_hartree_potential;
    int i_center_tile_size = MIN(i_center_tile_size_default, n_centers_hartree_potential);

    size_t localSize[] = {256};
    size_t globalSize[] = {256 * 128};

    // printf("index_cc size %d\n", n_cc_lm_ijk(l_max_analytic_multipole));
    int *index_cc_aos = (int *)malloc(sizeof(int) * n_cc_lm_ijk(l_max_analytic_multipole) * 6);
    for (int i = 1; i <= n_cc_lm_ijk(l_max_analytic_multipole); i++)
    {
        index_cc_aos[(i - 1) * 4] = index_cc(i, 3, n_cc_lm_ijk(l_max_analytic_multipole));
        index_cc_aos[(i - 1) * 4 + 1] = index_cc(i, 4, n_cc_lm_ijk(l_max_analytic_multipole));
        index_cc_aos[(i - 1) * 4 + 2] = index_cc(i, 5, n_cc_lm_ijk(l_max_analytic_multipole));
        index_cc_aos[(i - 1) * 4 + 3] = index_cc(i, 6, n_cc_lm_ijk(l_max_analytic_multipole));
    }
    _FW_(int, index_cc_aos, n_cc_lm_ijk(l_max_analytic_multipole) * 6, index_cc_aos, hipMemcpyHostToDevice);

    // TODO 这一块应该也可以塞到 sum_up_begin_0_ 中，256*128*11*11x8 也才几十兆，
    // sumup tmp
    _FW_(double, NULL, globalSize[0] * (l_pot_max + 2), Fp, 0);
    _FW_(double, NULL, globalSize[0] * 3 * (l_pot_max + 1), coord_c, 0);
    _FW_(double, NULL, globalSize[0] * 1, coord_mat, 0);
    _FW_(double, NULL, globalSize[0] * 1, rest_mat, 0);
    _FW_(double, NULL, globalSize[0] * 1, vector, 0); // 留下这个，不过这个理应已经废�?
    _FW_(double, NULL, globalSize[0] * 1, delta_v_hartree_multipole_component,
         0);
    _FW_(double, NULL, globalSize[0] * 1, rho_multipole_component, 0);
    _FW_(double, NULL, globalSize[0] * 1, ylm_tab, 0);
    // sumup tmp
    sum_up::arg_index = 63;
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Fp);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coord_c);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coord_mat);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rest_mat);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.vector);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.delta_v_hartree_multipole_component);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rho_multipole_component);
    setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ylm_tab);
    arg_index_i_center_begin = sum_up::arg_index;

    // TODO 这一块也可以放前面去
    int max_spl_atom = 1;
    for (int i_center = 1; i_center <= n_centers_hartree_potential; i_center++)
    {
        int current_center = MV(pbc_lists, centers_hartree_potential)[i_center - 1];
        int current_spl_atom = MV(pbc_lists, center_to_atom)[current_center - 1];
        max_spl_atom = max_spl_atom > current_spl_atom ? max_spl_atom : current_spl_atom;
    }

    int *spl_atom_to_i_center = (int *)malloc(sizeof(int) * (max_spl_atom + 1));
    int *i_center_to_centers_index = (int *)malloc(sizeof(int) * n_centers_hartree_potential);

    for (int i_center_tile = 0; i_center_tile < n_centers_hartree_potential; i_center_tile += i_center_tile_size)
    {

        for (int i = 0; i < (max_spl_atom + 1); i++)
            spl_atom_to_i_center[i] = -1;

        for (int i_center_ = i_center_tile; i_center_ < MIN(i_center_tile + i_center_tile_size, n_centers_hartree_potential); i_center_++)
        {
            int i_center = i_center_ + 1;
            int current_center = MV(pbc_lists, centers_hartree_potential)[i_center_];
            int current_spl_atom = MV(pbc_lists, center_to_atom)[current_center - 1];

            if (spl_atom_to_i_center[current_spl_atom] != -1)
            {
                i_center_to_centers_index[i_center_] = spl_atom_to_i_center[current_spl_atom];
            }
            else
            {
                i_center_to_centers_index[i_center_] = i_center_ - i_center_tile;
                spl_atom_to_i_center[current_spl_atom] = i_center_ - i_center_tile;
            }
        }
    }

    _FW_(int, i_center_to_centers_index, n_centers_hartree_potential, i_center_to_centers_index, hipMemcpyHostToDevice);
    setKernelArgs(sum_up::args, arg_index_i_center_begin + 2, sizeof(hipDeviceptr_t), &hip_buf_com.i_center_to_centers_index);

    error = hipMemcpy(hip_buf_com.rho_multipole_index, MV(hartree_potential_storage, rho_multipole_index),
                      sizeof(int) * n_atoms, hipMemcpyHostToDevice);
    IF_ERROR_EXIT(error != hipSuccess, error, "hipmemcpy hip_buf_com.rho_multipole_index failed");

    setKernelArgs(sum_up::args, remember_arg_Fp_function_spline, sizeof(hipDeviceptr_t), &hip_buf_com.Fp_function_spline_slice);      // TODO 赋值位置可能得�?
    setKernelArgs(sum_up::args, remember_arg_Fp_function_spline + 1, sizeof(hipDeviceptr_t), &hip_buf_com.Fpc_function_spline_slice); // TODO 赋值位置可能得�?

    _FW_(double, NULL, (l_pot_max + 1) * (l_pot_max + 1) * n_max_spline * (n_max_radial + 2) * i_center_tile_size, centers_rho_multipole_spl, 0);
    _FW_(double, NULL, (l_pot_max + 1) * (l_pot_max + 1) * n_coeff_hartree * n_hartree_grid * i_center_tile_size, centers_delta_v_hart_part_spl, 0);
    // _FW_(double, sum_up_param.centers_rho_multipole_spl, (l_pot_max + 1) * (l_pot_max + 1) * n_max_spline * (n_max_radial + 2) * n_atoms,
    //      centers_rho_multipole_spl, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    // _FW_(double, sum_up_param.centers_delta_v_hart_part_spl, (l_pot_max + 1) * (l_pot_max + 1) * n_coeff_hartree * n_hartree_grid * n_atoms,
    //      centers_delta_v_hart_part_spl, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);  // CL_MEM_USE_HOST_PTR

    // _FW_(double, MV(hartree_potential_storage, rho_multipole_shmem_ptr),
    //      ((l_pot_max + 1) * (l_pot_max + 1) * (n_max_radial + 2) * n_rho_multipole_atoms), rho_multipole_h_p_s, 1);

    if (MV(hartree_potential_storage, use_rho_multipole_shmem))
    {
        _FW_(double, MV(hartree_potential_storage, rho_multipole_shmem_ptr),
             ((l_pot_max + 1) * (l_pot_max + 1) * (n_max_radial + 2) * n_rho_multipole_atoms), rho_multipole_h_p_s, 1);
    }
    else
    {
        _FW_(double, MV(hartree_potential_storage, rho_multipole),
             ((l_pot_max + 1) * (l_pot_max + 1) * (n_max_radial + 2) * n_rho_multipole_atoms), rho_multipole_h_p_s, 1);
    }
    sum_up_pre_processing_init_();

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------

    // init_opencl_util_mpi_();
    // int vid = 0;
    // int vnum = 8;
    int vnum = MV(opencl_util, mpi_task_per_gpu);
    // std::cout << "vnum=" << vnum << std::endl;
#undef n_my_batches_work_sumup
// #undef n_max_batch_size
#undef n_full_points_work_sumup
#undef Fp_max_grid
#undef lmax_Fp
#undef Fp_grid_min
#undef Fp_grid_inc
#undef Fp_grid_max
#undef batches_size_sumup

    int i_full_points[8] = {0};
    int i_valid_points[8] = {0};

    for (int vid = 0; vid < vnum; vid++)
    {

        // printf("-------------vid=%d--------------\n", vid);

        _FWV_(int, ocl_util_vars_all[vid].batches_size_sumup, ocl_util_vars_all[vid].n_my_batches_work_sumup, cl_buf_sumup[vid].batches_size_sumup, hipMemcpyHostToDevice);                                             // FLAG 不同
        _FWV_(double, ocl_util_vars_all[vid].batches_points_coords_sumup, 3 * n_max_batch_size * ocl_util_vars_all[vid].n_my_batches_work_sumup, cl_buf_sumup[vid].batches_points_coords_sumup, hipMemcpyHostToDevice); // FLAG 不同

        // sumup
        _FWV_(double, ocl_util_vars_all[vid].partition_tab, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].partition_tab_std, hipMemcpyHostToDevice);
        _FWV_(double, ocl_util_vars_all[vid].delta_v_hartree, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].delta_v_hartree, hipMemcpyHostToDevice); // 输出
        _FWV_(double, ocl_util_vars_all[vid].rho_multipole, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].rho_multipole, hipMemcpyHostToDevice);     // 输出

        _FWV_(double, ocl_util_vars_all[vid].adap_outer_radius_sq, n_atoms, cl_buf_sumup[vid].adap_outer_radius_sq, hipMemcpyHostToDevice);
        _FWV_(double, ocl_util_vars_all[vid].multipole_radius_sq, n_atoms, cl_buf_sumup[vid].multipole_radius_sq, hipMemcpyHostToDevice);
        _FWV_(int, ocl_util_vars_all[vid].l_hartree_max_far_distance, n_atoms, cl_buf_sumup[vid].l_hartree_max_far_distance, hipMemcpyHostToDevice);
        _FWV_(double, ocl_util_vars_all[vid].outer_potential_radius, (l_pot_max + 1) * n_atoms, cl_buf_sumup[vid].outer_potential_radius, hipMemcpyHostToDevice);
        _FWV_(double, ocl_util_vars_all[vid].multipole_c, n_cc_lm_ijk(l_pot_max) * n_atoms, cl_buf_sumup[vid].multipole_c, hipMemcpyHostToDevice);

        // TODO 这一块也想办法提前面�?
        int *point_to_i_batch = (int *)malloc(sizeof(int) * ocl_util_vars_all[vid].n_full_points_work_sumup);
        int *point_to_i_index = (int *)malloc(sizeof(int) * ocl_util_vars_all[vid].n_full_points_work_sumup);
        int *valid_point_to_i_full_point = (int *)malloc(sizeof(int) * ocl_util_vars_all[vid].n_full_points_work_sumup);

        for (int i_batch = 1; i_batch <= ocl_util_vars_all[vid].n_my_batches_work_sumup; i_batch++)
        {
            for (int i_index = 1; i_index <= ocl_util_vars_all[vid].batches_size_sumup[i_batch - 1]; i_index++)
            {
                point_to_i_batch[i_full_points[vid]] = i_batch;
                point_to_i_index[i_full_points[vid]] = i_index;
                i_full_points[vid]++;
                if (ocl_util_vars_all[vid].partition_tab[i_full_points[vid] - 1] > 0.0)
                {
                    valid_point_to_i_full_point[i_valid_points[vid]] = i_full_points[vid] - 1;
                    i_valid_points[vid]++;
                }
            }
        }
        if (i_full_points[vid] > ocl_util_vars_all[vid].n_full_points_work_sumup)
        {
            printf("rank%d, i_full_points=%d, n_full_points_work_sumup=%d\n", myid + vid, i_full_points[vid], ocl_util_vars_all[vid].n_full_points_work_sumup);
            fflush(stdout);
            // char save_file_name[64];
            // sprintf(save_file_name, "mdata_outer_rank%d_%d.bin", myid + vid, 101010);
            // m_save_load(save_file_name, 0, 1);
            // m_save_load_sumup(save_file_name, 0, 1);
            // fflush(stdout);
            exit(-1);
        }

        // TODO 验证一下大小，看看这一块也能不能放前面去，4 * n_full_points_work_sumup / (1024 * 1024) < 20 就行
        // loop helper

        _FWV_(int, point_to_i_batch, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].point_to_i_batch, hipMemcpyHostToDevice);
        _FWV_(int, point_to_i_index, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].point_to_i_index, hipMemcpyHostToDevice);
        _FWV_(int, valid_point_to_i_full_point, ocl_util_vars_all[vid].n_full_points_work_sumup, cl_buf_sumup[vid].valid_point_to_i_full_point, hipMemcpyHostToDevice);

        free(point_to_i_batch);
        free(point_to_i_index);
        free(valid_point_to_i_full_point);
    }

    gettimeofday(&end, NULL);
    time_uses[time_index] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    time_infos[time_index++] = "sumup writebuf and setarg";
    // if (myid < 8)
    // {
    //     printf("rank%d, %s: %lf seconds\n", myid, time_infos[time_index - 1], time_uses[time_index - 1] / 1000000.0);
    //     fflush(stdout);
    // }
    gettimeofday(&start, NULL);

    // hipEvent_t event;
    // long time_preproc = 0;
    // long time_main_kernel = 0;

    double time_preproc = 0;
    double time_main_kernel = 0;
    float SUMUP_time;
    for (int i_center_tile = 0; i_center_tile < n_centers_hartree_potential; i_center_tile += i_center_tile_size)
    {
        // TODO 改用 kernel 版的 preprocessing，注意设备端数组指针的使�?

        int i_center_begin = i_center_tile;
        int i_center_end = MIN(i_center_tile + i_center_tile_size, n_centers_hartree_potential);
        // if (myid == 0)
        //     std::cout << " n_centers_hartree_potential: " << n_centers_hartree_potential << " i_center_tile_size: " << i_center_tile_size << " i_center_begin: " << i_center_begin << " i_center_end: " << i_center_end << std::endl;
        struct timeval start, end;
        // clock_t sumuppre_start, sumuppre_end;
        gettimeofday(&start, NULL);
        // sumuppre_start = clock();

        // OpenCL version
        sum_up_pre_processing_part_(&n_coeff_hartree, &i_center_begin, &i_center_end,
                                    &hip_buf_com.centers_rho_multipole_spl, &hip_buf_com.centers_delta_v_hart_part_spl,
                                    &hip_buf_com.i_center_to_centers_index, 1);

        // sumuppre_end = clock();
        // if (myid == 0)
        //     std::cout << "after n_centers_hartree_potential: " << n_centers_hartree_potential << " i_center_tile_size: " << i_center_tile_size << " i_center_begin: " << i_center_begin << " i_center_end: " << i_center_end << std::endl;
        gettimeofday(&end, NULL);
        time_preproc += 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
        // time_preproc = (sumuppre_end - sumuppre_start) / (double)CLOCKS_PER_SEC;

        setKernelArgs(sum_up::args, arg_index_i_center_begin, sizeof(int), &i_center_begin);
        setKernelArgs(sum_up::args, arg_index_i_center_begin + 1, sizeof(int), &i_center_end);

        // -------------------------------------

        for (int vid = 0; vid < vnum; vid++)
        {

            setKernelArgs(sum_up::args, remember_arg_n_my_batches_work_sumup, sizeof(int), &ocl_util_vars_all[vid].n_my_batches_work_sumup);
            setKernelArgs(sum_up::args, remember_arg_n_my_batches_work_sumup + 1, sizeof(int), &ocl_util_vars_all[vid].n_full_points_work_sumup);

            setKernelArgs(sum_up::args, remember_arg_Fp_max_grid, sizeof(int), &ocl_util_vars_all[vid].Fp_max_grid);
            setKernelArgs(sum_up::args, remember_arg_Fp_max_grid + 1, sizeof(int), &ocl_util_vars_all[vid].lmax_Fp);
            setKernelArgs(sum_up::args, remember_arg_Fp_max_grid + 2, sizeof(double), &ocl_util_vars_all[vid].Fp_grid_min);
            setKernelArgs(sum_up::args, remember_arg_Fp_max_grid + 3, sizeof(double), &ocl_util_vars_all[vid].Fp_grid_inc);
            setKernelArgs(sum_up::args, remember_arg_Fp_max_grid + 4, sizeof(double), &ocl_util_vars_all[vid].Fp_grid_max);

            setKernelArgs(sum_up::args, remember_arg_index_1, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].batches_size_sumup);
            setKernelArgs(sum_up::args, remember_arg_index_1 + 1, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].batches_points_coords_sumup);

            // sumup specific
            sum_up::arg_index = 0;
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &sum_up_param.forces_on);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].partition_tab_std);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].delta_v_hartree);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].rho_multipole);
            arg_index_centers_rho_multipole_spl = sum_up::arg_index + 4;
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_rho_multipole_spl);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_delta_v_hart_part_spl);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].adap_outer_radius_sq);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].multipole_radius_sq);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].l_hartree_max_far_distance);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].outer_potential_radius);
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].multipole_c);

            sum_up::arg_index = 58;
            // �? partition_tab 预判�? i_valid_points ，没有： i_full_points
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(int), &i_valid_points[vid]); // TODO 注意这个 ！！�?, 有没�? partition_tab 的预判不�? !
            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].point_to_i_batch);

            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].point_to_i_index);

            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &cl_buf_sumup[vid].valid_point_to_i_full_point);

            setKernelArgs(sum_up::args, sum_up::arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_cc_aos);

            // -------------------------------------
            dim3 blockDim(localSize[0], 1, 1);
            dim3 gridDim(globalSize[0] / localSize[0], 1, 1);
            // hipEvent_t sumup_start, sumup_stop;
            // float sumuptime;
            // hipEventCreate(&sumup_start);
            // hipEventCreate(&sumup_stop);
            // hipEventRecord(sumup_start);
            gettimeofday(&start, NULL);
            if (JIT_ENABLED)
            {
                error = hipModuleLaunchKernel(kernels[0], globalSize[0] / localSize[0], 1, 1, localSize[0], 1, 1,
                                              0, nullptr, sum_up::args, nullptr);
            }
            else
            {
                error = hipLaunchKernel(reinterpret_cast<const void *>(&sum_up_whole_potential_shanghui_sub_t_), gridDim, blockDim, sum_up::args, 0, 0);
            }

            IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel sumup failed");
            hipDeviceSynchronize();
            // hipEventRecord(sumup_stop);
            // hipEventSynchronize(sumup_stop);
            // hipEventElapsedTime(&sumuptime, sumup_start, sumup_stop);
            // SUMUP_time += sumuptime;
            gettimeofday(&end, NULL);
            long time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            time_main_kernel += time_use;
        }
    }
    n_centers_hartree_potential;
    std::ofstream outFileSUM;
    // outFileSUM.open("SUMdatas.csv", std::ios::app);
    // double batches_size_sumup = 0;
    // for (int i = 0; i < ocl_util_vars_all[0].n_my_batches_work_sumup; ++i)
    // {
    //     batches_size_sumup += ocl_util_vars_all[0].batches_size_sumup[i];
    // }

    for (int vid = 0; vid < vnum; vid++)
    {
        hipMemcpy(ocl_util_vars_all[vid].delta_v_hartree, cl_buf_sumup[vid].delta_v_hartree, sizeof(double) * ocl_util_vars_all[vid].n_full_points_work_sumup, hipMemcpyDeviceToHost);
        hipMemcpy(ocl_util_vars_all[vid].rho_multipole, cl_buf_sumup[vid].rho_multipole, sizeof(double) * ocl_util_vars_all[vid].n_full_points_work_sumup, hipMemcpyDeviceToHost);
    }
    hipDeviceSynchronize();
    // delta_v_hartree �? rho_multipole �? use_host_ptr 直接同步

    gettimeofday(&end, NULL);
    time_uses[time_index] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    time_infos[time_index++] = "sumup kernel and readbuf";
    // if (myid < 8)
    // {
    //     printf("rank%d, %s: %lf seconds, preproc %lf s, main kernel %lf s\n", myid, time_infos[time_index - 1], time_uses[time_index - 1] / 1000000.0,
    //            time_preproc / 1000000.0, time_main_kernel / 1000000.0);
    //     fflush(stdout);
    // }
    gettimeofday(&start, NULL);

    free(index_cc_aos);

    // free(local_centers_rho_multipole_spl);
    // free(local_centers_delta_v_hart_part_spl);
    free(spl_atom_to_i_center);
    free(i_center_to_centers_index);

    sum_up_pre_processing_finish();

    for (int vid = 0; vid < vnum; vid++)
    {
        hipFree(cl_buf_sumup[vid].batches_size_sumup);
        hipFree(cl_buf_sumup[vid].batches_points_coords_sumup);

        hipFree(cl_buf_sumup[vid].partition_tab_std);
        hipFree(cl_buf_sumup[vid].delta_v_hartree);
        hipFree(cl_buf_sumup[vid].rho_multipole);
        hipFree(cl_buf_sumup[vid].adap_outer_radius_sq);
        hipFree(cl_buf_sumup[vid].multipole_radius_sq);
        hipFree(cl_buf_sumup[vid].l_hartree_max_far_distance);
        hipFree(cl_buf_sumup[vid].outer_potential_radius);
        hipFree(cl_buf_sumup[vid].multipole_c);

        hipFree(cl_buf_sumup[vid].point_to_i_batch);
        hipFree(cl_buf_sumup[vid].point_to_i_index);
        hipFree(cl_buf_sumup[vid].valid_point_to_i_full_point);
    }

    hipFree(hip_buf_com.index_cc_aos);

    hipFree(hip_buf_com.centers_rho_multipole_spl);
    hipFree(hip_buf_com.centers_delta_v_hart_part_spl);

    hipFree(hip_buf_com.Fp);
    hipFree(hip_buf_com.coord_c);
    hipFree(hip_buf_com.coord_mat);
    hipFree(hip_buf_com.rest_mat);
    hipFree(hip_buf_com.vector);
    hipFree(hip_buf_com.delta_v_hartree_multipole_component);
    hipFree(hip_buf_com.rho_multipole_component);
    hipFree(hip_buf_com.ylm_tab);

    if (MV(hartree_potential_storage, use_rho_multipole_shmem))
    {
        hipHostUnregister(MV(hartree_potential_storage, rho_multipole_shmem_ptr));
    }
    else
    {
        hipHostUnregister(MV(hartree_potential_storage, rho_multipole));
    }

    hipFree(hip_buf_com.rho_multipole_h_p_s);
    // char save_file_name[64];
    // sprintf(save_file_name, "sumup_check_rank%d_%d.bin", myid, sumup_c_count);
    // FILE *file_p = fopen(save_file_name, "w");
    // for (int i = 0; i < n_full_points_work_stable; i++) {
    //   fprintf(file_p, "%6d, %.13lf, %.13lf\n", i, sum_up_param.delta_v_hartree[i], sum_up_param.rho_multipole[i]);
    // }
    // fclose(file_p);

    // if (myid == 0)
    //     m_save_check_sumup_(sum_up_param.delta_v_hartree, sum_up_param.rho_multipole);

    // printf("End\n");

    // gettimeofday(&endall, NULL);
    // time_all = 1000000 * (endall.tv_sec - startall.tv_sec) + endall.tv_usec - startall.tv_usec;
    // time_uses[time_index] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // time_infos[time_index++] = "sumup write check file(for debug)";
    // gettimeofday(&startall, NULL);
    // if (outFileSUM.is_open())
    // {
    //     if (myid == 0)
    //     {

    //         outFileSUM << "myid"
    //                    << ","
    //                    << "sumupcnt"
    //                    << ","
    //                    << "SUMUP_kernel_time"
    //                    << ","
    //                    << "kernel_and_others"
    //                    << std::endl;
    //         outFileSUM << myid << "," << sumupcnt << "," << (time_preproc / 1000.0) + (time_main_kernel / 1000.0) << "," << time_all / 1000.0 << std::endl;
    //         outFileSUM.close();
    //     }
    //     else
    //     {
    //         sleep(1);
    //         outFileSUM << myid << "," << sumupcnt << "," << (time_preproc / 1000.0) + (time_main_kernel / 1000.0) << "," << time_all / 1000.0 << std::endl;
    //         outFileSUM.close();
    //     }
    // }
    // else
    // {
    //     std::cout << "Failed to open file." << std::endl;
    // }
    // gettimeofday(&endall, NULL);
    // sumup_write_time = (1000000 * (endall.tv_sec - startall.tv_sec) + endall.tv_usec - startall.tv_usec) / 1000000.0;
    // if (myid < 8)
    // {
    //     printf("rank%d, %s: %lf seconds\n", myid, time_infos[time_index - 1], time_uses[time_index - 1] / 1000000.0);
    //     fflush(stdout);
    // }
    // for(size_t i=0; i<time_index; i++){
    //   printf("rank%d, %s: %lf seconds\n", myid, time_infos[i], time_uses[i]/1000000.0);
    // }
    sumupcnt++;
}

void sum_up_final_end()
{
    hipFree(hip_buf_com.Fp_function_spline_slice);
    hipFree(hip_buf_com.Fpc_function_spline_slice);
}

SUM_UP_PARAM sum_up_param;

void set_sum_up_param(int forces_on, double *partition_tab_std, double *delta_v_hartree, double *rho_multipole,
                      // double *centers_rho_multipole_spl, double *centers_delta_v_hart_part_spl,
                      double *adap_outer_radius_sq, double *multipole_radius_sq, int *l_hartree_max_far_distance,
                      double *outer_potential_radius, double *multipole_c)
{
    sum_up_param.forces_on = forces_on;
    sum_up_param.partition_tab = partition_tab_std;
    sum_up_param.delta_v_hartree = delta_v_hartree;
    sum_up_param.rho_multipole = rho_multipole;
    // sum_up_param.centers_rho_multipole_spl = centers_rho_multipole_spl;
    // sum_up_param.centers_delta_v_hart_part_spl = centers_delta_v_hart_part_spl;
    sum_up_param.adap_outer_radius_sq = adap_outer_radius_sq;
    sum_up_param.multipole_radius_sq = multipole_radius_sq;
    sum_up_param.l_hartree_max_far_distance = l_hartree_max_far_distance;
    sum_up_param.outer_potential_radius = outer_potential_radius;
    sum_up_param.multipole_c = multipole_c;
}

void init_sum_up_c_(int *forces_on, double *partition_tab_std, double *delta_v_hartree, double *rho_multipole,
                    double *adap_outer_radius_sq, double *multipole_radius_sq, int *l_hartree_max_far_distance,
                    double *outer_potential_radius, double *multipole_c)
{
    // sumup_c_count++;
    // char save_file_name[64];
    set_sum_up_param(*forces_on, partition_tab_std, delta_v_hartree, rho_multipole,
                     adap_outer_radius_sq, multipole_radius_sq, l_hartree_max_far_distance,
                     outer_potential_radius, multipole_c);
    // sprintf(save_file_name, "mdata_outer_rank%d_%d.bin", myid, sumup_c_count);
    // if (myid == 0 && sumup_c_count <= 1)
    // if (myid == 0)
    //   m_save_load_sumup(save_file_name, 0, 1);
    // if(sumup_c_count == 1 || myid == 0)
    //   m_save_load_sumup(save_file_name, 0, 0);

    // m_save_load_sumup_();

    if (init_sum_up)
    {
        return;
    }
    init_sum_up = 1;
    // if (Fp_function_spline_slice == NULL)
    //   Fp_function_spline_slice = (double *)malloc(sizeof(double) * (Fp_max_grid+1) * 4 * (lmax_Fp + 1));
    // if (Fpc_function_spline_slice == NULL)
    //   Fpc_function_spline_slice = (double *)malloc(sizeof(double) * (Fp_max_grid+1) * 4 * (lmax_Fp + 1));
    // for (int i = 0; i < Fp_max_grid; i++) {
    //   memcpy(&Fp_function_spline_slice[i * 4 * (lmax_Fp + 1)], &Fp_function_spline[i * n_max_spline * (lmax_Fp + 1)],
    //          sizeof(double) * 4 * (lmax_Fp + 1));
    //   memcpy(&Fpc_function_spline_slice[i * 4 * (lmax_Fp + 1)], &Fpc_function_spline[i * n_max_spline * (lmax_Fp + 1)],
    //          sizeof(double) * 4 * (lmax_Fp + 1));
    // }
    if (Fp == NULL)
        Fp = (double *)malloc(sizeof(double) * (l_pot_max + 2) * n_centers_hartree_potential);
    // FILE *file_p = fopen("tmp.bin", "w");
    // for (int i = 0; i < Fp_max_grid; i++) {
    //   for (int j = 0; j < n_max_spline; j++) {
    //     for (int k = 0; k < (lmax_Fp + 1); k++) {
    //       fprintf(file_p, "%6d,%6d,%6d, %.13lf, %.13lf\n", i, j, k,
    //               Fp_function_spline[((i * n_max_spline) + j) * (lmax_Fp + 1) + k],
    //               Fpc_function_spline[((i * n_max_spline) + j) * (lmax_Fp + 1) + k]);
    //     }
    //   }
    // }
    // fclose(file_p);

    // opencl_init_();
    // opencl_common_buffer_init_();
}

__global__ void warmupKernel()
{
    // Empty kernel
}

void pre_run_()
{
    warmupKernel<<<1, 1>>>();
    hipDeviceSynchronize(); // Synchronize after warmup
}

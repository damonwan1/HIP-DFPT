# HIP-DFPT: High-Throughput All-Electron Quantum Perturbation Simulations on GPU Clusters

This repository provides the performance optimization modules, scheduling mechanisms, and evaluation scripts developed for HIP-DFPT, a high-throughput all-electron DFPT simulation framework optimized for large-scale GPU clusters.

**Notice:**  
Due to licensing restrictions of the FHI-aims software, the core DFPT solver code is not open-sourced. This repository only releases the GPU optimization techniques, performance modeling modules, dynamic schedulers, and scaling evaluation scripts developed independently as part of this project.


## Environment and Dependencies

- HIP-enabled GPU platform (e.g., AMD MI250, Hygon GPUs)
- ROCm 5.x or HIP-CUDA backend
- CMake ≥ 3.14, GCC ≥ 9
- Python 3.9, PyTorch 2.0.2, scikit-learn, numpy, matplotlib
- MPI library (OpenMPI / MPICH)

## Quick Start

1. Load the environment modules required for HIP, MPI, and compilers.
2. Configure and compile the optimization modules using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j
   ```
3. Generate sample datasets using provided scripts.
4. Submit SLURM jobs via:
   ```bash
   sbatch scripts/job_scripts/run_scaling_test.sh
   ```

## Artifacts Overview

The released code and scripts correspond to the following experimental artifacts:

- **A1: HIP-DFPT GPU Optimization Modules**  
  Implements batch merging, memory placement, and pipeline techniques validated in performance ablation studies.

- **A2: Performance Modeling Module**  
  Scripts to train runtime prediction models with 5% task sampling, achieving high accuracy at low cost.

- **A3: Dynamic Scheduling Module**  
  Load balancing scheduler using predicted runtimes to optimize inter- and intra-GPU workload distribution.

- **A4: Scaling Evaluation Scripts**  
  Batch scripts and analysis tools to reproduce strong and weak scaling experiments up to 8192 GPUs.

- **A5: Branch-Level Performance Prediction**  
  MLP-based runtime predictors for branch-intensive modules (RHO and H), built from aggregated wavefront behavior.

## License

This repository is released under a custom academic license.  
The DFPT solver and FHI-aims source code are **not included** and **not open-sourced**.  
Only performance optimization techniques, models, and schedulers developed independently are provided.


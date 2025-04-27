#!/bin/bash
#SBATCH -J 11whipP
#SBATCH --partition=mar13
#SBATCH --nodes=2048
#SBATCH --gres=dcu:4
#SBATCH --tasks-per-node=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -o 2048nodes_mod4.out
#SBATCH -e 2048nodes_mod.err

module unload mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/intel/2017.5.239
module load mpi/intelmpi/2017.4.239
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/21.04

export OMP_NUM_THREADS=1

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo $SLURM_NODELIST
echo $SLURM_CLUSTER_NAME
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

filename="/public/home/aicao/jiazifan/submit-bak/fhi-aims-dcu-mod/bin/aims.191127.scalapack.mpi.x"

echo "filename=$filename"

ldd ${filename}

#mpirun ${filename}
mpirun -np 8192 ${filename}
# ./aims.191127.scalapack.mpi.x
# srun ../aims.191127.scalapack.mpi.x

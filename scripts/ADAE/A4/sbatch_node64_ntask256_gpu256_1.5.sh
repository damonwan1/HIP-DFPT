#!/bin/bash
#SBATCH -J C2H4_L
#SBATCH --partition=july10
#SBATCH --nodes=64
#SBATCH --gres=dcu:4
#SBATCH --tasks-per-node=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -o 64nodes_mod4.out
#SBATCH -e 64nodes_mod.err


export OMP_NUM_THREADS=1

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo $SLURM_NODELIST
echo $SLURM_CLUSTER_NAME
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

filename="/work1/aicao/duhao/fhi-aims-dcu-mod-mergeload/bin/aims.191127.scalapack.mpi.x"

echo "filename=$filename"

ldd ${filename}

#mpirun ${filename}
mpirun -np 256 ${filename}
# ./aims.191127.scalapack.mpi.x
# srun ../aims.191127.scalapack.mpi.x

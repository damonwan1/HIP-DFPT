#!/bin/bash
#SBATCH -J C2H4_L
#SBATCH --partition=july10
#SBATCH --nodes=128
#SBATCH --gres=dcu:4
#SBATCH --tasks-per-node=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -o 128nodes_mod4.out
#SBATCH -e 128nodes_mod.err

#//SBATCH --exclude=j13r1n19,j13r2n[00,13-15,18],j13r3n[01,03],j15r3n06,g04r1n18,f03r4n11
#//SBATCH --mem-per-cpu=14G
#//SBATCH --ntasks-per-node=32
#//SBATCH --exclude=g16r4n07
#//SBATCH --gres=dcu:1
#//SBATCH --ntasks=16

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
mpirun -np 512 ${filename}
# ./aims.191127.scalapack.mpi.x
# srun ../aims.191127.scalapack.mpi.x

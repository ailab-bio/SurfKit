#!/bin/bash

id_start=1031
id_end=1623
cpus=20
nmol=150 ; 
watermax=15000

cp Blank/run-GPU.slurm .
sed -i "s/XXXNMOLXXX/${nmol}/g" run-GPU.slurm
sed -i "s/XXXCPUSXXX/${cpus}/g" run-GPU.slurm
sed -i "s/XXXWATERMAXXXX/${watermax}/g" run-GPU.slurm
sbatch --array=${id_start}-${id_end} run-GPU.slurm

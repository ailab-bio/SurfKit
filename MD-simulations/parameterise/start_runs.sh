#!/bin/bash

for i in `seq 0 10` ; do
   molecule=surfpro_${i}
   cp Blank/parameterise.slurm .
   sed -i "s/XXXMOLECULEXXX/${molecule}/g" parameterise.slurm
   sbatch parameterise.slurm
   rm parameterise.slurm
done

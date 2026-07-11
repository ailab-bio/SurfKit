#!/bin/bash

id_start=1
id_end=10
nmol=150  
watermax=15000


runfile=run-GPU.slurm
for id in `seq ${id_start} ${id_end}`; do
	molecule=surfpro_${id}
	cp Blank/${runfile} .
	sed -i "s/XXXNMOLXXX/${nmol}/g" ${runfile}
	sed -i "s/XXXJOBNAMEXXX/${id}/g" ${runfile}
	sed -i "s/XXXMOLECULEXXX/${molecule}/g" ${runfile}
	sed -i "s/XXXWATERMAXXXX/${watermax}/g" ${runfile}
	sbatch --array=0-10 ${runfile}
done

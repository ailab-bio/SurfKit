#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=72:00:00
#SBATCH -p YOU_PARTITION 

####################
# DIRECTORIES      #
###################

PROJECT_ROOT=/INSERT/YOUR/PATH
SEED_DIR=/INSERT/YOUR/PATH
DATA_DIR=0
if [[ ${DATA_DIR} -eq 0 ]]; then
	echo "Please open run-GPU.slurm and set environment variables"
	echo "STOPPING NOW"
	exit
fi

############
#PARAMETERS#
############

molecule=XXXMOLECULEXXX
nmol=XXXNMOLXXX
temperature=298.15
shear=0.02
replica=${SLURM_ARRAY_TASK_ID}

#########################################
#		CODE			#
#########################################
APPTAINER=${PROJECT_ROOT}/MD-simulations/gromacs.sif
echo "starting run for $molecule with $nmol molecules"

cwd=$(pwd)
dir="${DATA_DIR}/viscosity/${molecule}_${nmol}"

#rm -r $dir
mkdir ${dir}
cd ${dir}

srcdir=${DATA_DIR}/equilibrations/${molecule}_${nmol}/long-NPT/


# begin production-ST run
mkdir ${dir}/ST_$(printf "%02g" $replica )
cd ${dir}/ST_$(printf "%02g" $replica )
scp ${srcdir}/npt_$(printf "%02g" $replica ).gro ${srcdir}/topol.top ${cwd}/Blank/production-shear.mdp  .

mv production-shear.mdp production.mdp
mv npt_$(printf "%02g" $replica ).gro begin.gro
sed -i "s/XXXSHEARXXX/${shear}/g" production.mdp


if [[ $nmol -ne 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}  /g" production.mdp
elif [[ $nmol -eq 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  /g" production.mdp
	sed -i "s/tc-grps.*/tc-grps = SOL  /g" production.mdp
	sed -i "s/tau_t.*/tau_t = 0.1  /g" production.mdp
fi
apptainer exec ${APPTAINER} gmx grompp -f production.mdp -c begin.gro -p topol.top -o production.tpr

apptainer exec ${APPTAINER} gmx mdrun -deffnm production

apptainer exec ${APPTAINER} gmx mdrun -deffnm production
echo '''34
35
36
37
38
39
40


''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f production.edr -o potential.xvg < input.temp

echo '''33
34
45
36
37
38
39
40
41
'''> input.temp
apptainer exec ${APPTAINER} gmx energy -f production.edr -o visco.xvg < input.temp


rm '#production#'

mv ${cwd}/slurm-${SLURM_JOB_ID}.out ${dir}/.


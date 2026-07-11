#!/bin/bash
#SBATCH --account=NAISS2025-5-462
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=2:00:00
#SBATCH -p alvis

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

molecule=surfpro_${SLURM_ARRAY_TASK_ID}
nmol=XXXNMOLXXX
nwater=$(echo "XXXWATERMAXXXX-${nmol}" | bc -l )
temperature=298.15
shear=0.02

########################################
#DO NOT TOUCH STUFF BELOW THIS LINE    #
#IF YOU DO NOT KNOW WHAT YOU ARE DOING #
########################################
APPTAINER=${PROJECT_ROOT}/MD-simulations/gromacs.sif
echo "starting run for $molecule with $nmol molecules"

cwd=$(pwd)
dir="${DATA_DIR}/bulk-properties/${molecule}_${nmol}"
srcdir="${DATA_DIR}/equilibrations/${molecule}_${nmol}"

if [[ $nwater -le 500 ]]; then
   boxsize=5
elif [[ $nwater -le 1000 ]]; then
   boxsize=7
elif [[ $nwater -le 5000 ]]; then
   boxsize=9
elif [[ $nwater -le 10000 ]]; then
   boxsize=12
elif [[ $nwater -le 20000 ]]; then
   boxsize=16
fi


rm -r $dir
mkdir ${dir}
cd ${dir}




# second equilibration NPT
mkdir ${dir}/long-NPT
cd ${dir}/long-NPT
cp ${srcdir}/equilibrate-NPT/npt.gro ${srcdir}/equilibrate-NPT/topol.top ${cwd}/Blank/long-NPT.mdp .
pwd

if [[ $nmol -ne 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}  /g" long-NPT.mdp
elif [[ $nmol -eq 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  /g" long-NPT.mdp
	sed -i "s/tc-grps.*/tc-grps = SOL  /g" long-NPT.mdp
	sed -i "s/tau_t.*/tau_t = 0.1  /g" long-NPT.mdp
fi

mv npt.gro npt_00.gro
for i in `seq 0 0`; do
	apptainer exec ${APPTAINER} gmx grompp -f long-NPT.mdp -c npt_$(printf "%02g" $i ).gro -p topol.top 
	echo "LOOKING FOR FILE"
	ls topol.tpr 
	apptainer exec ${APPTAINER} gmx mdrun -deffnm npt_$(printf "%02g" $(($i +1)) ) -s topol.tpr
done

apptainer exec ${APPTAINER} gmx trjcat -f npt_*trr -o npt.xtc

echo '''1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42

''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f npt_01.edr -o potential.xvg < input.temp




apptainer exec ${APPTAINER} gmx msd -f npt.xtc -s topol.tpr -o diffusion.xvg <<EOF
0
2
3
4
5
EOF

mv ${cwd}/slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${dir}/.


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

molecule=surfpro_${SLURM_ARRAY_TASK_ID}
nmol=150
watermax=15000

nwater=$(echo "XXXWATERMAXXXX-${nmol}" | bc -l )
temperature=298.15

echo "starting run for $molecule with $nmol molecules"
APPTAINER=${PROJECT_ROOT}/MD-simulations/gromacs.sif

cwd=$(pwd)
dir="${DATA_DIR}/equilibrations/${molecule}_${nmol}"

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


mkdir setup
cp -r ${DATA_DIR}/parameterise/${molecule}/* setup/.
cd setup
cp ${cwd}/Blank/ions.mdp .
if [[ $nmol -ne 0 ]]; then
	apptainer exec ${APPTAINER} gmx insert-molecules -ci molecule.gro -nmol ${nmol} -o solute.gro -box ${boxsize} ${boxsize} ${boxsize} 
	apptainer exec ${APPTAINER} gmx solvate -cp solute.gro -maxsol ${nwater} -o solvated.pdb -cs tip4p.gro > solvation.log # create a box filled with water
elif [[ $nmol -eq 0 ]]; then
	apptainer exec ${APPTAINER} gmx solvate  -maxsol ${nwater} -o solvated.pdb -cs tip4p.gro -box ${boxsize} ${boxsize} ${boxsize} > solvation.log # create a box filled with water
fi

#build the topology file
echo '''#include "/cephyr/users/ricbec/Alvis/software/gromacs-GPU/share/top/amber03.ff/forcefield.itp"
'''> topol.top
sed -n '/\[ atomtypes \]/,/\[ moleculetype \]/{/\[ moleculetype \]/!p;}' MOL.amb2apptainer exec ${APPTAINER} gmx/MOL_GMX.top >> topol.top

echo '''
#include "/cephyr/users/ricbec/Alvis/software/gromacs-GPU/share/top/amber03.ff/tip4p.itp" 
''' >> topol.top

sed -ne '/\[ moleculetype \]/,$ p' MOL.amb2apptainer exec ${APPTAINER} gmx/MOL_GMX.top >> topol.top

head -n -1 topol.top > topol.temp
mv topol.temp topol.top
nNA=$(( $nmol * $total_charge ))
if [[ $nmol -ne 0 ]]; then
	echo "MOL   ${nmol}" >> topol.top
fi
echo "SOL   $nwater" >> topol.top
echo $nwater


apptainer exec ${APPTAINER} gmx grompp -f ions.mdp -c solvated.pdb -p topol.top -o ions.tpr
apptainer exec ${APPTAINER} gmx genion -s ions.tpr -o ions.pdb -p topol.top -neutral<<EOF
4
EOF


mkdir ${dir}/EM
cd ${dir}/EM
cp ${dir}/setup/ions.pdb ${dir}/setup/topol.top ${cwd}/Blank/minim.mdp  . 

# create topology file for EM
if [[ $nmol -ne 0 ]]; then
	apptainer exec ${APPTAINER} gmx grompp -f minim.mdp -c ions.pdb -p topol.top -o em.tpr -maxwarn 1 | tee min.out
	apptainer exec ${APPTAINER} gmx mdrun -v -deffnm em -s em.tpr | tee min.out
elif [[ $nmol -eq 0 ]]; then
	cp ${dir}/setup/solvated.pdb .
	apptainer exec ${APPTAINER} gmx grompp -f minim.mdp -c solvated.pdb -p topol.top -o em.tpr -maxwarn 1 | tee min.out
	apptainer exec ${APPTAINER} gmx mdrun -v -deffnm em -s em.tpr | tee min.out
fi	
echo '''1
4

''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f em.edr -o potential.xvg < input.temp


#equilibrate NVT 
mkdir ${dir}/equilibrate-NVT
cd ${dir}/equilibrate-NVT
cp ${dir}/EM/em.gro ${dir}/EM/topol.top ${cwd}/Blank/equi-NVT.mdp  .
pwd

if [[ $nmol -ne 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}  /g" equi-NVT.mdp
elif [[ $nmol -eq 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  /g" equi-NVT.mdp
	sed -i "s/tc-grps.*/tc-grps = SOL  /g" equi-NVT.mdp
	sed -i "s/tau_t.*/tau_t = 0.1  /g" equi-NVT.mdp
fi
apptainer exec ${APPTAINER} gmx grompp -f equi-NVT.mdp -c em.gro -p topol.top -o nvt.tpr

apptainer exec ${APPTAINER} gmx mdrun -deffnm nvt
echo '''7
8
11
21
22
23
24
25
26
27
28
32

''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f nvt.edr -o potential.xvg < input.temp



#first equilibration NPT
mkdir ${dir}/equilibrate-NPT
cd ${dir}/equilibrate-NPT
cp ${dir}/equilibrate-NVT/nvt.gro ${dir}/equilibrate-NVT/topol.top ${cwd}/Blank/equi-NPT.mdp  .
pwd

if [[ $nmol -ne 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}  /g" equi-NPT.mdp
elif [[ $nmol -eq 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  /g" equi-NPT.mdp
	sed -i "s/tc-grps.*/tc-grps = SOL  /g" equi-NPT.mdp
	sed -i "s/tau_t.*/tau_t = 0.1  /g" equi-NPT.mdp
fi
apptainer exec ${APPTAINER} gmx grompp -f equi-NPT.mdp -c nvt.gro -p topol.top -o topol.tpr

apptainer exec ${APPTAINER} gmx mdrun -deffnm npt -s topol.tpr
echo '''7
8
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
34
38
42

''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f npt.edr -o potential.xvg < input.temp


# second equilibration NPT
mkdir ${dir}/long-NPT
cd ${dir}/long-NPT
cp ${dir}/equilibrate-NPT/npt.gro ${dir}/equilibrate-NPT/topol.top ${cwd}/Blank/long-NPT.mdp .
pwd

if [[ $nmol -ne 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}  /g" long-NPT.mdp
elif [[ $nmol -eq 0 ]]; then
	sed -i "s/ref_t.*/ref_t = ${temperature}  /g" long-NPT.mdp
	sed -i "s/tc-grps.*/tc-grps = SOL  /g" long-NPT.mdp
	sed -i "s/tau_t.*/tau_t = 0.1  /g" long-NPT.mdp
fi

mv npt.gro npt_00.gro
for i in `seq 0 20`; do
	apptainer exec ${APPTAINER} gmx grompp -f long-NPT.mdp -c npt_$(printf "%02g" $i ).gro -p topol.top 
	echo "LOOKING FOR FILE"
	ls topol.tpr 
	apptainer exec ${APPTAINER} gmx mdrun -deffnm npt_$(printf "%02g" $(($i +1)) ) -s topol.tpr
done

apptainer exec ${APPTAINER} gmx trjcat -f npt_*trr -o npt.xtc
apptainer exec ${APPTAINER} gmx trjconv -f npt.xtc -o npt_all.xtc -dt 1
echo '''10
11
14
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


''' > input.temp
apptainer exec ${APPTAINER} gmx energy -f npt.edr -o potential.xvg < input.temp

mv ${cwd}/slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${dir}/.


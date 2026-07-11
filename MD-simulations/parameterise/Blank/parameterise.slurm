#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=YOUR_PARTITION
#BATCH -C NOGPU

####################
# DIRECTORIES      #
###################

PROJECT_ROOT=/INSERT/YOUR/PATH
SEED_DIR=/INSERT/YOUR/PATH
DATA_DIR=0
if [[ ${DATA_DIR} -eq 0 ]]; then
	echo "Please open parameterise.slurm and set environment variables"
	echo "STOPPING NOW"
	exit
fi

############
# MODULES  #
############

module purge
module load PDC/24.11
module load amber/24-cpeGNU-24.11-ambertools-24
module load GCC/14.3.0

########################################
#              CODE                    #
########################################
molecule=XXXMOLECULEXXX

cwd=$(pwd)
dir="${DATA_DIR}/parameterise/${molecule}"

rm -r $dir
mkdir ${dir}
cd ${dir}

#Change the following when carrying out reference calculations for acids 
cp ${PROJECT_ROOT}/seedfiles/surfpro/${molecule}.smi . 


obabel -ismi ${molecule}.smi --gen3d -opdb -Omolecule.pdb
grep -v MASTER molecule.pdb > mol.temp
grep -v CONECT mol.temp > molecule.pdb

total_charge=$(( $(head -n 1 ${molecule}.smi | tr -cd '+' | wc -c) - $(head -n 1 ${molecule}.smi | tr -cd '-' | wc -c) ))
antechamber -i molecule.pdb -fi pdb -o mol.mol2 -fo mol2 -c bcc -nc ${total_charge} -rn MOL -at gaff2 

parmchk2 -i mol.mol2 -f mol2 -o mol.frcmod -s gaff2

echo '''
source leaprc.gaff
MOL = loadmol2 mol.mol2
check MOL
loadamberparams mol.frcmod
saveoff MOL mol.lib
saveamberparm MOL mol.prmtop mol.inpcrd
quit
''' > leap.in

tleap -f leap.in
run_acpype.py -x mol.inpcrd -p mol.prmtop -o gmx

cp MOL.amb2gmx/MOL_GMX.gro molecule.gro
cp MOL.amb2gmx/MOL_GMX.top mol_topol.top 

cp ${cwd}/slurm-${SLURM_JOB_ID}.out ${dir}/.


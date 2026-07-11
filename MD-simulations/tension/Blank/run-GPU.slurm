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

molecule=surfpro_XXXMOLIDXXX
nmol=XXXNMOLXXX
watermax=XXXWATERMAXXXX

temperature=298.15

#########################################
#	      CODE			#
#########################################
echo "starting run for $molecule with $nmol molecules"
APPTAINER=${PROJECT_ROOT}/MD-simulations/gromacs.sif

cwd=$(pwd)
dir="${DATA_DIR}/multiples/${molecule}_${nmol}"
dir="${DATA_DIR}/tension/${molecule}_${nmol}"

rm -r $dir
mkdir ${dir}
cd ${dir}

srcdir="${DATA_DIR}/equilibrations/${molecule}_${nmol}/long-NPT"

MAX_PARALLEL=4
TOTAL_RUNS=20

for i in $(seq 0 $TOTAL_RUNS); do
    run_dir=${dir}/ST_$(printf "%02g" $i)
    mkdir -p $run_dir
    cd $run_dir

    cp ${srcdir}/npt_$(printf "%02g" $i ).gro ${srcdir}/topol.top ${cwd}/Blank/production-ST.mdp ${cwd}/Blank/tip4p-ice.itp .
    mv production-ST.mdp production.mdp

    tail -n 1 npt_$(printf "%02g" $i ).gro > box.temp
    x=$(awk '{print $1 }' box.temp )
    y=$(awk '{print $2 }' box.temp )

    sed -i "s/ref_t.*/ref_t = ${temperature}  ${temperature}/g" production.mdp
    if [ ${nmol} == 0 ]; then
       sed -i "s/MOL  SOL/SOL  /g" production.mdp
       sed -i "s/ref_t.*/ref_t = ${temperature}  /g" production.mdp
       sed -i "s/tau_t.*/tau_t = 0.1  /g" production.mdp
    fi

    apptainer exec ${APPTAINER} gmx editconf -f npt_$(printf "%02g" $i ).gro -o new.gro -box ${x} ${y} 30
    apptainer exec ${APPTAINER} gmx grompp -maxwarn 1 -f production.mdp -c new.gro -p topol.top -o production.tpr

    # start mdrun in background
    apptainer exec -nv ${APPTAINER} gmx mdrun \
    -deffnm production \
    -ntmpi 1 \
    -ntomp 4 \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -tunepme & 
    -deffnm production &

    # wait if MAX_PARALLEL jobs are already running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done

    cd $cwd
done

# wait for all remaining background simulations to finish
wait

# now process outputs sequentially
for i in $(seq 0 $TOTAL_RUNS); do
    run_dir=${dir}/ST_$(printf "%02g" $i)
    cd $run_dir

    # energy extraction
    echo -e "30\n31\n32\n33\n34\n35\n36\n37\n38\n39\n40\n" > input.temp
    apptainer exec ${APPTAINER} gmx energy -f production.edr -o potential.xvg < input.temp

    # trajectory conversion
    apptainer exec ${APPTAINER} gmx trjconv -f production.xtc -o ST.pdb -b 3000 -e 5000 -s production.tpr <<EOF
2
EOF

    cp ../setup/molecule.pdb .
    python ${cwd}/Blank/compute_orientation.py ${nmol} > orientation.dat

    cd $cwd
done

mv ${cwd}/slurm-${SLURM_JOB_ID}.out ${dir}/.

#mv ${dir} ${DATA_DIR}/.

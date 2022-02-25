#!/bin/bash
#SBATCH -p med2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH -J qc_local1
#SBATCH -t 10-00:00:00
#SBATCH --no-requeue
#--mem=16000

geometric-neb psi4.in --images 21 --coordsys cart --engine psi4 --coords initial.xyz


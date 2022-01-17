#!/bin/bash
#SBATCH -p med2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH -J qc_local1
#SBATCH -t 10-00:00:00
#SBATCH --no-requeue
#--mem=16000

geometric-neb --guessw 0.5 --guessk 0.01 --skip --images 21 --coordsys cart --engine psi4 --plain 0 --ew --coords initial.xyz --maxcyc 200 --avgg 0.025 --maxg 0.05 --input tera.in


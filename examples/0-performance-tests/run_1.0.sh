#!/bin/bash

for i in azithromycin bucky_catcher calicheamicin carbene-ts cu-i-carborane dcb-tweezer diphenylmethane fe4n_catalyst fe-bpte-ncbh3 gcgc-stack hcccn_6 hcn-hnc-ts heme imatinib-salt ivermectin malonate-ts psb3-qmmm-meci sialyl-lewis-x taxol trp-cage_openmm vomilenine water6_qchem ; do
#for i in heme imatinib-salt ivermectin malonate-ts psb3-qmmm-meci sialyl-lewis-x taxol trp-cage_openmm vomilenine water6_qchem ; do
#for i in taxol trp-cage_openmm vomilenine water6_qchem ; do
    cd $i
    mkdir -p saved/v1.0
    cd saved/v1.0
    for f in $(ls -p ../.. | grep -v / | grep -v \.pdf | grep -v \.log | grep -v _optim\.xyz) ; do
        ln -s ../../$f .
    done
    sh command.sh
    bzip2 run_optim.xyz
    rm -rf run.tmp .EDISP
    # Remove symlinks so git ls-files --others won't see them.
    find . -type l -exec rm {} \;
    cd ../..
    cd ..
done


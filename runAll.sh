#!/bin/bash
nJobs=1
beta=225
xiR=2
N=24
T=24
Rmax=12
Tmax=12
runtime=00:10:00

#makes 'Configs' and 'Data' folders in filepath location



for beta in 225 230 235 240 245 250 255 260 265 270 275
do
for xiR in 1 2 3 4 5 6 7 8
do
base=L"$N"_b"$beta"_xi"$xiR"_g
datfile=./Data/"$base".dat
log=./Fits/"$base".log
fitfile=./Fits/"$base".vr

make && sbatch --time=$runtime submit.script $datfile $beta $xiR $N $T $Rmax $Tmax $fitfile $log
#make && ./bin/WStats $datfile $beta $xiR $N $T $Rmax $Tmax $fitfile $log
done
done


#PBS -l nodes=1:ppn=1:gpus=1,walltime=20:00:00
#PBS -N monatomic_LJ-optimization
#PBS -q batch
#PBS -j oe
#PBS -m bae
#PBS -M tcmoore3@gmail.com
#PBS -V

cd $PBS_O_WORKDIR
g++ -funroll-loops -O3 optimize.cpp -o opt
./opt > run.log
rm ./opt

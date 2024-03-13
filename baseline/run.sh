#! /bin/bash -l

#SBATCH -J baseline
#SBATCH -o baseline.%j.out
#SBATCH -e baseline.%j.err
#SBATCH -p small
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=12G
#SBATCH -A project_2005047
#SBATCH -t 1:00:00

module load python-data/3.10-23.11

for LANG in EN ES PT FR; do
    for METHOD in atomic expand; do
        python3 classify_svm.py $LANG $METHOD test
    done
done

# for BCMS, the 'expand' mode doesn't make sense because the training data is single-label
python3 classify_svm.py BCMS atomic test

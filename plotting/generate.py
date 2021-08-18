#Define list of hyperparameter varations
metrics = ['SRdefault', 'SRLRFdefault', 'SRsquare','SRLRFsquare', 'SRcov','SRLRFcov',
           'ERdefault', 'ERLRFdefault', 'ERsquare','ERLRFsquare', 'ERcov','ERLRFcov']
zetas   = { 'SRdefault':    [1, 2, 4, 8, 16, 32, 64, 128], 
            'SRLRFdefault': [1, 2, 4, 8, 16, 32, 64, 128], 
            'SRsquare':     [1, 2, 4, 8, 16, 32, 64, 128],
            'SRLRFsquare':  [1, 2, 4, 8, 16, 32, 64, 128], 
            'SRcov':        [1, 2, 4, 8, 16, 32, 64, 128],
            'SRLRFcov':     [1, 2, 4, 8, 16, 32, 64, 128],
            'ERdefault':    [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128], 
            'ERLRFdefault': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128], 
            'ERsquare':     [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128],
            'ERLRFsquare':  [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128], 
            'ERcov':        [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128],
            'ERLRFcov':     [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128]
           }


default = '''#!/bin/bash 
### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --account=def-msh
#SBATCH --time=24:00:0

source ~/scratch/August/HRMSGD/EnvAdas/bin/activate

'''

#Specify the constants in the config file


i = 0
for idm, metric in enumerate(metrics):
    f = open("f:/Research/Multimedia/August/HRMSGD/plotting/batch_files/" + metric + ".sh", "w")
    f.write(default)
    for z in zetas[metric]:
    
            print(i, ": python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + str(metric) + "' --zeta=" + str(z))
            i += 1

            f.write("python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + str(metric) + "' --zeta=" + str(z))
            f.write("\n")
    f.close()
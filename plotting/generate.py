#Define list of hyperparameter varations
'''
names = ['SRdefault_step0', 'SRdefault_step1', 'ERdefault_step0','ERdefault_step1',
         'SRdefault_mome0', 'SRdefault_mome1', 'ERdefault_mome0','ERdefault_mome1']'''
zetas   = { 'SRready0':    [15, 30, 45, 60],
            'SRready1':    [75, 90, 105, 120],
            'ERready0':    [0.03, 0.08, 0.13, 0.18],
            'ERready1':    [0.23, 0.28, 0.33, 0.38],
            'SRlrmom0':    [30],
            'SRlrmom1':    [60],
            'SRlrmom2':    [90],
            'ERlrmom0':    [0.08],
            'ERlrmom1':    [0.18],
            'ERlrmom2':    [0.28]
           }
betas = { 'SRready0':    [0],
            'SRready1':    [0],
            'ERready0':    [0],
            'ERready1':    [0],
            'SRlrmom0':    [0.1,0.2,0.3],
            'SRlrmom1':    [0.1,0.2,0.3],
            'SRlrmom2':    [0.1,0.2,0.3],
            'ERlrmom0':    [0.1,0.2,0.3],
            'ERlrmom1':    [0.1,0.2,0.3],
            'ERlrmom2':    [0.1,0.2,0.3]
           }
configs = ['config_step.yaml', 'config_mome.yamls']

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

#Specify the constants in the config file'''
'''i = 0
for name in names:
        metric = name.split('_')[0]
        f = open("f:/Research/Multimedia/August/HRMSGD/plotting/batch_files/" + name + ".sh", "w")
        f.write(default)

        config = name.split('_')[1][:-1]
        config = 'config_' + config + '.yaml'

        for z in zetas[metric + name.split('_')[1][-1]]:
        
            print(i, ": python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + metric + "' --zeta=" + str(z) + ' --config=\'' + config + "'")
            i += 1

            f.write("python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + metric + "' --zeta=" + str(z) + ' --config=\'' + config + "'")
            f.write("\n")

        f.close()
'''
# i = 0
# for idm, metric in enumerate(zetas.keys()):
#     f = open("f:/Research/Multimedia/August/HRMSGD/plotting/batch_files/" + metric + ".sh", "w")
#     f.write(default)
#     for z in zetas[metric]:
    
#             print(i, ": python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + str(metrics[idm]) + "' --zeta=" + str(z))
#             i += 1

#             f.write("python ~/scratch/August/HRMSGD/Adas-paper/src/adas/train.py --measure='" + str(metrics[idm]) + "' --zeta=" + str(z))
#             f.write("\n")
#     f.close()
for name, zeta in zetas.items():
    f = open(name + ".sh", "w")
    f.write(default)
    for z in zeta:
        for b in betas[name]:
            f.write("python train.py --config='"+name[2:-1]+"config.yaml"+"' --measure='" + str(name[0:2]) + "' --zeta=" + str(z)+" --beta="+str(b)+"\n")
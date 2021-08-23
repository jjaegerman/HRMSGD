import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import glob
from PIL import Image as im

MAX = ['4000']
processing = ['default']
metrics = ['${\widehat{Q}_{SR}}$','${{Q}_{SR}}$','$\widehat{Q}_{E}$','${Q}_{E}$']
model = 'RESNET18'

if model=='RESNET18':
    depth = 62
    conv = list(np.arange(0,60,3))
    conv.remove(21)
    conv.remove(36)
    conv.remove(51)
    std = np.arange(1,60,3)
    mean = np.arange(2,60,3)
    linear = [60]
    bias = [61]
    dwise = []
    pwise = []
    expan = []

if model=='SENET18':
    depth = 88
    conv = [0,5,8,15,18,25,28,36,39,46,49,57,60,67,70,78,81]
    std = [1,3,6,13,16,23,26,34,37,44,47,55,58,65,68,76,79]
    mean = [s+1 for s in std]
    linear = [9, 11, 19, 21, 30, 32, 40, 42, 51, 53, 61, 63, 72, 74, 82, 84, 86]
    bias = [b+1 for b in linear]
    dwise = []
    pwise = []
    expan = []

if model=='MOBILENETV2':
    depth = 158
    dwise = np.arange(3,156,9)
    pwise = np.arange(6,156,9)
    expan = np.arange(0,156,9)
    std = np.arange(1,156,3)
    mean = np.arange(2,156,3)
    linear = [156]
    bias = [157]
    conv = []

layers = list()

for layer in list(range(depth)):
    if layer in conv:
    	layers.append("conv")
    elif layer in std:
        layers.append("std")
    elif layer in mean:
        layers.append("mean")
    elif layer in linear:
        layers.append("linear")
    elif layer in bias:
        layers.append("bias")
    elif layer in dwise:
        layers.append("depth-wise")
    elif layer in pwise:
        layers.append("point-wise")
    elif layer in expan:
        layers.append("expansion")
    else:
        layers.append("skip")

filename='measureevo/results_date=2021-08-19-01-26-44_trial=0_ResNet18CIFAR_CIFAR10_HRMSGDweight_decay=0.0_momentum=0.0_None_LR=0.03_measure=ERdefault_zeta=0.23.pickle'
beta = ''
for block in filename.split('_'):
    if "measure" in block:
        measure = block.replace("measure=",'')
    if "zeta" in block:
        zeta = block.replace("zeta=",'')
    if "beta" in block:
        beta = block.replace("beta=",'')
measure = filename.split('_')[-2].split('=')[-1]
zeta = filename.split('_')[-1].split('=')[-1][:-7]
infile = open(filename,'rb')
outfile =  pickle.load(infile,  encoding='bytes')
infile.close()
for process in processing:
    plotting = {"conv":[[],[],[],[]], "std":[[],[],[],[]], "mean":[[],[],[],[]], "linear":[[],[],[],[]], "bias":[[],[],[],[]], "depth-wise":[[],[],[],[]], "point-wise":[[],[],[],[]], "expansion":[[],[],[],[]], "skip":[[],[],[],[]]}
    for i, layer in enumerate(layers):
        plotting[layer][0].append(outfile[process]['SRLRF'][i])
        plotting[layer][1].append(outfile[process]['SR'][i])
        plotting[layer][2].append(outfile[process]['FNLRF'][i])
        plotting[layer][3].append(outfile[process]['FN'][i])
    for layer_type, plot in plotting.items():
        for x, metric in enumerate(plot):
            length = len(metric)
            if length==0:
                continue
            for i, layer in enumerate(metric):
                plt.plot(layer, c=[i/length, 0, (length-i)/length,1-np.abs((1/(1.5*3.141592))*np.arctan(3*(i-length/2)/length))])
            plt.title(layer_type+" "+ process+" "+ metrics[x]+ " trained on " + measure+ zeta)
            plt.ylabel(metrics[x])
            plt.xlabel("Epoch")
            plt.savefig('measureplots/' + model +'_'+layer_type+'_'+measure+'-'.join(zeta.split('.'))+'-'.join(beta.split('.'))+"_"+process+"-"+str(x)+".png")
            plt.clf()
exit()
    
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_headers(df):
    keys = list(df.keys())[1:]
    headers = list()
    for key in keys:
        key = ''.join(key.split('_')[:-2]).lower()
        if key not in headers:
            headers.append(key)
        else:
            break
    assert(len(set(headers)) == len(headers))
    return headers

if __name__ == "__main__":
    f='metricevo/results_date=2021-08-17-03-54-44_trial=0_ResNet18CIFAR_CIFAR10_HRMSGD_None_LR=0.03_measure=ERdefault_zeta=0.16.xlsx'
    colors = ['#003f5c' , '#634e86','#ff6e54', '#ffa600']
    model = "ResNet18"
    df = pd.read_excel(f)
    headers = get_headers(df)
    print(headers)
    df = df.T
    tag = 'testacc1'
    if 'testacc1' not in headers:
        tag = 'testacc'
        if 'testacc' not in headers:
            tag = 'acc'
    test_acc = np.asarray(
        df.iloc[headers.index(tag) + 1::len(headers), :])[:,0]
    tag = 'trainacc1'
    if 'trainacc1' not in headers:
        tag = 'trainacc'
    try:
        train_acc = np.asarray(
            df.iloc[headers.index(tag) + 1::len(headers), :])[:,0]
    except:
        print(1)
    test_loss = np.asarray(
        df.iloc[headers.index('testloss')+ 1::len(headers), :])[:,0]
    train_loss = np.asarray(
        df.iloc[headers.index('trainloss')+ 1::len(headers), :])[:,0]



    plt.plot(test_acc, c=colors[0], label="Test Accuracy")
    plt.plot(test_loss, c=colors[1], label="Test Loss")
    plt.plot(train_acc, c=colors[2], label="Train Accuracy")
    plt.plot(train_loss, c=colors[3], label="Train Loss")
    plt.legend()
    plt.xlabel("Epoch")
    measure = f.split('_')[-2].split('=')[-1]
    zeta = f.split('_')[-1].split('=')[-1][:-5]
    plt.title("Performance Metrics using HRMSGD measure: "+measure+" zeta: "+zeta)
    #we can also plot baseline with dashed lines or something

    #Baseline
    f='metricevo/results_date2021-08-16-22-30-26_trial0_ResNet18CIFAR_CIFAR10_SGDweight_decay0.0_momentum0.0_StepLRst.xlsx'
    colors = ['#003f5c' , '#634e86','#ff6e54', '#ffa600']
    model = "ResNet18"
    df = pd.read_excel(f)
    headers = get_headers(df)
    print(headers)
    df = df.T
    tag = 'testacc1'
    if 'testacc1' not in headers:
        tag = 'testacc'
        if 'testacc' not in headers:
            tag = 'acc'
    test_acc = np.asarray(
        df.iloc[headers.index(tag) + 1::len(headers), :])[:,0]
    tag = 'trainacc1'
    if 'trainacc1' not in headers:
        tag = 'trainacc'
    try:
        train_acc = np.asarray(
            df.iloc[headers.index(tag) + 1::len(headers), :])[:,0]
    except:
        print(1)
    test_loss = np.asarray(
        df.iloc[headers.index('testloss')+ 1::len(headers), :])[:,0]
    train_loss = np.asarray(
        df.iloc[headers.index('trainloss')+ 1::len(headers), :])[:,0]



    plt.plot(test_acc, c=colors[0], label="(Base) Test Accuracy", linestyle='dashed')
    plt.plot(test_loss, c=colors[1], label="(Base) Test Loss", linestyle='dashed')
    plt.plot(train_acc, c=colors[2], label="(Base) Train Accuracy", linestyle='dashed')
    plt.plot(train_loss, c=colors[3], label="(Base) Train Loss", linestyle='dashed')
    plt.legend()
    plt.xlabel("Epoch")
    measure = f.split('_')[-2].split('=')[-1]
    zeta = f.split('_')[-1].split('=')[-1][:-5]
    plt.title("Performance Metrics using HRMSGD measure: "+measure+" zeta: "+zeta)
    #we can also plot baseline with dashed lines or something
    plt.savefig('metricplots/'+model+"_"+measure+'-'.join(zeta.split('.'))+".png")

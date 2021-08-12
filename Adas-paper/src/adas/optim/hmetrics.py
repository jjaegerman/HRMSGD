"""
"""
from typing import List, Union, Tuple

import sys

import numpy as np
import torch

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .matrix_factorization import EVBMF
else:
    from optim.matrix_factorization import EVBMF


class Metrics:
    def __init__(self, params, MAX, S, measure) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.history = list()
        mask = list()
        #create mask
        self.measure = measure
        self.mask = set(mask)
        self.MAX = MAX
        self.len = S
        self.weight_hist = list()
        self.random_selection = list()
        for layer_index, layer in enumerate(self.params):
            self.weight_hist.append(0)
            self.random_selection.append(0)
            layer_tensor = layer.data
            if(np.prod(layer_tensor.shape)>self.MAX):
                self.random_selection[layer_index] = np.random.choice(np.prod(layer_tensor.shape)-1,size=self.MAX,replace=False)
                self.weight_hist[layer_index]=np.empty((self.MAX,self.len))
            else:
                self.weight_hist[layer_index]=np.empty((np.prod(layer_tensor.shape),self.len))

    def __call__(self):
        '''
        Updates weight history
        '''
        for layer_index, layer in enumerate(self.params):
            weight = layer.cpu().data
            weight = torch.reshape(weight, [-1,1])
            weight = np.append(weight, [[0,0,0,0,0]])
            weight = np.expand_dims(weight,1)
            weight = weight[:(self.weight_hist[layer_index].shape)[0],:]
            if(weight.shape[0]>self.MAX):
                weight = weight[self.random_selection[layer_index],:]
            self.weight_hist[layer_index] = np.concatenate((self.weight_hist[layer_index], weight),axis=1)
            self.weight_hist[layer_index] = self.weight_hist[layer_index][:,1:]    
        return None

    def update(self):
        '''
        returns metrics on recent window
        '''
        out = list()

        for layer_index in range(len(self.weight_hist)):
            #formattin, slicing
            #can add the random selection stuff here, vert and horizontal
            slice = self.weight_hist[layer_index][:,-self.len:]
            slice_shape = slice.shape
            if(slice_shape[0]>slice_shape[1]):
                slice = slice.T
                slice_shape = slice.shape
            slice = torch.from_numpy(slice)
            if("square" in self.measure):
                slice = np.matmul(slice,slice.T)

            if("cov" in self.measure):
                #LRF
                if("LRF" in self.measure):
                    U_approx, S_approx, V_approx = EVBMF(slice)
                    if(len(torch.diag(S_approx).data.numpy())!=0):
                        low_slice = np.matmul(np.matmul(U_approx.data.numpy(),S_approx.data.numpy()),V_approx.data.numpy().T)
                        low_rank_cov = np.cov(low_slice, rowvar=True)
                        low_rank_eigen = np.linalg.eigvals(low_rank_cov)
                        if("SR" in self.measure):
                            out.append((np.sum(low_rank_eigen/np.max(low_rank_eigen))-1)/slice_shape[0])
                        elif("FN" in self.measure):
                            out.append(np.sqrt(np.sum(low_rank_eigen**2)))
                        elif("ER" in self.measure):
                            out.append(-np.sum(np.multiply(low_rank_eigen/np.sum(low_rank_eigen),np.log(low_rank_eigen/np.sum(low_rank_eigen)))))
                    else:
                        out.append(0)
                else:
                    #raw
                    slice = slice.data.numpy()
                    cov = np.cov(slice, rowvar=True)
                    eigen = np.linalg.eigvals(cov)
                    if("SR" in self.measure):
                        out.append(np.sum((eigen/np.max(eigen))-1)/slice_shape[0])
                    elif("FN" in self.measure):
                        out.append(np.sqrt(np.sum(eigen**2)))
                    elif("ER" in self.measure):
                        out.append(-np.sum(np.multiply(eigen/np.sum(eigen),np.log(eigen/np.sum(eigen)))))
            else:
                #LRF
                if("LRF" in self.measure):
                    U_approx, S_approx, V_approx = EVBMF(slice)
                    low_rank_eigen = torch.diag(S_approx).data.numpy()
                    if(len(torch.diag(S_approx).data.numpy())!=0):
                        if("SR" in self.measure):
                            out.append(np.sum(low_rank_eigen/np.max(low_rank_eigen))/slice_shape[0])
                        elif("FN" in self.measure):
                            out.append(np.sqrt(np.sum(low_rank_eigen**2)))
                        elif("ER" in self.measure):
                            out.append(-np.sum(np.multiply(low_rank_eigen/np.sum(low_rank_eigen),np.log(low_rank_eigen/np.sum(low_rank_eigen)))))
                    else:
                        out.append(0)
                else:
                    #raw
                    U, S, V = torch.svd(slice)
                    eigen = S.data.numpy()
                    if("SR" in self.measure):
                        out.append(np.sum(eigen/np.max(eigen))/slice_shape[0])
                    elif("FN" in self.measure):
                        out.append(np.sqrt(np.sum(eigen**2)))
                    elif("ER" in self.measure):
                        out.append(-np.sum(np.multiply(eigen/np.sum(eigen),np.log(eigen/np.sum(eigen)))))
        return np.array(out)
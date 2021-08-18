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
        self.history = {'square':dict(),'default':dict(),'cov':dict(), 'lr':list()}
        for key in self.history:
            if key == 'lr':
                continue
            self.history[key] = {'SRLRF':[],'SR':[],'FNLRF':[],'FN':[],'ERLRF':[],'ER':[]}
        
        mask = list()
        #create mask
        self.measure = measure
        self.mask = set(mask)
        self.MAX = MAX
        self.len = S
        self.weight_hist = list()
        self.random_selection = list()
        for layer_index, layer in enumerate(self.params):
            for key in self.history:
                if(key=='lr'):
                    continue
                for metric in self.history[key]:
                    self.history[key][metric].append([])
            self.history['lr'].append([])
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
        returns metrics on recent window for updating of learning rate
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
                        low_rank_cov = np.cov(low_slice, rowvar=True, bias=1)
                        low_rank_eigen = np.linalg.eigvals(low_rank_cov)
                        low_rank_eigen = np.absolute(low_rank_eigen)
                        low_rank_eigen = low_rank_eigen[low_rank_eigen!=0]
                        if(len(low_rank_eigen)!=0):
                            if("SR" in self.measure):
                                out.append((np.sum(low_rank_eigen/np.max(low_rank_eigen))-1)/slice_shape[0])
                            elif("FN" in self.measure):
                                out.append(np.sqrt(np.sum(low_rank_eigen**2)))
                            elif("ER" in self.measure):
                                out.append(-np.sum(np.multiply(low_rank_eigen/np.sum(low_rank_eigen),np.log(low_rank_eigen/np.sum(low_rank_eigen)))))
                        else:
                            out.append(0)
                    else:
                        out.append(0)
                else:
                    #raw
                    slice = slice.data.numpy()
                    cov = np.cov(slice, rowvar=True, bias=1)
                    eigen = np.linalg.eigvals(cov)
                    eigen = np.absolute(eigen)
                    eigen = eigen[eigen!=0]
                    if(len(eigen)!=0):
                        if("SR" in self.measure):
                            out.append((np.sum(eigen/np.max(eigen))-1)/slice_shape[0])
                        elif("FN" in self.measure):
                            out.append(np.sqrt(np.sum(eigen**2)))
                        elif("ER" in self.measure):
                            out.append(-np.sum(np.multiply(eigen/np.sum(eigen),np.log(eigen/np.sum(eigen)))))
                    else:
                        out.append(0)
            else:
                #LRF
                if("LRF" in self.measure):
                    U_approx, S_approx, V_approx = EVBMF(slice)
                    low_rank_eigen = torch.diag(S_approx).data.numpy()
                    low_rank_eigen = low_rank_eigen[low_rank_eigen!=0]
                    if(len(low_rank_eigen)!=0):
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
                    U, S, V = torch.svd(slice)
                    eigen = S.data.numpy()
                    eigen = eigen[eigen!=0]
                    if(len(eigen)!=0):
                        if("SR" in self.measure):
                            out.append((np.sum(eigen/np.max(eigen))-1)/slice_shape[0])
                        elif("FN" in self.measure):
                            out.append(np.sqrt(np.sum(eigen**2)))
                        elif("ER" in self.measure):
                            out.append(-np.sum(np.multiply(eigen/np.sum(eigen),np.log(eigen/np.sum(eigen)))))
                    else:
                        out.append(0)
        return np.array(out)

    def history_update(self):
        '''
        updates history of metrics
        '''
        for layer_index in range(len(self.weight_hist)):
            #formattin, slicing
            #can add the random selection stuff here, vert and horizontal
            slice = self.weight_hist[layer_index][:,-self.len:]
            slice_shape = slice.shape
            if(slice_shape[0]>slice_shape[1]):
                slice = slice.T
                slice_shape = slice.shape

            slice = torch.from_numpy(slice)

            square = np.matmul(slice,slice.T)
            
            _, square_svals, __ = torch.svd(square)
            square_svals = square_svals.data.numpy()
            square_svals = square_svals[square_svals!=0]
            if(len(square_svals)!=0):
                self.history['square']["SR"][layer_index].append((np.sum(square_svals/np.max(square_svals))-1)/(square_svals.shape)[0])
                self.history['square']["FN"][layer_index].append(np.sqrt(np.sum(square_svals**2)))
                self.history['square']["ER"][layer_index].append(-np.sum(np.multiply(square_svals/np.sum(square_svals),np.log(square_svals/np.sum(square_svals)))))
            else:
                self.history['square']["SR"][layer_index].append(0)
                self.history['square']["FN"][layer_index].append(0)
                self.history['square']["ER"][layer_index].append(0)

            _, low_rank_square_svals, _ = EVBMF(square)
            low_rank_square_svals = torch.diag(low_rank_square_svals).data.numpy()
            low_rank_square_svals = low_rank_square_svals[low_rank_square_svals!=0]
            if(len(low_rank_square_svals != 0)):
                self.history['square']["SRLRF"][layer_index].append((np.sum(low_rank_square_svals/np.max(low_rank_square_svals))-1)/(low_rank_square_svals.shape)[0])
                self.history['square']["FNLRF"][layer_index].append(np.sqrt(np.sum(low_rank_square_svals**2)))
                self.history['square']["ERLRF"][layer_index].append(-np.sum(np.multiply(low_rank_square_svals/np.sum(low_rank_square_svals),np.log(low_rank_square_svals/np.sum(low_rank_square_svals)))))
            else:
                self.history['square']["SRLRF"][layer_index].append(0)
                self.history['square']["FNLRF"][layer_index].append(0)
                self.history['square']["ERLRF"][layer_index].append(0)

            U_approx, S_approx, V_approx = EVBMF(slice)
            low_rank_slice = np.matmul(np.matmul(U_approx.data.numpy(),S_approx.data.numpy()),V_approx.data.numpy().T)
            low_rank_svals = torch.diag(S_approx).data.numpy()
            low_rank_svals = low_rank_svals[low_rank_svals!=0]
            if(len(low_rank_svals!=0)):
                low_rank_cov = np.cov(low_rank_slice, rowvar=True, bias=1)
                low_rank_cov_eigen = np.linalg.eigvals(low_rank_cov)
                low_rank_cov_eigen = np.absolute(low_rank_cov_eigen)
                low_rank_cov_eigen = low_rank_cov_eigen[low_rank_cov_eigen!=0]
                if(len(low_rank_cov_eigen)!=0):
                    self.history['cov']["SRLRF"][layer_index].append((np.sum(low_rank_cov_eigen/np.max(low_rank_cov_eigen))-1)/(low_rank_cov_eigen.shape)[0])
                    self.history['cov']["FNLRF"][layer_index].append(np.sqrt(np.sum(low_rank_cov_eigen**2)))
                    self.history['cov']["ERLRF"][layer_index].append(-np.sum(np.multiply(low_rank_cov_eigen/np.sum(low_rank_cov_eigen),np.log(low_rank_cov_eigen/np.sum(low_rank_cov_eigen)))))
                else:
                    self.history['cov']["SRLRF"][layer_index].append(0)
                    self.history['cov']["FNLRF"][layer_index].append(0)
                    self.history['cov']["ERLRF"][layer_index].append(0)
                self.history['default']["SRLRF"][layer_index].append((np.sum(low_rank_svals/np.max(low_rank_svals))-1)/(low_rank_svals.shape)[0])
                self.history['default']["FNLRF"][layer_index].append(np.sqrt(np.sum(low_rank_svals**2)))
                self.history['default']["ERLRF"][layer_index].append(-np.sum(np.multiply(low_rank_svals/np.sum(low_rank_svals),np.log(low_rank_svals/np.sum(low_rank_svals)))))
            else:
                self.history['cov']["SRLRF"][layer_index].append(0)
                self.history['cov']["FNLRF"][layer_index].append(0)
                self.history['cov']["ERLRF"][layer_index].append(0)
                self.history['default']["SRLRF"][layer_index].append(0)
                self.history['default']["FNLRF"][layer_index].append(0)
                self.history['default']["ERLRF"][layer_index].append(0)

            U, S, V = torch.svd(slice)
            svals = S.data.numpy()
            svals = svals[svals!=0]
            if(len(svals)!=0):
                self.history['default']["SR"][layer_index].append((np.sum(svals/np.max(svals))-1)/(svals.shape)[0])
                self.history['default']["FN"][layer_index].append(np.sqrt(np.sum(svals**2)))
                self.history['default']["ER"][layer_index].append(-np.sum(np.multiply(svals/np.sum(svals),np.log(svals/np.sum(svals)))))
            else:
                self.history['default']["SR"][layer_index].append(0)
                self.history['default']["FN"][layer_index].append(0)
                self.history['default']["ER"][layer_index].append(0)

            cov = np.cov(slice, rowvar=True, bias=1)
            cov_eigen = np.linalg.eigvals(cov)
            cov_eigen = np.absolute(cov_eigen)
            cov_eigen = cov_eigen[cov_eigen!=0]
            if(len(cov_eigen)!=0):
                self.history['cov']["SR"][layer_index].append((np.sum(cov_eigen/np.max(cov_eigen))-1)/(cov_eigen.shape)[0])
                self.history['cov']["FN"][layer_index].append(np.sqrt(np.sum(cov_eigen**2)))
                self.history['cov']["ER"][layer_index].append(-np.sum(np.multiply(cov_eigen/np.sum(cov_eigen),np.log(cov_eigen/np.sum(cov_eigen)))))
            else:
                self.history['cov']["SR"][layer_index].append(0)
                self.history['cov']["FN"][layer_index].append(0)
                self.history['cov']["ER"][layer_index].append(0)

        return self.history
            
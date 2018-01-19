# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 18:42:16 2017

@author: ht
"""

from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt

f = open('gridsearch_result_option2_w32_reg.save', 'rb')
result = cPickle.load(f)
f.close()

params=np.zeros((len(result),8),dtype=np.float32)

for i in xrange(len(result)):
    param,score=result[i]

    #auc=score['auc-mean']
    logloss=score['binary_logloss-mean']
    
    #assert(len(auc)==len(logloss))
    nRound=len(logloss)
    #best_auc=auc[nRound-1]
    best_logloss=logloss[nRound-1]
    params[i]=np.array([param[0],param[1],param[2],param[3],param[4],param[5],nRound,best_logloss])#best_auc,
    
target=params[:,7].argsort()[:3]
best_param=params[np.argmin(params[:,7])]


plt.plot(np.arange(0,len(result)),params[:,7],'r')
    
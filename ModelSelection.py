# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:56:27 2016

@author: Hajime
"""

import numpy as np
from sklearn import model_selection

class ModelSelection:
    
    def __init__(self, EstClass, Data_All, models, setting, cvtype='KFold',cvsetting={},cvhypara={}):
        self.EstClass = EstClass
        self.Data_All = Data_All
        self.models = models #list of dictionary. If dictionary of list, model_selection.ParameterGrid(models)
        self.N_models = len(list(models))
        self.setting = setting
        if cvtype=='KFold':
            self.cvsplit = model_selection.KFold(**cvsetting)
        if cvtype=='GroupKFold':
            self.cvsplit = model_selection.GroupKFold(**cvsetting)
        self.cvhypara = cvhypara
            
        
    def score(self, Data_train, Data_test , model):
        est = self.EstClass( **model, **self.setting )
        est.fit(Data=Data_train)
        score_model = est.score(Data=Data_test)
        return score_model
        
    def fit(self):
        score_models = []
        for train_index, test_index in self.cvsplit.split(self.Data_All, **self.cvhypara):
            Data_train, Data_test = self.Data_All[train_index], self.Data_All[test_index]
            score_models_eachsplit=[]
            for model in self.models:
                score_model = self.score(Data_train, Data_test , model)
                score_models_eachsplit.append(score_model)
            score_models.append(score_models_eachsplit)
        self.score_models_all = score_models
        self.score_models = np.mean(score_models,axis=0)
        self.best_model = np.array(self.models)[np.where( np.max(self.score_models)==self.score_models )]
        self.score_std = np.std(score_models,axis=0)
        

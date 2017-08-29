# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:56:27 2016

@author: Hajime
"""

import numpy as np
from sklearn import model_selection
from itertools import product

class ModelSelection:
    
    def __init__(self, EstClass, Data_All, models, setting, cvtype='KFold',cvsetting={},cvhypara={}):
        self.EstClass = EstClass
        self.Data_All = Data_All
        self.models = models #list of dictionary. If dictionary of list, model_selection.ParameterGrid(models)
        self.N_models = len(list(models))
        self.setting = setting
        self.cvtype = cvtype
        if cvtype=='KFold':
            self.cvsplit = model_selection.KFold(**cvsetting)
        if cvtype=='GroupKFold':
            self.cvsplit = model_selection.GroupKFold(**cvsetting)
        self.cvhypara = cvhypara
        
        self.warning = 0
            
        
    def score(self, Data_train, Data_test , model):
        est = self.EstClass( **model, **self.setting )
        est.fit(Data=Data_train)
        score_model,moment_vec = est.score(Data=Data_test)
        if est.warning!=0:
            self.warning =est.warning
        score_model = score_model.flatten()
        return score_model,moment_vec
        
    def fit(self):
        score_models = np.array([])
        moment_vec_eachsplit_models=[]
        moment_vec_models=[]
        N_split=0
        if self.cvtype is not 'InSample':
            for train_index, test_index in self.cvsplit.split(self.Data_All, **self.cvhypara):
                Data_train, Data_test = self.Data_All[train_index], self.Data_All[test_index]
                score_models_eachsplit=np.array([])
                for model in self.models:
                    score_model, moment_vec = self.score(Data_train, Data_test , model)
                    score_models_eachsplit = np.append(score_models_eachsplit,score_model)
                    moment_vec_eachsplit_models.append(moment_vec)
                score_models = np.append( score_models, score_models_eachsplit )
                N_split = N_split+1
            self.moment_vec_eachsplit_models = moment_vec_eachsplit_models
            
            
            #Note: output of moment vector not complete            
            '''
            temp=0
            for i,j in product(range(self.N_models), range(N_split) ):
                temp=temp+moment_vec_eachsplit_models[i+j*(N_split-1)]
            ''' 
                
            self.score_models_all = score_models.reshape([-1, self.N_models]).T
            self.score_models = np.mean(self.score_models_all,axis=1)        
            self.score_std = np.std(score_models,axis=0)
        if self.cvtype is 'InSample':
            for model in self.models:
                score_model, moment_vec = self.score(self.Data_All, self.Data_All , model)
                score_models = np.append(score_models,score_model)
                moment_vec_models.append(moment_vec)
            self.score_models = score_models                    
            self.moment_vec_models = moment_vec_models
        self.best_model = np.array(self.models)[np.where( np.max(self.score_models)==self.score_models )]
        #Note that best model MAXIMIZE the score. 
        self.best_model_maxscore = np.array(self.models)[np.where( np.max(self.score_models)==self.score_models )]
        self.best_model_minscore = np.array(self.models)[np.where( np.min(self.score_models)==self.score_models )]
        

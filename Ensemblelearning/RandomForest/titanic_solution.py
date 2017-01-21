#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: linjiexin
"""

#载入基分类器的代码
from model import *

#ETL:same procedure to training set and test set
training=pd.read_csv('train.csv', index_col=0)
test=pd.read_csv('test.csv', index_col=0)
#将性别转化为０１
SexCode=pd.DataFrame([1,0], index=['female','male'],columns=['Sexcode'])
training=training.join(SexCode, how='left',on=training.Sex)
#删去几个不参与建模的变量，包括姓名、船票号，船舱号
training=training.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'Sex'], axis=1)
test=test.join(SexCode, how='left', on=test.Sex)
test=test.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'Sex'],axis=1)
print('ETL IS DONE!')


#MODEL FITTING
#===============PARAMETER AJUSTMENT============
min_leaf=1
min_dec_gini=0.0001
n_trees=5
n_fea=int(math.sqrt(len(training.columns)-1))
#==============================================

'''
BEST SCORE:0.83
min_leaf=30
min_dec_gini=0.001
n_trees=20
'''

#ESSEMBLE BY RANDOM FOREST
FOREST={}
tmp=list(training.columns)
tmp.pop(tmp.index('Survived'))
feaList=pd.Series(tmp)
for t in range(n_trees):
    #fea=[]
    feasample=feaList.sample(n=n_fea, replace=False)#select feature
    fea=feasample.tolist()
    fea.append('Survived')
    #feaNew=fea.append(target)
    #generate the dataset with replacement
    subset=training.sample(n=len(training), replace=True)
    subset=subset[fea]
    #print(str(t)+' Classifier built on feature:')
    #print(list(fea))
    FOREST[t]=tree_grow(subset, 'Survived', min_leaf, min_dec_gini) #save the tree


#MODEL PREDICTION
#======================
currentdata=training
output='submission_rf_20151116_30_0.001_20'
#======================

prediction={}
for r in currentdata.index:#a row
    prediction_vote={1:0, 0:0}
    row=currentdata.get(currentdata.index==r)
    for n in range(n_trees):
        tree_dict=FOREST[n] #a tree
        p=model_prediction(tree_dict, row)
        prediction_vote[p]+=1
    vote=pd.Series(prediction_vote)
    prediction[r]=list(vote.order(ascending=False).index)[0]#the vote result
result=pd.Series(prediction, name='Survived_p')
#del prediction_vote
#del prediction


#result.to_csv(output)


t=training.join(result,how='left')
accuracy=round(len(t[t['Survived']==t['Survived_p']])/len(t), 5)
print(accuracy)


import sys
# # give preference to local update
sys.path.append("~/.local/lib/python3.4/site-packages")

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
import numpy.random as rn
from scipy import sparse
from copy import copy

def mce(f,fhat):
   return np.mean( f!=fhat )


beer = pd.read_csv('data/beer.csv')
print(list(beer))
yb = beer["brand"].values
Xb = beer.drop("brand",axis=1).values

def EBF(X,y,Xtest,ytest,mslpre=3000,nblock=5,pretree=True,ntree=100,msl=100):
    print(X.shape)
    if pretree: 
        dt = tree.DecisionTreeClassifier(min_samples_leaf=mslpre)
        dt.fit(X,y)
        bvec = [str(a) for a in dt.tree_.apply(X.astype(tree._tree.DTYPE))]
        print("%d blocks" % sum(dt.tree_.feature < 0))
    else:
        bvec = [str(a) for a in rn.random_integers(0,nblock-1,X.shape[0])]
        print("%d obs in train" % X.shape[0], end=" ")
    bset = set(bvec)
    forest = {}
    for b in bset:
        print(b, end=" ")
        forest[b] = ensemble.RandomForestClassifier(
                        ntree,min_samples_leaf=msl,n_jobs=4,bootstrap=2)
        isb = np.array([bi==b for bi in bvec])
        forest[b].fit(X[isb,:],y[isb])
    if pretree:
        yhat = np.empty(Xtest.shape[0], dtype=object)
        btest = [str(a) for a in dt.tree_.apply(Xtest.astype(tree._tree.DTYPE))]
        bset = set(btest)
        for b in bset:
            print(b, end=" ")
            isb = np.array([bi==b for bi in btest])
            yhat[isb] = forest[b].predict(Xtest[isb,:])
    else:
        sampforest = copy(forest['0'])
        sampforest.estimators_ = []
        for fk in forest:
            sampforest.estimators_ += forest[fk].estimators_
        sampforest.n_estimators = len(sampforest.estimators_)
        yhat = sampforest.predict(Xtest)
    
    print(yhat)     
    err = mce(ytest,yhat)
    print(err)
    return err



MC = {key: [] for key in ['DT','EBF','SSF','BF']}
from sklearn.cross_validation import KFold
kb = KFold(len(yb), n_folds=10,shuffle=True,random_state=5800)

k = 0
for train, test in kb:
    print(k)

    Xtrain = Xb[train,:]
    Xtest = Xb[test,:]
    y = yb[train]
    f = yb[test]
    
    MC['BF'] += [EBF(Xtrain,y,Xtest,f,pretree=False,nblock=1)]
    MC['EBF'] += [EBF(Xtrain,y,Xtest,f,mslpre=10000)]
    MC['DT'] += [EBF(Xtrain,y,Xtest,f,mslpre=10000,msl=10000)]
    MC['SSF'] += [EBF(Xtrain,y,Xtest,f,pretree=False)]
    

    k+=1


DF = pd.DataFrame(MC)
me = DF.mean()
me.sort()
pe = [round((e - me[0])/me[0] * 100,1) for e in me]

E = pd.DataFrame({'MC': [round(e,4) for e in me.values], 'WTB':pe})
E.index = me.index
print(E)
#          MC   WTB
# BF   0.4341   0.0
# EBF  0.4531   4.4
# SSF  0.5989  38.0
# DT   0.6979  60.8

mods = ['DT','SSF','BF','EBF']
DF = DF.reindex_axis(mods, axis=1)

fig = plt.figure(figsize=(3.3,3.3))
bp = plt.boxplot(DF.values, sym='r.', labels=mods, patch_artist=True)
for b in bp["boxes"]:
	b.set_facecolor("lightgrey")

plt.setp(bp['whiskers'], color='black')
plt.ylabel("missclass rate", fontsize=15)
fig.savefig("graphs/beer.pdf", format="pdf", bbox_inches="tight")

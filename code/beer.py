import sys
# # give preference to local update
# sys.path.append("~/.local/lib/python3.4/site-packages")

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
import numpy.random as rn
from scipy import sparse

def mce(f,fhat):
   return np.mean( f!=fhat )

def EBF(x,y,xtest,f,k=None,mslpre=3000,nblock=5,pretree=True,ntree=100):
    if pretree: 
        dt = tree.DecisionTreeClassifier(min_samples_leaf=mslpre)
        dt.fit(x,y)
        print("fit done")
        bvec = dt.tree_.apply(x.astype(tree._tree.DTYPE))
        print(sum(dt.tree_.feature < 0))
    else:
        bvec = rn.random_integers(0,nblock-1,x.shape[0])
        print("%d obs in train" % x.shape[0], end=" ")
    bset = set(bvec)
    print(bset)
    forest = {}
    for b in bset:
        print(b, end=" ")
        forest[b] = ensemble.RandomForestClassifier(
                        ntree,min_samples_leaf=100,n_jobs=4)
        isb = (bvec==b)
        forest[b].fit(x[isb,:],y[isb])
    
    if pretree:
        yhat = np.empty(test.shape[0])
        btest = dt.tree_.apply(xtest.astype(tree._tree.DTYPE))
        for b in bset:
            print(b, end=" ")
            isb = btest==b
            yhat[isb] = forest[b].predict(xtest[isb,:])
            print(yhat)
    else:
        yhat = np.zeros(xtest.shape[0])
        for b in bset:
            print(b, end=" ")
            yhat += forest[b].predict(xtest)/float(len(bset))
    
    print(yhat)     
    err = mce(f,yhat)
    print(err)
    return err

beer = pd.read_csv('data/beer.csv')
print(list(beer))
yb = beer["brand"].values
Xb = sparse.csr_matrix(beer.drop("brand",axis=1).values)

MC = {key: [] for key in ['EBF','SSF','BF']}
from sklearn.cross_validation import KFold
kb = KFold(len(yb), n_folds=10,shuffle=True,random_state=5800)

k = 0
for train, test in kb:
    print(k)

    Xtrain = Xb[train,:]
    Xtest = Xb[test,:]
    y = yb[train]
    f = yb[test]
    
    MC['EBF'] += [EBF(Xtrain,y,Xtest,f,k=k,mslpre=10000)]
    MC['SSF'] += [EBF(Xtrain,y,Xtest,f,k=k,pretree=False)]
    MC['BF'] += [EBF(Xtrain,y,Xtest,f,k=k,pretree=False,nblock=1)]

    k+=1


DF = pd.DataFrame(MC)
me = DF.mean()
me.sort()
pe = [round((e - me[0])/me[0] * 100,1) for e in me]

E = pd.DataFrame({'MC': [round(e,4) for e in me.values], 'WTB':pe})
E.index = me.index
print(E)
#        RSME   WTB
# BF   0.6058   0.0
# EBF  0.6117   1.0
# SSF  0.6712  10.8

mods = ['BF','EBF','SSF']
DF = DF.reindex_axis(mods, axis=1)

fig = plt.figure(figsize=(3.3,3.3))
bp = plt.boxplot(DF.values, sym='r.', labels=mods, patch_artist=True)
for b in bp["boxes"]:
	b.set_facecolor("lightgrey")

plt.setp(bp['whiskers'], color='black')
plt.ylabel("missclass rate", fontsize=15)
fig.savefig("graphs/beer.pdf", format="pdf", bbox_inches="tight")

import sys
# give preference to local update
sys.path.append("~/.local/lib/python3.4/site-packages")

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
import numpy.random as rn

def rmse(f,fhat):
   return np.sqrt(np.mean( (f-fhat)**2 ))

def EBF(x,y,test,f,k=None,mslpre=3000,nblock=5,pretree=True,ntree=100):
    if pretree: 
        dt = tree.DecisionTreeRegressor(min_samples_leaf=mslpre)
        dt.fit(x,y)
        bvec = dt.tree_.apply(x.astype(tree._tree.DTYPE))
        print("%d leaves" % sum(dt.tree_.feature < 0), end=" ")
    else:
        bvec = rn.random_integers(0,nblock-1,x.shape[0])
        print("%d obs in train" % x.shape[0], end=" ")
    bset = set(bvec)
    forest = {}
    for b in bset:
        print(b, end=" ")
        forest[b] = ensemble.RandomForestRegressor(
                        ntree,bootstrap=1,min_samples_leaf=3,n_jobs=4)
        isb = bvec==b
        forest[b].fit(x[isb,:],y[isb])
    
    if pretree:
        yhat = np.empty(test.shape[0])
        btest = dt.tree_.apply(test.astype(tree._tree.DTYPE))
        for b in bset:
            print(b, end=" ")
            isb = btest==b
            yhat[isb] = forest[b].predict(test[isb,:])
    else:
        yhat = np.zeros(test.shape[0])
        for b in bset:
            print(b, end=" ")
            yhat += forest[b].predict(test)/float(len(bset))
            
    err = rmse(f,yhat)
    print(err)
    return err

wine = pd.read_csv('data/wine.csv')
print(list(wine))
yw = wine["quality"] 
Xw = wine.drop("quality",axis=1)
Xw["color"] = (Xw["color"].values == "red").astype("int")
wine[wine.color == "white"]["quality"]
wine_RMSE = {key: [] for key in ['EBF','SSF','BF']}
from sklearn.cross_validation import KFold
kw = KFold(len(yw), n_folds=10,shuffle=True,random_state=5807)

k = 0
for train, test in kw:
    print(k)

    Xtrain = Xw.iloc[train].values
    Xtest = Xw.iloc[test].values
    y = yw[train]
    f = yw[test]
    
    wine_RMSE['EBF'] += [EBF(Xtrain,y,Xtest,f,k=k,mslpre=1000)]
    wine_RMSE['SSF'] += [EBF(Xtrain,y,Xtest,f,k=k,pretree=False)]
    wine_RMSE['BF'] += [EBF(Xtrain,y,Xtest,f,k=k,pretree=False,nblock=1)]

    k+=1


DF = pd.DataFrame(wine_RMSE)
me = DF.mean()
me.sort()
pe = [round((e - me[0])/me[0] * 100,1) for e in me]

E = pd.DataFrame({'RSME': [round(e,4) for e in me.values], 'WTB':pe})
E.index = me.index
E
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
plt.ylabel("RMSE", fontsize=15)
fig.savefig("graphs/wine.pdf", format="pdf", bbox_inches="tight")

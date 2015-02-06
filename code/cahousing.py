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


cah = pd.read_csv('data/CAhousing.csv')
print(list(cah))
cah["medianIncome"] = cah["medianIncome"]*1e4
Xh = cah.drop("medianHouseValue",axis=1)
yh = cah["medianHouseValue"] 

fig = plt.figure(figsize=(5,5))
plt.hist(cah["medianHouseValue"]/100000,color="lightgrey", normed=1)
plt.xlabel("median home value in $100k", fontsize=12)
plt.ylabel("density", fontsize=12)
#fig.savefig("graphs/ca_hist.pdf", format="pdf", bbox_inches="tight")

from sklearn.cross_validation import KFold
kf = KFold(len(yh), n_folds=10,shuffle=True,random_state=5807)
## folds output for bart in R
import os
os.makedirs("data/cafolds", exist_ok=True)
k=0
for train, test in kf:
    np.savetxt("data/cafolds/%d.txt"%k,test,fmt='%d')
    k+=1


## run our loop
ne = 100
msl = 2
cah_RMSE = {key: [] for key in ['DT','RF','ET','BF']}
k=0
for train, test in kf:
    ### data
    Xtrain = Xh.iloc[train].values
    Xtest = Xh.iloc[test].values
    ytrain = yh[train]
    ytest = yh[test]
    #DT
    cah_dt = tree.DecisionTreeRegressor(min_samples_leaf=msl)
    cah_dt.fit(Xtrain,ytrain)
    cah_dtp = cah_dt.predict(Xtest)
    cah_RMSE['DT'] += [rmse(ytest,cah_dtp)]
    #RF
    cah_rf = ensemble.RandomForestRegressor(n_estimators=ne,min_samples_leaf=msl,n_jobs=4)
    cah_rf.fit(Xtrain,ytrain)
    cah_rfp = cah_rf.predict(Xtest)
    cah_RMSE['RF'] += [rmse(ytest,cah_rfp)]
    #ET
    cah_et = ensemble.ExtraTreesRegressor(n_estimators=ne,min_samples_leaf=msl,n_jobs=4)
    cah_et.fit(Xtrain,ytrain)
    cah_etp = cah_et.predict(Xtest)
    cah_RMSE['ET'] += [rmse(ytest,cah_etp)]
    #BF
    cah_bf = ensemble.RandomForestRegressor(n_estimators=ne,bootstrap=2, n_jobs=4,
                                           min_weight_fraction_leaf=1e-4 )
    cah_bf.fit(Xtrain,ytrain)
    cah_bfp = cah_bf.predict(Xtest)
    cah_RMSE['BF'] += [rmse(ytest,cah_bfp)]
    print(k, end=" ")
    k+=1

RMSE = pd.DataFrame(cah_RMSE)

## cobine with output from code/bart.R, since this is not implemented in python
rdata = pd.read_table("graphs/bartca.txt", sep=" ")
rdata.columns = ["BART","BCART"]

DF = pd.concat([RMSE,rdata], axis=1)
mods = ['DT','RF','BF','ET','BART','BCART']
DF = DF.reindex_axis(mods, axis=1)
DF.mean().sort(inplace=False)
# BF       48346.634773
# RF       48523.256219
# ET       52568.428925
# BART     54754.587034
# DT       65385.540888
# BCART    82695.262723

fig = plt.figure(figsize=(6,3.3))
bp = plt.boxplot(DF.values, sym='r.', labels=mods, patch_artist=True)
for b in bp["boxes"]:
	b.set_facecolor("lightgrey")

plt.ylim(40000,90000)
plt.setp(bp['whiskers'], color='black')
plt.ylabel("RMSE", fontsize=15)
#fig.savefig("graphs/ca_rmse.pdf", format="pdf", bbox_inches="tight")


#### EBF
ca_trunk = tree.DecisionTreeRegressor(min_samples_leaf=3500)
ca_trunk.fit(Xh,yh)

## plot it and show inline
#tree.export_graphviz(ca_trunk,feature_names=list(Xh),out_file="graphs/catrunk.dot")
## a bunch of unescessary formatting (default is fine, but this is nicer)
# !sed -i  's/mse = [0-9]*\.[0-9]*\\n/ /g' graphs/catrunk.dot
# !sed -i  's/samples/size/g' graphs/catrunk.dot
# !sed -i  's/\[ /\$/g' graphs/catrunk.dot
# !sed -i  's/\.\]//g' graphs/catrunk.dot
# !sed -i  's/value/mean/g' graphs/catrunk.dot

# !dot -Tpdf graphs/catrunk.dot -o graphs/catrunk.pdf
# note that 34.5 degrees lat is just north of santa barbara

cah_forest = ensemble.RandomForestRegressor(n_estimators=100, bootstrap=2, n_jobs=4,
                                           min_samples_leaf=3500)
cah_forest.fit(Xh,yh)
trees = cah_forest.estimators_
var = list(Xh)

loc0 = [ t.tree_.threshold[0]/10000 for t in trees ]
loc1 = [ t.tree_.threshold[1]/10000 for t in trees ]

fig = plt.figure(figsize=(5,5))
xlim = [min(Xh['medianIncome'])/10000,max(Xh['medianIncome'])/10000]
plt.hist(Xh['medianIncome']/10000,normed=True,
	color="lightgrey",label="full sample", linewidth=0, log=True)
plt.hist(loc0,color="darkorange",normed=True,
	label="first split", linewidth=0, log=True, bins=3)
plt.hist(loc1,color="darkturquoise",normed=True,
	label="second split", linewidth=0, log=True, bins=3)
plt.xlim(xlim)
plt.legend(frameon=False, loc='upper right')
plt.xlabel("median income in $10k")
plt.ylabel("log density")
#fig.savefig("graphs/ca_splits.pdf", format="pdf", bbox_inches="tight")


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

sube = {key: [] for key in ['EBF','SSF']}
k = 0
for train, test in kf:
    print(k)
    Xtrain = Xh.iloc[train].values
    Xtest = Xh.iloc[test].values
    y = yh[train]
    f = yh[test] 
    sube['EBF'] += [EBF(Xtrain,y,Xtest,f,k=k)]
    sube['SSF'] += [EBF(Xtrain,y,Xtest,f,k=k,pretree=False)]
    k+=1


sube = pd.DataFrame(sube)
DF = pd.concat([RMSE,rdata,sube], axis=1)
me = DF.mean()
me.sort()
pe = [round((e - me[0])/me[0] * 100,1) for e in me]

E = pd.DataFrame({'RSME': [round(e) for e in me.values], 'WTB':pe})
E.index = me.index
print(E)
#     RSME      WTB
# 0  48347        0
# 1  48523    17662
# 2  49352   100583
# 3  52568   422179
# 4  53213   486643
# 5  54755   640795
# 6  65386  1703891
# 7  82695  3434863



DF.drop("BCART",1)
mods = ['DT','BART','ET','RF','BF','EBF','SSF']
DF = DF.reindex_axis(mods, axis=1)

fig = plt.figure(figsize=(6,3.3))
bp = plt.boxplot(DF.values, sym='r.', labels=mods, patch_artist=True)
for b in bp["boxes"]:
	b.set_facecolor("lightgrey")

#plt.ylim(40000,90000)
plt.setp(bp['whiskers'], color='black')
plt.ylabel("RMSE", fontsize=15)
#fig.savefig("graphs/ca_rmse.pdf", format="pdf", bbox_inches="tight")

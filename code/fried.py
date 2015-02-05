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

def fried(x, noisy=True):
	f = 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.5)**2 + 10*x[:,3]+5*x[:,4]
	if noisy: 
		f += rn.normal(size=n)
	return f


def rmse(f,fhat):
   return np.sqrt(np.mean( (f-fhat)**2 ))

ne = 100
n = 100
p = 10
B = 100
msl = 3
fried_RMSE = {key: [] for key in ['DT','RF','ET','BF']}
for b in range(B):
    ### data
    Xtrain = rn.uniform(0,1,n*p).reshape(n,p)
    Xtest = rn.uniform(0,1,1000*p).reshape(1000,p)
    y = fried(Xtrain)
    f = fried(Xtest,noisy=False)
    
    #DT
    fried_dt = tree.DecisionTreeRegressor(min_samples_leaf=msl)
    fried_dt.fit(Xtrain,y)
    fried_dtp = fried_dt.predict(Xtest)
    fried_RMSE['DT'] += [rmse(f,fried_dtp)]
    
    #RF
    fried_rf = ensemble.RandomForestRegressor(n_estimators=ne,min_samples_leaf=msl)
    fried_rf.fit(Xtrain,y)
    fried_rfp = fried_rf.predict(Xtest)
    fried_RMSE['RF'] += [rmse(f,fried_rfp)]

    #ET
    fried_et = ensemble.ExtraTreesRegressor(n_estimators=ne,min_samples_leaf=msl)
    fried_et.fit(Xtrain,y)
    fried_etp = fried_et.predict(Xtest)
    fried_RMSE['ET'] += [rmse(f,fried_etp)]

    #BF
    fried_bf = ensemble.RandomForestRegressor(n_estimators=ne,bootstrap=2,min_samples_leaf=msl)
    fried_bf.fit(Xtrain,y)
    fried_bfp = fried_bf.predict(Xtest)
    fried_RMSE['BF'] += [rmse(f,fried_bfp)]

    print(b,end=" ")


RMSE = pd.DataFrame(fried_RMSE)

## cobine with output from code/bart.R, since this is not implemented in python
rdata = pd.read_table("graphs/bartfried.txt", sep=" ")
rdata.columns = ["BART","BCART"]

DF = pd.concat([RMSE,rdata], axis=1)
mods = ['DT','BCART','RF','BF','ET','BART']
DF = DF.reindex_axis(mods, axis=1)
DF.mean().sort(inplace=False)
# BART     1.811829
# ET       2.624876
# BF       2.666475
# RF       2.741743
# DT       3.690980
# BCART    3.910470

fig = plt.figure(figsize=(6,3.3))
bp = plt.boxplot(DF.values, sym='r.', labels=mods, patch_artist=True)
for b in bp["boxes"]:
	b.set_facecolor("lightgrey")

plt.setp(bp['whiskers'], color='black')
plt.ylabel("RMSE", fontsize=15)
fig.savefig("graphs/fried.pdf", format="pdf", bbox_inches="tight")
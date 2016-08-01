library(BayesTree)
library(MASS)
data(mcycle)
library(tgp)

plot(mcycle, pch=21, cex=.75, bg=8)
xtest <- seq(0,60,length=400)
bartFit = bart(mcycle[,1],mcycle[,2],xtest,ntree=100,ndpost=200)
bcartFit = bcart(mcycle[,1],mcycle[,2],xtest,tree=c(0.99,.1,2),basemax=1)
plot(bcartFit)
plot(bartFit)

plot(mcycle, pch=21, cex=.75, bg=8)
lines(xtest, bartFit$yhat.test.mean, col="darkorange", lwd=2)
sig <- mean(bartFit$sigma)
barthi <- apply(bartFit$yhat.test,2,quantile,.95) + 2*sig
bartlo <- apply(bartFit$yhat.test,2,quantile,.05) - 2*sig

lines(xtest, )
cah = read.csv("data/CAhousing.csv")
y = cah["medianHouseValue"][,1]
X = cah[,1:8]
X["medianIncome"] = X["medianIncome"]*1e4

hrmse = list(bart=c(), bcart=c())
for(k in 0:9){
	test = read.table(sprintf("data/cafolds/%d.txt",k))[,1]
	xtrain = X[-test,]
	xtest = X[test,]
	ytrain = y[-test]
	ytest = y[test]
	bartFit = bart(xtrain,ytrain,xtest,ntree=100,ndpost=200)
	bcartFit = bcart(xtrain,ytrain,xtest,tree=c(0.99,.1,2),basemax=1)
	bfhat = bartFit$yhat.test.mean
	bcfhat = bcartFit$ZZ.mean
	print(system.time(hrmse[['bart']] <- c(hrmse[['bart']], sqrt(mean( (bfhat-ytest)^2 )))))
	print(system.time(hrmse[['bcart']] <- c(hrmse[['bcart']], sqrt(mean( (bcfhat-ytest)^2 )))))
	cat(k,": ",hrmse[["bart"]][k+1],hrmse[["bcart"]][k+1],"\n")
}
hrmse= as.data.frame(hrmse)
lapply(hrmse,mean)
write.table(hrmse, "graphs/bartca.txt", row.names=FALSE)

###########################

##simulate data (example from Friedman MARS paper)
fried = function(x){
10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
}

n = 100     
p = 10
B = 100
 
frmse = list(bart=c(), bcart=c())

for(b in 1:B){
	print(b)
	xtrain = matrix(runif(n*p),n,p) 
	xtest = matrix(runif(1000*p),1000,p) 
	y = fried(xtrain) + rnorm(n)
	f = fried(xtest)
	bartFit = bart(xtrain,y,xtest,ntree=100,ndpost=200)
	bcartFit = bcart(xtrain,y,xtest,tree=c(0.99,.1,2),basemax=1)
	bfhat = bartFit$yhat.test.mean
	bcfhat = bcartFit$ZZ.mean
	frmse[['bart']] <- c(frmse[['bart']], sqrt(mean( (bfhat-f)^2 )))
	frmse[['bcart']] <- c(frmse[['bcart']], sqrt(mean( (bcfhat-f)^2 )))
}
frmse= as.data.frame(frmse)
lapply(frmse,mean)
write.table(frmse, "graphs/bartfried.txt", row.names=FALSE)

library(BayesTree)

cah = read.csv("data/CAhousing.csv")
y = cah["medianHouseValue"][,1]
X = cah[,1:8]
X["medianIncome"] = X["medianIncome"]*1e4

hrmse <- c()
for(k in 0:9){
	test = read.table(sprintf("data/cafolds/%d.txt",k))[,1]
	xtrain = X[-test,]
	xtest = X[test,]
	ytrain = y[-test]
	ytest = y[test]
	print(system.time(bartFit <- bart(xtrain,ytrain,xtest,ntree=100,ndpost=100)))
	hrmse <- c(hrmse, 
		sqrt(mean( (bartFit$yhat.test.mean-ytest)^2 )))
	cat(k,": ",hrmse[k+1],"\n")
}

write.table(hrmse, "graphs/bartca.txt", row.names=FALSE, col.names=FALSE)

###########################

# ##simulate data (example from Friedman MARS paper)
# fried = function(x){
# 10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
# }

# n = 100     
# p = 10
# B = 100
 
# rmse = c()

# for(b in 1:B){
# 	print(b)
# 	xtrain = matrix(runif(n*p),n,p) 
# 	xtest = matrix(runif(1000*p),1000,p) 
# 	y = fried(xtrain) + rnorm(n)
# 	f = fried(xtest)
# 	bartFit = bart(xtrain,y,xtest,ntree=100)
# 	fhat = bartFit$yhat.test.mean
# 	rmse <- c(rmse, sqrt(mean( (fhat-f)^2 )))
# }
# mean(rmse)
# # 1.7


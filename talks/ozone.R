data(airquality)
n <- nrow(airquality)
B <- 2000
beta <- vector(length=B)
for(b in 1:B){
	fit <- lm(Ozone ~., data=airquality, weights=rexp(n))
	beta[b] <- coef(fit)["Wind"]
}
hist(beta, col=8, main="", 
	xlab="Coefficient for Ozone on Wind", 
	freq=FALSE,ylim=c(0,0.6),breaks=25)
sampfit <- summary(lm(Ozone ~ ., data=airquality))$coef["Wind",1:2]
grid <- seq(-6,5,length=500)
lines(grid, dnorm(grid,sampfit[1],sampfit[2]),col=2,lwd=2)
legend("topleft",col=c(8,2),lwd=4, 
	legend=c("Bayes nonparametric posterior",
		"theoretical sampling distribution"),bty="n")

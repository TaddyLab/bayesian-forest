data(airquality)
n <- nrow(airquality)
B <- 1000
beta <- vector(length=B)
for(b in 1:B){
  fit <- lm(Ozone ~., data=airquality, weights=rexp(n))
  beta[b] <- coef(fit)["Wind"]
}

sampfit <- lm(Ozone ~ ., data=airquality)
coef <- summary(sampfit)$coef["Wind",1:2]

x <- as.matrix(cbind(1,na.omit(airquality)[,-1]))
xxi <- solve(crossprod(x))
sandwich <- xxi%*%t(x)%*%diag(sampfit$resid^2)%*%x%*%xxi

hist(beta, col=8, main="", 
  xlab="Coefficient for Ozone on Wind", 
  freq=FALSE,ylim=c(0,0.6),breaks=25)
grid <- seq(-6,5,length=500)
lines(grid, dnorm(grid,coef[1],coef[2]),col=2,lwd=2)
lines(grid, dnorm(grid,coef[1],sqrt(sandwich[3,3])),col=4,lwd=2)
legend("topleft",col=c(8,4,2),lwd=4, 
  legend=c("bootstrap BNP posterior",
  		   "normal approx BNP posterior",
           "standard sampling distribution"),bty="n")

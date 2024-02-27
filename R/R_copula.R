library(copula)
set.seed(100)
myCop <- normalCopula(param=c(0.4,0.2,-0.8), dim = 3, dispstr = "un")
myMvd <- mvdc(copula=myCop, margins=c("gamma", "beta", "t"),
              paramMargins=list(list(shape=2, scale=1),
                                list(shape1=2, shape2=2), 
                                list(df=5)) )

Z2 <- rmvdc(myMvd, 2000)
colnames(Z2) <- c("x1", "x2", "x3")
pairs.panels(Z2)

asCall <- function(fun, param)
{
  cc <-
    if (length(param) == 0)
      quote(FUN(x))
  else if(is.list(param)) {
    as.call(c(quote(FUN), c(quote(x), as.expression(param))))
  } else { ## assume that [dpq]<distrib>(x, param) will work
    as.call(c(quote(FUN), c(quote(x), substitute(param))))
  }
  cc[[1]] <- as.name(fun)
  cc
}

#https://github.com/cran/copula/blob/master/R/mvdc.R

rMvdc <- function (n, mvdc) 
{
  dim <- dim(mvdc@copula)
  u <- rCopula(n, mvdc@copula)
  x <- u
  for (i in 1:dim) {
    qdf.expr <- asCall(paste0("q", mvdc@margins[i]), 
                       mvdc@paramMargins[[i]])
    x[, i] <- eval(qdf.expr, list(x = u[, i]))
  }
  x
}

cl <- rCopula(2000, normalCopula(0.9, dim = 2))
plot(cl[,1], cl[,2])

copula <- normalCopula(0.9, dim = 2)

rnormalCopula <- function(n, copula, ...) {
  pnorm(rmvnorm(n, sigma = getSigma(copula)))
}

copula <- normalCopula(0.99, dim = 2)

cl <- rnormalCopula(2000, copula)
plot(cl[,1], cl[,2]) 

#https://github.com/cran/copula/blob/master/R/frankCopula.R

rfrankBivCopula <- function(n, alpha) {
  U <- runif(n); V <- runif(n)
  ## FIXME : |alpha| << 1  (including alpha == 0)
  ## to fix numerical rounding problems for alpha >35 but not for alpha < -35 :
  a <- -abs(alpha)
  ## reference: Joe (1997, p.147)
  V <- -1/a * log1p(-V * expm1(-a) / (exp(-a * U) * (V - 1) - V))
  cbind(U, if(alpha > 0) 1 - V else V,  deparse.level=0L)
}

samples <- rfrankBivCopula(1000, 50)
plot(samples[,1], samples[,2])

# https://github.com/cran/copula/blob/master/R/claytonCopula.R
rclaytonBivCopula <- function(n, alpha) {
  val <- cbind(runif(n), runif(n))
  ## This implementation is confirmed by Splus module finmetrics
  val[,2] <- (val[,1]^(-alpha) * (val[,2]^(-alpha/(alpha + 1)) - 1) + 1)^(-1/alpha)
  ## Frees and Valdez (1998, p.11): wrong! Can be checked by sample Kendall' tau.
  ## Their general expression for $F_k$ is all right for $k > 2$.
  ## But for dimension 2, they are missing $\phi'$ in the numerator and
  ## the denominator should be 1. So their formula for $U_2$ on p.11 is incorrect.
  val
}

samples <- rclaytonBivCopula(1000, 5)
plot(samples[,1], samples[,2])

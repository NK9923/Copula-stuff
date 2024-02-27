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



# ---- Inverse

inverse_pareto <- function(p, b, k) {
  if (any(p <= 0) || any(p >= 1) || k <= 0) {
    stop("Invalid parameters for inverse Pareto function.")
  }
  
  return(b / (1 - p)^(1/k))
}

# Example usage
p_values <- seq(0.01, 0.99, by = 0.01)
b_value <- 1.0
k_value <- 2.0

quantiles <- inverse_pareto(0.05, b_value, k_value)

qPareto(0.05, b_value, k_value)


# Function to generate random numbers from Pareto distribution
rpareto_custom <- function(n, b, k) {
  if (any(c(n, b, k) <= 0)) {
    stop("Invalid parameters for Pareto distribution.")
  }
  
  u <- runif(n)  # Generate n uniform random numbers
  x <- b / (u^(1/k))  # Inverse transform sampling
  
  return(x)
}

# Example usage
n <- 1000
b_value <- 1.0
k_value <- 2.0

pareto_samples <- rpareto_custom(n, b_value, k_value)

# Plot histogram of generated samples
hist(pareto_samples, breaks = 300, col = "lightblue", main = "Pareto Distribution")




# Function to calculate the Beta probability density function (PDF)
pdf <- function(x, shape1, shape2) {
  if (any(x < 0) || any(x > 1) || shape1 <= 0 || shape2 <= 0) {
    stop("Invalid parameters for Beta distribution.")
  }
  
  beta.1 <- function(alpha, beta) {
    (gamma(alpha) * gamma(beta))/gamma(alpha + beta)
  }
  
  # Beta PDF
  pdf <- x^(shape1 - 1) * (1 - x)^(shape2 - 1) / beta.1(shape1, shape2)
  
  return(pdf)
}

# Function to calculate the Beta cumulative distribution function (CDF)
cdf <- function(x, shape1, shape2) {
  if (any(x < 0) || any(x > 1) || shape1 <= 0 || shape2 <= 0) {
    stop("Invalid parameters for Beta distribution.")
  }
  
  beta.1 <- function(alpha, beta) {
    (gamma(alpha) * gamma(beta))/gamma(alpha + beta)
  }
  
  # Beta CDF (using numerical integration)
  cdf <- sapply(x, function(x_i) {
    integrate(function(t) t^(shape1 - 1) * (1 - t)^(shape2 - 1), lower = 0, upper = x_i)$value
  })
  
  beta <- beta.1(shape1, shape2)
  
  return(cdf/beta)
}

cdf(0.05, shape1_value, shape2_value)
pbeta(0.05, shape1_value, shape2_value)


# Function to calculate quantiles for the Beta distribution (numerical inversion)
qbeta_custom <- function(p, shape1, shape2) {
  if (any(p < 0) || any(p > 1) || shape1 <= 0 || shape2 <= 0) {
    stop("Invalid parameters for Beta distribution.")
  }
  
  # Inverse of the normalized Beta CDF (numerical inversion)
  find_quantile <- function(prob, a, b) {
    lower <- 0
    upper <- 1
    
    while (upper - lower > 1e-10) {
      mid <- (lower + upper) / 2
      if (cdf(mid, a, b) < prob) {
        lower <- mid
      } else {
        upper <- mid
      }
    }
    
    return(mid)
  }
  
  # Calculate quantiles using numerical inversion
  quantiles <- sapply(p, function(prob) find_quantile(prob, shape1, shape2))
  
  return(quantiles)
}

# Example usage
p_values <- seq(0.01, 0.99, by = 0.01)
shape1_value <- 2.0
shape2_value <- 5.0

# Calculate PDF, CDF, and quantiles
beta_pdf <- dbeta_custom(p_values, shape1_value, shape2_value)
beta_cdf <- pbeta_custom(p_values, shape1_value, shape2_value)
beta_quantiles <- qbeta_custom(p_values, shape1_value, shape2_value)
qbeta_custom(0.05, shape1_value, shape2_value)

# Print the results
print(data.frame(p = p_values, pdf = beta_pdf, cdf = beta_cdf, quantile = beta_quantiles))




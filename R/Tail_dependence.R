#### Tail dependency
# Using Pedro's method.


### Functions ----
## GPD fit
f_FitGPD <- function( x, lower = NULL, min.obs = 150 ) {
  x <- as.matrix(x)
  
  if ( is.null(lower) ) {
    lower <- ecdf(x)(head(sort(x), min.obs)[min.obs])
  }
  
  lower_quant <- quantile(x, lower)
  excess <- x[x <= lower_quant]*(-1) - lower_quant*(-1)
  if ( !all(excess >= 0) ) { stop("Support of the GPD is >= 0!") }
  
  gpd_fit <- qrmtools::fit_GPD_MLE(x = excess, estimate.cov = F)
  #fit.MOM <- qrmtools::fit_GPD_MOM(excess)
  
  res <- list(
    "Excess" = sort(excess),
    "Shape" = gpd_fit$par[["shape"]],
    "Scale" = gpd_fit$par[["scale"]],
    "Threshold" = lower_quant
  )
  
  return(res)
}

f_FitGPDList <- function( x, lower = NULL, min.obs = 150 ) {
  x <- as.matrix(x)
  tickers <- colnames(x)
  fits <- list(NULL)
  fits <- lapply(1:length(tickers), function(i) {
    loss_dist <- x[, tickers[i]]
    f_FitGPD(x = x, lower = lower, min.obs = min.obs)
  })
  names(fits) <- paste0(colnames(x), "_fit")
  return(fits)
}

f_FastpSPGPD <- function(x, fit) {
  x <- as.matrix(x)
  
  emp.cdf <- sort(x)
  x <- cbind(x, seq_along(x))
  
  shape <- fit$Shape
  scale <- fit$Scale
  u <- fit$Threshold
  n_u <- length(fit$Excess)
  n <- length(emp.cdf)
  x.lower <- x[x[, 1] <= u,]
  most.similar <- sapply(x.lower[, 1], function(x) {
    which.min(abs(emp.cdf-x))
  })
  i <- ifelse(x.lower[, 1] < emp.cdf[most.similar], most.similar - 1, most.similar)
  x.lower[, 1] <- (i - 0.5)/n
  
  x.upper <- x[x[, 1] > u,]
  x.upper[, 1] <- 1 - (n_u/n)*(1 + shape*(x.upper[, 1] - u)/scale)^(-1/shape)
  prob <- rbind(x.lower, x.upper)
  
  prob <- prob[order(prob[,2]), ]
  prob[, 1]
  
  return(prob[, 1])
}

## Empirical copulas
f_CopulasEmp <- function( x, fits) {
  x <- as.matrix(x)
  copula <- matrix( nrow = nrow(x), ncol = ncol(x) )
  names <- colnames(x)
  for ( i in 1:ncol(x) ) {
    fit_i <- fits[[i]]
    ss_i <- x[, i]
    copula[, i] <- f_FastpSPGPD(x = ss_i, fit = fit_i)
  }
  colnames(copula) <- names
  return(copula)
}

## Tail dependency
f_TailDep <- function(x) {
  # GPD fits
  fits <- f_FitGPDList(x)
  # Empirical copulas
  emp_cops <- f_CopulasEmp(x = x, fits = fits)
  # Vine copulas
  vine_cops <- rvinecopulib::vinecop(data = emp_cops, selcrit = "bic", cores = 4)
  rVine_cops <- rvinecopulib::rvinecop(n = 10000, vinecop = vine_cops, qrng = T)
  # tail dependency
  res <- FRAPO::tdc(x = rVine_cops, method = "EmpTC")
  
  return(res)
}

library(data.table)

set.seed(123)
x <- data.table(X = rnorm(1000), Y = rnorm(1000))

x <- data.table(X = rlnorm(1000), Y = rlnorm(1000))
f_TailDep(x)

# ### Running ----
# df_Combi$TailDep <- lapply(l_Synths, function(synth) {
#   f_TailDep(synth[, c("RoRLog_A", "RoRLog_B")])
# })


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

cdf(0.2,1,5)
qbeta_custom(0.1, 1, 5)
pdf(0.2, 1, 5)



dbeta(0.2, 1, 5)
pbeta(0.2,1,5)
qbeta(0.1, 1, 2)


hist(rbeta(1000, 1, 5))

data <- read.csv("C://Users//Nikolaus Kresse//Desktop//Tests//Project1//Project1//random_t_values.csv")
hist(data$X0.832473)

hist(rnorm(5000, 10))




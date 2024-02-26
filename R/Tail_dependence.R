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

library(gamma)

# Definition der pdf der t-Verteilung
dt_t_distribution <- function(x, df) {
  gamma_term <- gamma((df + 1) / 2) / (sqrt(pi * df) * gamma(df / 2))
   <- (1 + (x^2 / df))^(-(df + 1) / 2)
  return(gamma_term * expression_term)
}

# Numerische Integration mit der Trapezregel
trapezoidal_rule <- function(f, a, b, n, df) {
  h <- (b - a) / n
  sum <- 0
  for (i in 1:n) {
    x0 <- a + (i - 1) * h
    x1 <- a + i * h
    sum <- sum + (h / 2) * (f(x0, df) + f(x1, df))
  }
  return(sum)
}

# Berechnung des Quantils der t-Verteilung
quantile_t <- function(p, df, tol = 1e-6, max_iter = 1000) {
  x <- 0
  a <- -1000  # Untere Grenze für die Integration
  for (i in 1:max_iter) {
    pdf <- dt_t_distribution(x, df)
    cdf <- trapezoidal_rule(dt_t_distribution, a, x, 1000, df) / 
      trapezoidal_rule(dt_t_distribution, a, Inf, 1000, df)
    if (abs(cdf - p) < tol) {
      break
    }
    x <- x - (cdf - p) / pdf
  }
  return(x)
}


# Test der Funktion
p <- 0.95
df <- 5
qt(p, df)

pt(2, 5)
dt(2, 5)


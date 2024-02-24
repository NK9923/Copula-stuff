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

f_FastpSPGPD <- function( x, fit) {
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
    copula[, i] <- f_FastpSPGPD(x = ss_i, fit =  fit_i)
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
 <- data.table(X = rnorm(1000), Y = rnorm(1000))
f_TailDep(x)

# ### Running ----
# df_Combi$TailDep <- lapply(l_Synths, function(synth) {
#   f_TailDep(synth[, c("RoRLog_A", "RoRLog_B")])
# })

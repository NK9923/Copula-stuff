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


# ---- bootstrapping

library(data.table)
library(RODBC)
library(smn)

IMF <- as.data.table(sqlQuery(con1, sql_statement, stringsAsFactors=FALSE))
IMF[,returns:=c(1,(exp(diff(log(Value)))))]

#Date      Value   returns
#1: 1996-10-31   72.67000 1.0000000
#2: 1996-11-01   72.87019 1.0027548

tickers<-c("SPXT Index", "Div i14 10%", "IMF")
raw_prices_list<-sql_load_tickers(tickers = tickers, convert_to_EUR = TRUE)

#$`SPXT Index`
#Date     Price
#1: 1996-10-01  709.3367
#2: 1996-10-02  709.5279
#3: 1996-10-03  718.4373
#4: 1996-10-06  720.1127
#5: 1996-10-07  716.1134

merge_two<-function(dtab1, dtab2){
  merge(dtab1, dtab2, by="Date", all=TRUE)
}

spxt_smn<-Reduce(function(...) merge(..., by="Date", all=TRUE), x = raw_prices_list)
names(spxt_smn)<-c("Date", "spxt", "i14", "imf")
#fwrite(x = spxt_smn, file = "U:/WU/WiSe_2223/QRM/spxt_smn.csv")
spxt_i14<-na.omit(spxt_smn[,c("Date", "spxt", "i14")])
r_spxt_i14<-data.table(Date=spxt_i14[,Date], apply(X = spxt_i14[,-1], MARGIN = 2, function(x) c(1,(exp(diff(log(x)))))))
spxt_imf<-na.omit(spxt_smn[,c("Date", "spxt", "imf")])
r_spxt_imf<-data.table(Date=spxt_imf[,Date], apply(X = spxt_imf[,-1], MARGIN = 2, function(x) c(1,(exp(diff(log(x)))))))

summary_table<-function(returns, cor_to=NULL, time_unit_scaling, convert_returns=FALSE){
  dtab<-returns[,-1]
  if(convert_returns){
    dtab<-apply(dtab, MARGIN = 2, FUN = function(x) c(1,(exp(diff(log(x))))))
  }
  asset_names<-names(dtab)
  tot_return<-apply(dtab, MARGIN = 2, prod)-1
  cagr<-(tot_return+1)^(time_unit_scaling/nrow(dtab))-1
  ann_sd<-apply(dtab, MARGIN = 2, sd)*sqrt(time_unit_scaling)
  risk_to_return<-cagr/ann_sd
  max_dd<-apply(dtab, MARGIN = 2, FUN = function(x) f_dd(x-1))
  if(!is.null(cor_to)){
    cor_smn<-apply(dtab, MARGIN = 2, FUN = function(x) cor(x, returns[, cor_to, with=F]))
    sumry<-data.table("Instrument"=asset_names, "CAGR"=round(cagr*100, 2), 
                      "Ann. SD"=round(ann_sd*100, 2), 
                      "Return-to-Risk Ratio"=round(risk_to_return, 2), 
                      "Max DD"=round(max_dd*100, 2), "Korrelation zum SMN i14"=round(cor_smn, 2))
  }
  else{
    sumry<-data.table("Instrument"=asset_names, "Tot. Ret"=tot_return, "CAGR"=round(cagr*100, 2), 
                      "Ann. SD"=round(ann_sd*100, 2), 
                      "Return-to-Risk Ratio"=round(risk_to_return, 2), 
                      "Max DD"=round(max_dd*100, 2))
  }
  #cbind(, tot_return, cagr, ann_sd, max_dd)
  print(sumry)
}

portfolio_backtest<-function(returns, tickers, weights, rebal_freq, start_cap=1){
  if(length(weights)!=length(tickers)){
    stop("There is not a weight assigned to each asset")
  }
  dtab<-returns[ ,c("Date", tickers), with=F]
  year<-format(dtab$Date, "%Y")
  month<-format(dtab$Date, "%m")
  if(rebal_freq=="Yearly"){
    year_s<-shift(year)
    rebal<-year!=year_s
  }
  else if(rebal_freq=="Semesterly"){
    semester<-fifelse(month<=6, 1, 2)
    semester_s<-shift(semester)
    rebal<-semester!=semester_s
  }
  else if(rebal_freq=="Quarterly"){
    quarter<-as.numeric(cut(as.numeric(month), breaks = c(0, 3, 6, 9, 12), labels = c(1, 2, 3, 4)))
    quarter_s<-shift(quarter)
    rebal<-quarter!=quarter_s
  }
  else if(rebal_freq=="Monthly"){
    month_s<-shift(month)
    rebal<-month!=month_s
  }
  else if(rebal_freq=="Daily"){
    rebal_true<-rep(TRUE, nrow(dtab))
  }
  rebal[is.na(rebal)]<-TRUE
  rebal_true<-which(rebal)
  pl<-numeric(nrow(returns))
  for(i in seq_along(rebal_true)){
    start_index<-rebal_true[i]
    end_index<-ifelse(is.na(rebal_true[i+1]-1), nrow(returns), rebal_true[i+1]-1)
    subdt<-dtab[start_index:end_index, tickers, with=F]
    subdt_cum<-apply(subdt, 2, cumprod)
    pl[start_index:end_index]<-rowSums(t(t(subdt_cum)*weights*start_cap))
    start_cap<-pl[end_index]
  }
  result<-data.table(Date=returns[,Date])
  result[,portfolio_pl:=pl]
  result
}

#undebug(portfolio_backtest)
spxt_100<-portfolio_backtest(returns = r_spxt_i14, tickers="spxt", weights=1, 
                             rebal_freq = "Yearly", start_cap = 1)
i14_9010<-portfolio_backtest(returns = r_spxt_i14, tickers = c("spxt", "i14"), 
                             weights = c(0.8, 0.2), rebal_freq = "Yearly", start_cap = 1)
spxt_100_imf<-portfolio_backtest(returns = r_spxt_imf, tickers="spxt", weights=1, 
                                 rebal_freq = "Yearly", start_cap = 1)
imf_9010<-portfolio_backtest(returns = r_spxt_imf, tickers = c("spxt", "imf"), 
                             weights = c(0.9, 0.1), rebal_freq = "Yearly", start_cap = 1)
summary_table(returns = spxt_100, time_unit_scaling = 260, convert_returns = T)
summary_table(returns = i14_9010, time_unit_scaling = 260, convert_returns = T)
summary_table(returns = spxt_100_imf, time_unit_scaling = 260, convert_returns = T)
summary_table(returns = imf_9010, time_unit_scaling = 260, convert_returns = T)

#Bootstrap
bootstrapped_dtab<-function(dtab, boot_cols, t=nrow(dtab), n=100, cum.prod=TRUE, convert_returns){
  returns<-dtab[, get(boot_cols)]
  if(convert_returns){
    if(class(returns)=="numeric"){
      returns<-c(1,(exp(diff(log(returns)))))
    }
    else{
      returns<-apply(returns, MARGIN = 2, FUN = function(x) c(1,(exp(diff(log(x))))))
    }
  }
  boot_matrix<-matrix(NA, nrow = t, ncol = n)
  boot_matrix<-sapply(1:n, function(i){
    returns[sample(1:length(returns), size = t, replace = T)]
  })
  if(cum.prod){
    boot_matrix<-apply(boot_matrix, 2, cumprod)
  }
  as.data.table(boot_matrix)
}
#debug(bootstrapped_dtab)
boot_spxt<-bootstrapped_dtab(dtab = spxt_100, boot_cols = "portfolio_pl", t = 6800, n = 10000, convert_returns = T)
boot_i14<-bootstrapped_dtab(dtab = i14_9010, boot_cols = "portfolio_pl", t = 6800, n = 10000, convert_returns = T)
boot_spxt_imf<-bootstrapped_dtab(dtab = spxt_100_imf, boot_cols = "portfolio_pl", t = 3800, n = 10000, convert_returns = T)
boot_imf<-bootstrapped_dtab(dtab = imf_9010, boot_cols = "portfolio_pl", t = 3800, n = 10000, convert_returns = T)

quantile(unlist(last(boot_spxt)), c(0.05, 0.5, 0.95))
quantile(unlist(last(boot_i14)), c(0.05, 0.5, 0.95))
quantile(unlist(last(boot_spxt_imf)), c(0.05, 0.5, 0.95))
quantile(unlist(last(boot_imf)), c(0.05, 0.5, 0.95))

plot_paths_boot<-function(boot_dt){
  boot_dt<-copy(boot_dt)
  terminals<-unlist(boot_dt[nrow(boot_dt),])
  q_5<-which.min(abs(terminals-quantile(terminals, 0.05)))
  q_50<-which.min(abs(terminals-quantile(terminals, 0.5)))
  q_95<-which.min(abs(terminals-quantile(terminals, 0.95)))
  q_paths<-c(q_5, q_50, q_95)
  q_paths<-c(q_5, q_50, q_95)
  color_vector<-c(rep("grey80", ncol(boot_dt)-3), "red4", "red", "red4")
  colorder<-1:ncol(boot_dt)
  colorder<-colorder[-(q_paths)]
  setcolorder(boot_dt, c(colorder, q_paths))
  index<-1:nrow(boot_dt)
  par(mar=c(5.1, 4.1, 2, 8.1), xpd=TRUE)
  plot(index, unlist(boot_dt[,1]), type="l", col="grey70", 
       xlab="Time", ylab="Wealth", ylim=c(0.7, max(terminals)))
  for(i in 2:ncol(boot_dt)){
    lines(index, unlist(boot_dt[,i, with=FALSE]), type="l", col=color_vector[i])
  }
  legend("topright", inset = c(-0.18, 0.5), legend=c("Paths","5th and/n95th Percentile", "Median"),
         lty=1, col=c("grey70", "red4", "red"), cex=0.7, bg = "white")
  par(xpd=FALSE)
  grid(lty = 1,      
       col = "gray90", 
       lwd = 0.5) 
}
#plot_paths_boot(boot_i14)
cagr<-function(terminal_value, n, ann=T, as_percentage=T){
  if(ann){
    cagr<-terminal_value^(260/n)
  }
  else{
    cagr<-terminal_value^(1/n)
  }
  if(as_percentage){
    cagr<-(cagr-1)*100
  }
  as.numeric(cagr)
}

terminal_cagr_spxt<-cagr(terminal_value = as.numeric(last(boot_spxt)), n = nrow(boot_spxt))
terminal_cagr_i14<-cagr(terminal_value = as.numeric(last(boot_i14)), n = nrow(boot_i14))
terminal_cagr_spxt_imf<-cagr(terminal_value = as.numeric(last(boot_spxt_imf)), n = nrow(boot_spxt_imf))
terminal_cagr_imf<-cagr(terminal_value = as.numeric(last(boot_imf)), n = nrow(boot_imf))

spxt_q<-c(quantile(terminal_cagr_spxt, c(0.05, 0.5, 0.95)))
IMF_q<-c(quantile(terminal_cagr_imf, c(0.05, 0.5, 0.95)))

hist_spxt<-hist(terminal_cagr_spxt, breaks  = 40, plot = F)
hist_iMF<-hist(terminal_cagr_imf, breaks  = 40, plot = F)
plot(hist_spxt, col=rgb(0,0.7,0,0.4), xlab="CAGR in %", freq=F, 
     main="Histogram of CAGRs", xlim=c(min(terminal_cagr_spxt), max(terminal_cagr_i14)),
     ylim=c(0, 0.12))
plot(hist_iMF, xaxt = 'n', yaxt = 'n', col = rgb(0,0,1,0.4), add = TRUE, freq = FALSE)
abline(v=spxt_q[1], col="black")
abline(v=IMF_q[1], col="black", lty=2)
abline(v=spxt_q[2], col="red")
abline(v=IMF_q[2], col="red", lty=2)
abline(v=spxt_q[3], col="black")
abline(v=IMF_q[3], col="black", lty=2)
abline(v=mean(terminal_cagr_imf), col="black", lty=2)
legend("topright", legend = c("SPXT", "90% SPXT + 10% i14",
                              paste("Median (SPXT)=", round(spxt_q[2], 2), "%"),
                              paste("Median (Portfolio)=", round(IMF_q[2], 2), "%"),
                              paste("95th-to-5th-Quantile Range (SPXT)=", round(spxt_q[3]-spxt_q[1], 2), "%"),
                              paste("95th-to-5th-Quantile Range (Portfolio)=", round(IMF_q[3]-IMF_q[1], 2), "%")),
       col=c(rgb(0,0.7,0,0.4), rgb(0,0,1,0.4), "red", "red", "black", "black"),
       lty=c(NA, NA, 1, 2, 1, 2), 
       pch=c(15, 15, NA, NA, NA, NA), cex=0.8)

quantile_comparison<-function(cagr_vector_list){
  quant_tab<-do.call(rbind, cagr_vector_list)
  #Column with dispersion as delta btw. extreme quantiles
  quant_tab<-cbind(quant_tab, Quantile_delta=quant_tab[,3]-quant_tab[,1])
  #Row with improvement delta
  #quant_tab<-rbind(quant_tab, Delta=quant_tab[2,]-quant_tab[1,])
  quant_tab
}


quantile_comparison(list(spxt_q, i14_q))







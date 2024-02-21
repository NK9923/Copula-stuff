# Copula

In this repository I deepened my knowledge about probability theory and the foundationas of various Copulas. Copulas are a flexible way to construct dependence strucutres and honestly look quite cool. Since C++ does not come with a lot of probabilistic functions, I had to implement some by my own. I wanted to make the overall project to a .dll in order to later use it in different projects with the namespace 'copula'. 

In detail I implemented the Gaussian copula, which is defined as:

$C_\Sigma (u_1,u_2) = \Phi_\Sigma(\Phi^{-1}(u_1),\Phi^{-1}(u_2))$

where $\Phi_R$ is the bivariate normal cdf with covariance matrix $\Sigma$ (equal to the correlation matrix), and $\Phi^{-1}$ is the quantile normal function. For the ordinary Gaussian copula I sampled from a multivariate normal distribution applied the distribution function of the standard normal and took the quantile

as well as the Frank copula

$\frac{-1.0}{\alpha} log \biggl[1.0 + exp(-sum) * (exp(-\alpha) - 1)) \biggl]$

## To-Do's

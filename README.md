# Copula

In this repository I deepened my knowledge about probability theory and the foundationas of various Copulas. Copulas are a flexible way to construct dependence strucutres. Since C++ does not come with a lot of probabilistic functions, I had to implement some by my own. I wanted to make the overall project to a .dll in order to later use it in different projects with the namespace 'copula'. The Statsfunctions class contains plenty of probabilistic functions and random number generation. Numerical Integration especially for the t-distribution was performed by the trapezoidal method. Since I am not a professional developer I expect quite natually that my implementations still feature some bugs and can be implemented in a more efficient manner. 

### Gaussian copula

In detail I implemented the Gaussian copula, which is defined as:

$C_\Sigma (u_1,u_2) = \Phi_\Sigma(\Phi^{-1}(u_1),\Phi^{-1}(u_2))$

where $\Phi_R$ is the bivariate normal cdf with covariance matrix $\Sigma$ (equal to the correlation matrix), and $\Phi^{-1}$ is the quantile normal function. For the ordinary Gaussian copula I sampled from a multivariate normal distribution applied the distribution function of the standard normal and took the quantile to generate random variables. I still need to implement some functions related to the Gaussian copula such as

### Frank copula

I also performed random number generation from the Frank-copula.

$\frac{-1.0}{\alpha} log \biggl[1.0 + exp(-sum) * (exp(-\alpha) - 1)) \biggl]$

### EVT copula and tail dependence

I was also quite interested in the Tail dependence coefficient obtainable by the generalized pareto distribution and Vine-copulas. 

## Dependencies

The plotting was performed with pybind11. Thus, if one would like to use this .dll one either needs to change the path to the python lib files as well as to the pybind11 Header-files, or alternatively remove the plotting class entirely and recompile the project. Other than that I used the Eigen library, which is header-only.  

## To-Do's

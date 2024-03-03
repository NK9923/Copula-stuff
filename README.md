# Copula

In this repository I deepened my knowledge about probability theory and the foundationas of various Copulas. Copulas are a flexible way to construct dependence strucutres. Since C++ does not come with a lot of probabilistic functions, I had to implement some by my own. I wanted to make the overall project to a .dll in order to later use it in different projects with the namespace 'copula'. The Statsfunctions class contains plenty of probabilistic functions and random number generation. Numerical Integration especially for the t-distribution was performed by the trapezoidal method. Since I am not a professional developer I expect quite natually that my implementations still feature some bugs and can be implemented in a more efficient manner. 

I was particularly interested in sampling from copulas. I focused on the following Copulas:

### Gaussian copula

In detail I implemented the Gaussian copula, which is defined as:

$C_\Sigma (u_1,u_2) = \Phi_\Sigma(\Phi^{-1}(u_1),\Phi^{-1}(u_2))$

where $\Phi_R$ is the bivariate normal cdf with covariance matrix $\Sigma$ (equal to the correlation matrix), and $\Phi^{-1}$ is the quantile normal function. For the ordinary Gaussian copula I sampled from a multivariate normal distribution applied the distribution function of the standard normal and took the quantile to generate random variables. I still need to implement some functions related to the Gaussian copula such as

### Student t-copula

I have implemented the Student t-copula. I generated random numbers from a multivariate t-distribution and applied the probability density function (PDF) of the standard t-distribution to obtain random variable samples from the copula. However, I suspect an error in the implementation, as I crosschecked my simulation results with those delivered by R, and the simulation appeared somewhat awkward.


### Clayton copula

For the Clayton copula I generated random uniform numbers and applied the inverse of the Clayton copula to obtain random variables. 

### Frank copula

I also performed random number generation from the Frank-copula.

$
\begin{align*}
&\text{double tmp} = -\frac{1}{a} \cdot \log1p\left(-\frac{random2 \cdot \expm1(-a)}{\left(\exp(-a \cdot random1) \cdot (random2 - 1)\right) - random2}\right); \\
&u[i] \sim \text{random1}; \\
&v[i] = (\text{copula.parameters[0]} > 0) ? (1 - \text{tmp}) : \text{tmp};
\end{align*}
$


### EVT copula and tail dependence

This is still a toDo. I would like to implement the extrem value copula and the tail dependence. I 

## Dependencies

The plotting was done with pybind11 and matplotlib. Thus, if one would like to use this .dll one either needs to change the path to the python lib files as well as to the pybind11 Header-files, or alternatively remove the plotting class entirely and recompile the project. Other than that I used the Eigen library, which is header-only. Thus one would only have to include the Header files.   

## To-Do's
